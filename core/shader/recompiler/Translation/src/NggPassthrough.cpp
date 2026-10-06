#include "Translation/NggPassthrough.hpp"

#include "ControlFlow/GraphBuilder.hpp"
#include "ControlFlow/Structurizer.hpp"
#include "IntermediateRepresentation/IrMetadata.hpp"
#include "Optimization/ConstantFolder.hpp"
#include "Optimization/DeadCodeEliminator.hpp"
#include "Optimization/SsaBuilder.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include "Translation/InstructionTranslator.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <exception>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ShaderRecompiler {

namespace {

constexpr std::uint64_t MaxContexts = 3'000'000u;
constexpr std::uint32_t MaxCleanBools = 4u;
constexpr int VectorZeroMarker = static_cast<int>(SubgroupMarkerVectorOffset);

struct SubgroupContext {
    bool host;
    std::uint32_t lane;
    std::uint32_t vertices;
    std::uint32_t primitives;
    std::uint32_t wave;
    std::uint32_t waves;
    std::uint32_t groupVertices;
    std::uint32_t groupPrimitives;
};

using Word = std::array<std::uint64_t, 4>;

int markerOf(const IrValue* value) {
    if (value->Opcode() != IrOpcode::GetUserData || value->ArgumentCount() != 1u) return -1;
    const auto reg = value->Argument(0)->Register();
    if (reg.bank != RegisterBank::UserData || reg.index < SubgroupMarkerFirstRegister) return -1;
    return static_cast<int>(reg.index - SubgroupMarkerFirstRegister);
}

bool forbiddenOpcode(IrOpcode opcode) {
    const std::string_view name = IrOpcodeName(opcode);
    for (const std::string_view part : {"Store", "Atomic", "Shared", "Lane", "Ballot", "Dpp", "Permlane", "Bpermute", "PermuteU32", "Wqm", "Gds", "Discard", "EmitVertex", "WriteImage", "ShaderClock", "RealtimeClock", "Barrier"}) {
        if (name.find(part) != std::string_view::npos && opcode != IrOpcode::LaneId) return true;
    }
    return false;
}

std::uint64_t mask(IrType type, std::uint64_t value) {
    if (type == IrType::Bool) return value != 0u ? 1u : 0u;
    if (type == IrType::U64 || type == IrType::S64) return value;
    return value & 0xffffffffu;
}

std::uint64_t signExtend32(std::uint64_t value) {
    return static_cast<std::uint64_t>(static_cast<std::int64_t>(static_cast<std::int32_t>(static_cast<std::uint32_t>(value))));
}

class Analysis {
public:
    Analysis(IrProgram& program, const NggSubgroupLimits& limits) : program(program), limits(limits) {}

    std::optional<std::string> Run() {
        std::vector<IrValue*> values;
        for (const auto& block : program.Blocks()) {
            for (IrValue* value : block->Instructions()) {
                if (value != nullptr && !value->IsEmpty()) values.push_back(value);
            }
        }
        for (IrValue* value : values) {
            if (forbiddenOpcode(value->Opcode())) return std::string("uses ") + std::string(IrOpcodeName(value->Opcode()));
            if (value->Opcode() == IrOpcode::Loop || value->Opcode() == IrOpcode::LoopMerge) return std::string("contains a loop");
        }
        if (auto failure = buildContexts()) return failure;
        for (IrValue* value : values) classify(value);
        std::uint32_t primitiveExports = 0;
        for (IrValue* value : values) {
            const auto opcode = value->Opcode();
            if (opcode == IrOpcode::SetAttribute) {
                const auto& info = program.Metadata().exportInfo.at(value->Flags<ExportFlags>().index);
                if (info.kind == ExportTargetKind::Primitive) {
                    if (auto failure = checkPrimitiveExport(*value)) return failure;
                    primitiveExports++;
                    continue;
                }
                for (std::size_t i = 0; i < value->ArgumentCount(); i++) {
                    if (dependent(value->Argument(i))) return std::string("an export depends on the subgroup or another lane");
                }
                continue;
            }
            if (opcode == IrOpcode::Reference || opcode == IrOpcode::BranchConditional) {
                for (std::size_t i = 0; i < value->ArgumentCount(); i++) {
                    if (dependent(value->Argument(i))) return std::string("a branch depends on the subgroup or another lane");
                }
                continue;
            }
            if (opcode == IrOpcode::MeshAllocate || opcode == IrOpcode::Sendmsg) continue;
            if (value->MayHaveSideEffects()) {
                for (std::size_t i = 0; i < value->ArgumentCount(); i++) {
                    if (dependent(value->Argument(i))) return std::string(IrOpcodeName(opcode)) + " depends on the subgroup or another lane";
                }
            }
        }
        if (primitiveExports != 1u) return std::string("exports ") + std::to_string(primitiveExports) + " primitives instead of one";
        return std::nullopt;
    }

private:
    struct State {
        bool dependent = false;
        bool resolved = false;
        std::optional<bool> constant;
    };

    IrProgram& program;
    NggSubgroupLimits limits;
    std::unordered_map<const IrValue*, State> states;
    std::vector<SubgroupContext> vertexContexts;
    std::vector<SubgroupContext> laneContexts;

    std::optional<std::string> buildContexts() {
        const auto wave = limits.waveSize;
        if ((wave != 32u && wave != 64u) || limits.vertices == 0u || limits.primitives == 0u || limits.vertices > 256u || limits.primitives > 256u) return std::string("invalid subgroup limits");
        const std::uint64_t estimate = static_cast<std::uint64_t>(limits.vertices) * (limits.primitives + 1u) * ((std::max(limits.vertices, limits.primitives) + wave - 1u) / wave) * wave;
        if (estimate > MaxContexts) return std::string("subgroup limits too large to verify");
        for (std::uint32_t gv = 1; gv <= limits.vertices; gv++) {
            for (std::uint32_t gp = 1; gp <= limits.primitives; gp++) {
                const auto waves = (std::max(gv, gp) + wave - 1u) / wave;
                for (std::uint32_t k = 0; k < waves; k++) {
                    const auto nv = std::min(gv > k * wave ? gv - k * wave : 0u, wave);
                    const auto np = std::min(gp > k * wave ? gp - k * wave : 0u, wave);
                    for (std::uint32_t lane = 0; lane < wave; lane++) {
                        const SubgroupContext context{false, lane, nv, np, k, waves, gv, gp};
                        laneContexts.push_back(context);
                        if (lane < nv) vertexContexts.push_back(context);
                    }
                }
            }
        }
        for (std::uint32_t lane = 0; lane < wave; lane++) vertexContexts.push_back({true, lane, wave, wave, 0u, 1u, wave, wave});
        return std::nullopt;
    }

    bool dependent(const IrValue* value) {
        const auto it = states.find(value->Resolve());
        return it != states.end() && it->second.dependent && !it->second.resolved;
    }

    std::optional<bool> constantOf(const IrValue* value) {
        const auto* resolved = value->Resolve();
        if (resolved->HasImmediate() && resolved->Type() == IrType::Bool) return resolved->ImmediateBool();
        const auto it = states.find(resolved);
        if (it != states.end() && it->second.resolved) return it->second.constant;
        return std::nullopt;
    }

    void classify(IrValue* value) {
        auto& state = states[value];
        if (state.resolved) return;
        const auto opcode = value->Opcode();
        if (markerOf(value) >= 0 || opcode == IrOpcode::LaneId) {
            state.dependent = true;
            return;
        }
        bool any = false;
        const auto select = opcode == IrOpcode::Select || opcode == IrOpcode::SelectU32 || opcode == IrOpcode::SelectU1 || opcode == IrOpcode::SelectF32;
        if (select && value->ArgumentCount() == 3u) {
            if (const auto condition = constantOf(value->Argument(0))) {
                state.dependent = dependent(value->Argument(*condition ? 1u : 2u));
                return;
            }
        }
        if ((opcode == IrOpcode::LogicalAnd || opcode == IrOpcode::LogicalOr) && value->ArgumentCount() == 2u) {
            for (std::size_t i = 0; i < 2u; i++) {
                const auto constant = constantOf(value->Argument(i));
                if (constant && *constant == (opcode == IrOpcode::LogicalOr)) {
                    state.dependent = true;
                    state.resolved = true;
                    state.constant = *constant;
                    return;
                }
            }
        }
        for (std::size_t i = 0; i < value->ArgumentCount(); i++) any = any || dependent(value->Argument(i));
        state.dependent = any;
        if (any && value->Type() == IrType::Bool && !value->IsPhi()) resolve(value, state);
    }

    enum class NodeKind : std::uint8_t { Fixed, Clean, Lane, GroupInfo, WaveInfo, Operation };

    struct Node {
        NodeKind kind = NodeKind::Fixed;
        IrOpcode opcode = IrOpcode::Void;
        IrType type = IrType::Void;
        std::array<std::uint32_t, 4> args{};
        Word fixed{};
        std::uint32_t clean = 0;
    };

    struct Evaluation {
        std::vector<Node> nodes;
        std::unordered_map<const IrValue*, std::uint32_t> slots;
        std::vector<const IrValue*> cleanBools;
    };

    static bool supported(IrOpcode opcode) {
        switch (opcode) {
        case IrOpcode::IAdd32: case IrOpcode::IAdd64: case IrOpcode::ISub32: case IrOpcode::ISub64: case IrOpcode::IMul32: case IrOpcode::IMul64:
        case IrOpcode::BitwiseAnd32: case IrOpcode::BitwiseAnd64: case IrOpcode::BitwiseOr32: case IrOpcode::BitwiseXor32: case IrOpcode::BitwiseNot32:
        case IrOpcode::ShiftLeftLogical32: case IrOpcode::ShiftRightLogical32: case IrOpcode::ShiftRightArithmetic32: case IrOpcode::ShiftLeftLogical64: case IrOpcode::ShiftRightLogical64:
        case IrOpcode::BitFieldUExtract: case IrOpcode::BitCount32: case IrOpcode::BitCount64: case IrOpcode::UMin32: case IrOpcode::UMax32: case IrOpcode::SMin32: case IrOpcode::SMax32:
        case IrOpcode::IEqual32: case IrOpcode::IEqual64: case IrOpcode::INotEqual32: case IrOpcode::INotEqual64: case IrOpcode::ULessThan32: case IrOpcode::ULessThan64:
        case IrOpcode::ULessThanEqual32: case IrOpcode::UGreaterThan32: case IrOpcode::UGreaterThan64: case IrOpcode::UGreaterThanEqual32:
        case IrOpcode::SLessThan32: case IrOpcode::SLessThanEqual32: case IrOpcode::SGreaterThan32: case IrOpcode::SGreaterThanEqual32:
        case IrOpcode::LogicalAnd: case IrOpcode::LogicalOr: case IrOpcode::LogicalXor: case IrOpcode::LogicalNot:
        case IrOpcode::Select: case IrOpcode::SelectU32: case IrOpcode::SelectU1:
        case IrOpcode::CompositeConstructU64: case IrOpcode::CompositeExtractU64: case IrOpcode::CompositeConstructU32x2: case IrOpcode::CompositeConstructU32x3: case IrOpcode::CompositeConstructU32x4:
        case IrOpcode::CompositeExtractU32x2: case IrOpcode::CompositeExtractU32x3: case IrOpcode::CompositeExtractU32x4: case IrOpcode::IAddCarry32:
            return true;
        default:
            return false;
        }
    }

    std::optional<std::uint32_t> collect(const IrValue* value, Evaluation& evaluation) {
        value = value->Resolve();
        if (const auto found = evaluation.slots.find(value); found != evaluation.slots.end()) return found->second;
        Node node;
        node.type = value->Type();
        const auto it = states.find(value);
        const bool isDependent = it != states.end() && it->second.dependent && !it->second.resolved;
        const int marker = markerOf(value);
        if (!isDependent) {
            if (value->HasImmediate()) {
                node.fixed[0] = value->Type() == IrType::Bool ? (value->ImmediateBool() ? 1u : 0u) : value->Type() == IrType::U64 || value->Type() == IrType::S64 ? value->ImmediateU64() : value->ImmediateU32();
            } else if (value->Type() != IrType::Bool) {
                return std::nullopt;
            } else if (const auto constant = constantOf(value)) {
                node.fixed[0] = *constant ? 1u : 0u;
            } else {
                if (evaluation.cleanBools.size() >= MaxCleanBools) return std::nullopt;
                node.kind = NodeKind::Clean;
                node.clean = static_cast<std::uint32_t>(evaluation.cleanBools.size());
                evaluation.cleanBools.push_back(value);
            }
        } else if (value->Opcode() == IrOpcode::LaneId) {
            node.kind = NodeKind::Lane;
        } else if (marker == 2 || marker == 3) {
            node.kind = marker == 2 ? NodeKind::GroupInfo : NodeKind::WaveInfo;
        } else if (marker >= 0 || value->IsPhi() || !supported(value->Opcode()) || value->ArgumentCount() > 4u) {
            return std::nullopt;
        } else {
            node.kind = NodeKind::Operation;
            node.opcode = value->Opcode();
            for (std::size_t i = 0; i < value->ArgumentCount(); i++) {
                const auto slot = collect(value->Argument(i), evaluation);
                if (!slot) return std::nullopt;
                node.args[i] = *slot;
            }
        }
        const auto slot = static_cast<std::uint32_t>(evaluation.nodes.size());
        evaluation.nodes.push_back(node);
        evaluation.slots.emplace(value, slot);
        return slot;
    }

    static std::optional<Word> evaluate(const Evaluation& evaluation, const SubgroupContext& context, std::uint32_t assignment, std::vector<Word>& scratch) {
        scratch.resize(evaluation.nodes.size());
        for (std::size_t index = 0; index < evaluation.nodes.size(); index++) {
            const Node& node = evaluation.nodes[index];
            Word result{};
            switch (node.kind) {
            case NodeKind::Fixed: result = node.fixed; break;
            case NodeKind::Clean: result[0] = (assignment >> node.clean) & 1u; break;
            case NodeKind::Lane: result[0] = context.lane; break;
            case NodeKind::GroupInfo: result[0] = (static_cast<std::uint64_t>(context.groupVertices) << 12u) | (static_cast<std::uint64_t>(context.groupPrimitives) << 22u); break;
            case NodeKind::WaveInfo: result[0] = context.vertices | (context.primitives << 8u) | (context.wave << 24u) | (static_cast<std::uint64_t>(context.waves) << 28u); break;
            case NodeKind::Operation: {
                const auto arg = [&](std::size_t i) -> const Word& { return scratch[node.args[i]]; };
                const auto a = [&](std::size_t i) { return arg(i)[0]; };
                const auto type = node.type;
                const auto shift = [&](std::uint64_t amount, std::uint32_t width) -> std::optional<std::uint64_t> {
                    if (context.host && width == 32u && amount >= width) return std::nullopt;
                    return amount & (width - 1u);
                };
                switch (node.opcode) {
                case IrOpcode::IAdd32: case IrOpcode::IAdd64: result[0] = mask(type, a(0) + a(1)); break;
                case IrOpcode::ISub32: case IrOpcode::ISub64: result[0] = mask(type, a(0) - a(1)); break;
                case IrOpcode::IMul32: case IrOpcode::IMul64: result[0] = mask(type, a(0) * a(1)); break;
                case IrOpcode::BitwiseAnd32: case IrOpcode::BitwiseAnd64: result[0] = a(0) & a(1); break;
                case IrOpcode::BitwiseOr32: result[0] = a(0) | a(1); break;
                case IrOpcode::BitwiseXor32: result[0] = a(0) ^ a(1); break;
                case IrOpcode::BitwiseNot32: result[0] = mask(type, ~a(0)); break;
                case IrOpcode::ShiftLeftLogical32: case IrOpcode::ShiftRightLogical32: case IrOpcode::ShiftRightArithmetic32: {
                    const auto amount = shift(a(1), 32u);
                    if (!amount) return std::nullopt;
                    result[0] = node.opcode == IrOpcode::ShiftLeftLogical32 ? mask(type, a(0) << *amount) : node.opcode == IrOpcode::ShiftRightLogical32 ? mask(type, a(0)) >> *amount : mask(type, signExtend32(a(0)) >> *amount);
                    break;
                }
                case IrOpcode::ShiftLeftLogical64: case IrOpcode::ShiftRightLogical64: {
                    const auto amount = shift(a(1), 64u);
                    if (!amount) return std::nullopt;
                    result[0] = node.opcode == IrOpcode::ShiftLeftLogical64 ? a(0) << *amount : a(0) >> *amount;
                    break;
                }
                case IrOpcode::BitFieldUExtract: {
                    const auto offset = a(1);
                    const auto count = a(2);
                    if (offset > 31u || count > 32u || offset + count > 32u) return std::nullopt;
                    result[0] = count == 0u ? 0u : (a(0) >> offset) & ((1ull << count) - 1u);
                    break;
                }
                case IrOpcode::BitCount32: case IrOpcode::BitCount64: result[0] = static_cast<std::uint64_t>(std::popcount(a(0))); break;
                case IrOpcode::UMin32: result[0] = std::min(a(0), a(1)); break;
                case IrOpcode::UMax32: result[0] = std::max(a(0), a(1)); break;
                case IrOpcode::SMin32: result[0] = mask(type, static_cast<std::uint64_t>(std::min(static_cast<std::int64_t>(signExtend32(a(0))), static_cast<std::int64_t>(signExtend32(a(1)))))); break;
                case IrOpcode::SMax32: result[0] = mask(type, static_cast<std::uint64_t>(std::max(static_cast<std::int64_t>(signExtend32(a(0))), static_cast<std::int64_t>(signExtend32(a(1)))))); break;
                case IrOpcode::IEqual32: case IrOpcode::IEqual64: result[0] = a(0) == a(1); break;
                case IrOpcode::INotEqual32: case IrOpcode::INotEqual64: result[0] = a(0) != a(1); break;
                case IrOpcode::ULessThan32: case IrOpcode::ULessThan64: result[0] = a(0) < a(1); break;
                case IrOpcode::ULessThanEqual32: result[0] = a(0) <= a(1); break;
                case IrOpcode::UGreaterThan32: case IrOpcode::UGreaterThan64: result[0] = a(0) > a(1); break;
                case IrOpcode::UGreaterThanEqual32: result[0] = a(0) >= a(1); break;
                case IrOpcode::SLessThan32: result[0] = static_cast<std::int64_t>(signExtend32(a(0))) < static_cast<std::int64_t>(signExtend32(a(1))); break;
                case IrOpcode::SLessThanEqual32: result[0] = static_cast<std::int64_t>(signExtend32(a(0))) <= static_cast<std::int64_t>(signExtend32(a(1))); break;
                case IrOpcode::SGreaterThan32: result[0] = static_cast<std::int64_t>(signExtend32(a(0))) > static_cast<std::int64_t>(signExtend32(a(1))); break;
                case IrOpcode::SGreaterThanEqual32: result[0] = static_cast<std::int64_t>(signExtend32(a(0))) >= static_cast<std::int64_t>(signExtend32(a(1))); break;
                case IrOpcode::LogicalAnd: result[0] = (a(0) != 0u) && (a(1) != 0u); break;
                case IrOpcode::LogicalOr: result[0] = (a(0) != 0u) || (a(1) != 0u); break;
                case IrOpcode::LogicalXor: result[0] = (a(0) != 0u) != (a(1) != 0u); break;
                case IrOpcode::LogicalNot: result[0] = a(0) == 0u; break;
                case IrOpcode::Select: case IrOpcode::SelectU32: case IrOpcode::SelectU1: result = a(0) != 0u ? arg(1) : arg(2); break;
                case IrOpcode::CompositeConstructU64: result[0] = (a(0) & 0xffffffffu) | (a(1) << 32u); break;
                case IrOpcode::CompositeExtractU64: result[0] = a(1) == 0u ? a(0) & 0xffffffffu : a(0) >> 32u; break;
                case IrOpcode::CompositeConstructU32x2: result = {a(0), a(1)}; break;
                case IrOpcode::CompositeConstructU32x3: result = {a(0), a(1), a(2)}; break;
                case IrOpcode::CompositeConstructU32x4: result = {a(0), a(1), a(2), a(3)}; break;
                case IrOpcode::CompositeExtractU32x2: case IrOpcode::CompositeExtractU32x3: case IrOpcode::CompositeExtractU32x4: result[0] = arg(0)[a(1) & 3u]; break;
                case IrOpcode::IAddCarry32: {
                    const auto sum = a(0) + a(1);
                    result = {sum & 0xffffffffu, sum >> 32u};
                    break;
                }
                default:
                    return std::nullopt;
                }
                break;
            }
            }
            scratch[index] = result;
        }
        return scratch.back();
    }

    std::string signature(const Evaluation& evaluation) const {
        std::string key;
        key.reserve(16u + evaluation.nodes.size() * sizeof(Node));
        const auto append = [&key](const auto& field) { key.append(reinterpret_cast<const char*>(&field), sizeof(field)); };
        append(limits.waveSize);
        append(limits.vertices);
        append(limits.primitives);
        for (const Node& node : evaluation.nodes) {
            append(node.kind);
            append(node.opcode);
            append(node.type);
            append(node.args);
            append(node.fixed);
            append(node.clean);
        }
        return key;
    }

    void resolve(IrValue* value, State& state) {
        Evaluation evaluation;
        if (!collect(value, evaluation)) return;
        struct Resolution {
            bool resolved;
            std::optional<bool> constant;
        };
        static std::mutex cacheMutex;
        static std::unordered_map<std::string, Resolution> cache;
        const auto key = signature(evaluation);
        {
            std::lock_guard lock(cacheMutex);
            if (const auto found = cache.find(key); found != cache.end()) {
                state.resolved = found->second.resolved;
                state.constant = found->second.constant;
                return;
            }
        }
        const auto remember = [&](bool resolved, std::optional<bool> constant) {
            std::lock_guard lock(cacheMutex);
            cache.emplace(key, Resolution{resolved, constant});
            state.resolved = resolved;
            state.constant = constant;
        };
        std::vector<Word> scratch;
        std::optional<bool> constant;
        bool first = true;
        for (std::uint32_t assignment = 0; assignment < (1u << evaluation.cleanBools.size()); assignment++) {
            std::optional<std::uint64_t> reference;
            for (const auto& context : vertexContexts) {
                const auto word = evaluate(evaluation, context, assignment, scratch);
                if (!word || (reference && *reference != (*word)[0])) {
                    remember(false, std::nullopt);
                    return;
                }
                if (!reference) reference = (*word)[0];
            }
            const bool bit = reference.value_or(0u) != 0u;
            if (first) constant = bit;
            else if (constant && *constant != bit) constant.reset();
            first = false;
        }
        remember(true, constant);
    }

    std::optional<std::string> checkPrimitiveExport(const IrValue& exportValue) {
        if (exportValue.ArgumentCount() != 2u) return std::string("malformed primitive export");
        const IrValue* data = exportValue.Argument(0)->Resolve();
        if (data->Opcode() != IrOpcode::CompositeConstructU32x4 || markerOf(data->Argument(0)->Resolve()) != VectorZeroMarker) return std::string("exports a primitive other than the one it received");
        Evaluation evaluation;
        if (!collect(exportValue.Argument(1), evaluation) || !evaluation.cleanBools.empty()) return std::string("the primitive export mask is not a subgroup function");
        std::vector<Word> scratch;
        for (const auto& context : laneContexts) {
            if (context.lane >= context.primitives) continue;
            const auto word = evaluate(evaluation, context, 0u, scratch);
            if (!word || (*word)[0] == 0u) return std::string("does not export every primitive of the subgroup");
        }
        return std::nullopt;
    }
};

}

std::optional<std::string> NggPassthroughSubgroupDependence(std::span<const std::uint32_t> code, std::uint32_t userDataCount, const NggSubgroupLimits& limits) {
    try {
        constexpr RdnaInstructionDecoder decoder;
        const auto decoded = decoder.Decode(code);
        for (const auto& instruction : decoded.instructions) {
            if (instruction.family == RdnaInstructionFamily::DS) return std::string("uses LDS or GDS");
        }
        constexpr GraphBuilder graphBuilder;
        auto cfg = graphBuilder.Build(decoded);
        constexpr Structurizer structurizer;
        structurizer.Structurize(cfg);
        ShaderVertexInputInfo vertex{};
        TranslateOptions options{};
        options.stage = ShaderStageKind::Vertex;
        options.waveSize = limits.waveSize;
        options.userDataBaseRegister = 8u;
        options.userDataCount = std::min(userDataCount, NumScalarRegs - 8u);
        options.inputInfo.vertex = &vertex;
        options.subgroupContextMarkers = true;
        constexpr InstructionTranslator translator;
        auto program = translator.Translate(decoded, cfg, options);
        constexpr SsaBuilder ssaBuilder;
        ssaBuilder.Rewrite(program);
        constexpr ConstantFolder constantFolder;
        constantFolder.Fold(program);
        ResolveControlFlowIdentities(program);
        constexpr DeadCodeEliminator deadCodeEliminator;
        deadCodeEliminator.RemoveIdentities(program);
        deadCodeEliminator.Eliminate(program);
        return Analysis(program, limits).Run();
    } catch (const std::exception& error) {
        return std::string("cannot be analysed: ") + error.what();
    }
}

}
