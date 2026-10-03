#include "SpirvBackend/SpirvWaveExchange.hpp"
#include "SpirvBackend/SpirvEmitterHelpers.hpp"
#include <spirv/unified1/spirv.hpp>
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace ShaderRecompiler {

namespace {

// Dwords per wave: two generations of WaveExchangeDwords from each half.
constexpr std::uint32_t WaveSlotDwords = 2u * 2u * WaveExchangeDwords;
static_assert(WaveSlotDwords == 8u, "SingleLaneExchangeDwords assumes eight dwords per wave");

std::uint32_t WaveCount(const SpirvEmitterState& state) {
    return (state.splitThreads + 63u) / 64u;
}

std::uint32_t ExchangePointer(SpirvEmitterState& state, std::uint32_t index) {
    const auto pointer = state.module.AllocateId();
    state.module.AddFunction(spv::OpAccessChain, TypeU32ElementPointer(state, spv::StorageClassWorkgroup), pointer, state.waveExchangeVariable, index);
    return pointer;
}

std::uint32_t Semantics(SpirvEmitterState& state, std::uint32_t order) {
    return ConstantU32(state, order | spv::MemorySemanticsWorkgroupMemoryMask);
}

void WorkgroupBarrier(SpirvEmitterState& state) {
    const auto workgroup = ConstantU32(state, spv::ScopeWorkgroup);
    state.module.AddFunction(spv::OpControlBarrier, workgroup, workgroup, Semantics(state, spv::MemorySemanticsAcquireReleaseMask));
}

// The counters' pairing: publishes generation `count` of this half (after its slot writes) and
// waits until the other half, if the workgroup has it, published it too.
void PairWithOtherHalf(SpirvEmitterState& state, std::uint32_t count) {
    const auto workgroup = ConstantU32(state, spv::ScopeWorkgroup);
    const auto subgroup = ConstantU32(state, spv::ScopeSubgroup);
    // Every lane of this half is past its workgroup memory accesses before any lane publishes: the
    // other half acquires from one lane's store, which then carries all the half's writes.
    state.module.AddFunction(spv::OpControlBarrier, subgroup, workgroup, Semantics(state, spv::MemorySemanticsAcquireReleaseMask));
    state.module.AddFunction(spv::OpAtomicStore, ExchangePointer(state, state.splitOwnArrival), workgroup, Semantics(state, spv::MemorySemanticsReleaseMask), count);
    const auto header = state.module.AllocateId();
    const auto cont = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto observed = state.module.AllocateId();
    state.module.AddFunction(spv::OpAtomicLoad, TypeU32(state), observed, ExchangePointer(state, state.splitOtherArrival), workgroup, Semantics(state, spv::MemorySemanticsAcquireMask));
    auto ready = Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), observed, count);
    if (state.splitOtherExists != 0) ready = Binary(state, spv::OpLogicalOr, TypeBool(state), ready, Unary(state, spv::OpLogicalNot, TypeBool(state), state.splitOtherExists));
    state.module.AddFunction(spv::OpLoopMerge, merge, cont, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, ready, merge, cont);
    EmitLabel(state, cont);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    // And no lane of this half goes on before all of them saw the other half arrive.
    state.module.AddFunction(spv::OpControlBarrier, subgroup, workgroup, Semantics(state, spv::MemorySemanticsAcquireReleaseMask));
}

// Advances the generation counter; returns the generation this exchange uses (counting from 1
// with counters, alternating 0/1 with barriers).
std::uint32_t NextGeneration(SpirvEmitterState& state) {
    const auto current = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, TypeU32(state), current, state.waveExchangeGeneration);
    const auto next = state.splitWaveCounters ? EmitBinaryU32(state, spv::OpIAdd, current, ConstantU32(state, 1u)) : EmitBinaryU32(state, spv::OpBitwiseXor, current, ConstantU32(state, 1u));
    state.module.AddFunction(spv::OpStore, state.waveExchangeGeneration, next);
    return state.splitWaveCounters ? next : current;
}

}

void PrepareWaveExchange(SpirvEmitterState& state, const ShaderWorkgroupInputInfo& workgroup) {
    state.splitWave = true;
    state.splitThreads = std::max(workgroup.threadsNum[0], 1u) * std::max(workgroup.threadsNum[1], 1u) * std::max(workgroup.threadsNum[2], 1u);
    state.splitWaveCounters = state.splitThreads > 64u;
    state.requirements.subgroupBallot = true;
    state.requirements.subgroupShuffle = true;
    state.requirements.subgroupLocalInvocationId = true;
    state.waveExchangeVariable = state.module.DefineGlobalVariable(TypeU32ArrayPointer(state, spv::StorageClassWorkgroup, SingleLaneExchangeDwords(state.splitThreads)), spv::StorageClassWorkgroup);
    state.module.AddName(state.waveExchangeVariable, "wave_exchange");
    state.waveExchangeGeneration = state.module.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, TypeU32(state)), spv::StorageClassPrivate);
    state.module.AddName(state.waveExchangeGeneration, "wave_exchange_generation");
}

void EmitWaveExchangeEntry(SpirvEmitterState& state) {
    const auto index = EmitLocalInvocationIndex(state);
    const auto wave = EmitBinaryU32(state, spv::OpShiftRightLogical, index, ConstantU32(state, 6u));
    const auto half = EmitBinaryU32(state, spv::OpBitwiseAnd, EmitBinaryU32(state, spv::OpShiftRightLogical, index, ConstantU32(state, 5u)), ConstantU32(state, 1u));
    const auto other = EmitBinaryU32(state, spv::OpBitwiseXor, half, ConstantU32(state, 1u));
    state.splitFirstHalf = Binary(state, spv::OpIEqual, TypeBool(state), half, ConstantU32(state, 0u));
    const auto base = EmitBinaryU32(state, spv::OpIMul, wave, ConstantU32(state, WaveSlotDwords));
    state.splitOwnSlot = EmitBinaryU32(state, spv::OpIAdd, base, EmitBinaryU32(state, spv::OpIMul, half, ConstantU32(state, WaveExchangeDwords)));
    state.splitOtherSlot = EmitBinaryU32(state, spv::OpIAdd, base, EmitBinaryU32(state, spv::OpIMul, other, ConstantU32(state, WaveExchangeDwords)));
    if (state.splitThreads % 64u != 0u) {
        // The first invocation index of the other half.
        const auto otherFirst = EmitBinaryU32(state, spv::OpBitwiseXor, EmitBinaryU32(state, spv::OpBitwiseAnd, index, ConstantU32(state, ~31u)), ConstantU32(state, 32u));
        state.splitOtherExists = Binary(state, spv::OpULessThan, TypeBool(state), otherFirst, ConstantU32(state, state.splitThreads));
    }
    state.module.AddFunction(spv::OpStore, state.waveExchangeGeneration, ConstantU32(state, 0u));
    if (state.splitWaveCounters) {
        const auto arrivals = EmitBinaryU32(state, spv::OpIAdd, ConstantU32(state, WaveCount(state) * WaveSlotDwords), EmitBinaryU32(state, spv::OpIMul, wave, ConstantU32(state, 2u)));
        state.splitOwnArrival = EmitBinaryU32(state, spv::OpIAdd, arrivals, half);
        state.splitOtherArrival = EmitBinaryU32(state, spv::OpIAdd, arrivals, other);
        // Workgroup memory starts undefined: each half clears its own counter before any half
        // waits on one (the entry is uniform over the workgroup).
        state.module.AddFunction(spv::OpAtomicStore, ExchangePointer(state, state.splitOwnArrival), ConstantU32(state, spv::ScopeWorkgroup), ConstantU32(state, spv::MemorySemanticsMaskNone), ConstantU32(state, 0u));
        WorkgroupBarrier(state);
    }
}

std::vector<std::array<std::uint32_t, 2>> EmitWaveExchange(SpirvEmitterState& state, std::span<const std::uint32_t> values) {
    if (!state.splitWave || values.empty() || values.size() > WaveExchangeDwords || state.conditionalDepth != 0) {
        throw std::runtime_error("SPIR-V emission failed: invalid wave exchange");
    }
    const auto generation = NextGeneration(state);
    const auto offset = EmitBinaryU32(state, spv::OpIMul, EmitBinaryU32(state, spv::OpBitwiseAnd, generation, ConstantU32(state, 1u)), ConstantU32(state, 2u * WaveExchangeDwords));
    const auto own = EmitBinaryU32(state, spv::OpIAdd, state.splitOwnSlot, offset);
    const auto other = EmitBinaryU32(state, spv::OpIAdd, state.splitOtherSlot, offset);
    // Every invocation of the half stores the same value (it is uniform over the half).
    for (std::size_t i = 0; i < values.size(); ++i) {
        state.module.AddFunction(spv::OpStore, ExchangePointer(state, EmitBinaryU32(state, spv::OpIAdd, own, ConstantU32(state, static_cast<std::uint32_t>(i)))), values[i]);
    }
    if (state.splitWaveCounters) PairWithOtherHalf(state, generation);
    else WorkgroupBarrier(state);
    std::vector<std::array<std::uint32_t, 2>> result;
    result.reserve(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        const auto loaded = state.module.AllocateId();
        state.module.AddFunction(spv::OpLoad, TypeU32(state), loaded, ExchangePointer(state, EmitBinaryU32(state, spv::OpIAdd, other, ConstantU32(state, static_cast<std::uint32_t>(i)))));
        const auto theirs = state.splitOtherExists != 0 ? Select(state, TypeU32(state), state.splitOtherExists, loaded, ConstantU32(state, 0u)) : loaded;
        result.push_back({Select(state, TypeU32(state), state.splitFirstHalf, values[i], theirs), Select(state, TypeU32(state), state.splitFirstHalf, theirs, values[i])});
    }
    return result;
}

void EmitWaveSync(SpirvEmitterState& state) {
    if (state.conditionalDepth != 0) throw std::runtime_error("SPIR-V emission failed: a wave sync under a conditional");
    if (state.splitWaveCounters) PairWithOtherHalf(state, NextGeneration(state));
    else WorkgroupBarrier(state);
}

std::uint32_t EmitSplitGuestLane(SpirvEmitterState& state) {
    return EmitBinaryU32(state, spv::OpBitwiseAnd, EmitLocalInvocationIndex(state), ConstantU32(state, 63u));
}

std::uint32_t EmitSplitFirstHalf(SpirvEmitterState& state) {
    return state.splitFirstHalf;
}

std::uint32_t EmitSplitBallotWord(SpirvEmitterState& state, std::uint32_t predicate) {
    const auto ballot = state.module.AllocateId();
    state.module.AddFunction(spv::OpGroupNonUniformBallot, TypeU32Vector(state, 4u), ballot, ConstantU32(state, spv::ScopeSubgroup), predicate);
    const auto word = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeExtract, TypeU32(state), word, ballot, 0u);
    return word;
}

std::uint32_t EmitSplitHalfBallot(SpirvEmitterState& state, std::uint32_t predicate) {
    const auto word = EmitSplitBallotWord(state, predicate);
    const auto zero = ConstantU32(state, 0u);
    const auto result = state.module.AllocateId();
    state.module.AddFunction(spv::OpCompositeConstruct, TypeU32Vector(state, 4u), result, Select(state, TypeU32(state), state.splitFirstHalf, word, zero), Select(state, TypeU32(state), state.splitFirstHalf, zero, word), zero, zero);
    return result;
}

}

namespace ShaderRecompiler {

namespace {

// Values that are the same in every lane of a wave whenever its lanes compute them together.
bool UniformSource(const IrValue& value) {
    switch (value.Opcode()) {
    case IrOpcode::Ballot:
    case IrOpcode::ReadFirstLane:
    case IrOpcode::ReadLane:
    case IrOpcode::GetUserData:
    case IrOpcode::GetShaderBase:
    case IrOpcode::ReadConst:
    case IrOpcode::ReadConstBuffer:
    case IrOpcode::MeshDrawParameter:
    case IrOpcode::MeshArgument:
    case IrOpcode::GetSrtResource:
    case IrOpcode::GetBufferResource:
    case IrOpcode::GetAddressResource:
    case IrOpcode::GetScratchResource:
    case IrOpcode::GetImageResource:
    case IrOpcode::GetSamplerResource:
        return true;
    default:
        return false;
    }
}

// Values that differ between lanes whatever their operands.
bool LaneSource(const IrValue& value) {
    switch (value.Opcode()) {
    case IrOpcode::LaneId:
    case IrOpcode::GetAttribute:
    case IrOpcode::GetInterpolationParameter:
    case IrOpcode::DppMoveU32:
    case IrOpcode::DppUpdateU32:
    case IrOpcode::Permlane16U32:
    case IrOpcode::BpermuteU32:
    case IrOpcode::SwizzleU32:
    case IrOpcode::WriteLane:
    case IrOpcode::DataAppend:
    case IrOpcode::DataConsume:
        return true;
    case IrOpcode::GetBuiltin: {
        const auto* kind = value.Argument(0)->Resolve();
        return kind == nullptr || !kind->HasImmediate() || static_cast<StageInputKind>(kind->ImmediateU32()) != StageInputKind::WorkgroupId;
    }
    default:
        // Memory a lane loads: its address or the memory may differ by lane.
        return BufferAccessOf(value.Opcode()) != BufferAccess::None || SharedAccessOf(value.Opcode()) != SharedAccess::None || AddressOpcodeInfoOf(value.Opcode()).access != AddressAccess::None || ImageOpcodeInfoOf(value.Opcode()).access != ImageAccess::None;
    }
}

}

namespace {

std::unordered_map<const IrValue*, bool> LaneVaryingValues(const IrProgram& program, bool scalarLoadsUniform) {
    std::unordered_map<const IrValue*, bool> varying;
    bool changed = true;
    const auto isVarying = [&](const IrValue* value) {
        const auto* resolved = value->Resolve();
        if (resolved == nullptr || resolved->HasImmediate()) return false;
        const auto found = varying.find(resolved);
        return found != varying.end() && found->second;
    };
    const auto scalarLoad = [&](const IrValue& value) {
        if (!scalarLoadsUniform || AddressOpcodeInfoOf(value.Opcode()).access != AddressAccess::Read) return false;
        const auto index = value.Flags<MemoryFlags>().index;
        const auto& memory = program.Resources().memoryInfo;
        return index < memory.size() && memory[index].kind == ResourceKind::ScalarAddress;
    };
    while (changed) {
        changed = false;
        for (const auto* block : program.BlockOrder()) {
            for (const auto* inst : block->Instructions()) {
                bool lanes = false;
                if (UniformSource(*inst)) lanes = false;
                else if (LaneSource(*inst) && !scalarLoad(*inst)) lanes = true;
                else {
                    for (std::size_t i = 0; i < inst->ArgumentCount() && !lanes; ++i) {
                        if (inst->Argument(i) != nullptr) lanes = isVarying(inst->Argument(i));
                    }
                }
                auto& entry = varying[inst];
                if (lanes && !entry) {
                    entry = true;
                    changed = true;
                }
            }
        }
    }
    return varying;
}

}

std::vector<const BlockInfo*> LaneVaryingScalarBranches(const IrProgram& program) {
    const auto varying = LaneVaryingValues(program, false);
    const auto isVarying = [&](const IrValue* value) {
        const auto* resolved = value->Resolve();
        if (resolved == nullptr || resolved->HasImmediate()) return false;
        const auto found = varying.find(resolved);
        return found != varying.end() && found->second;
    };
    std::vector<const BlockInfo*> result;
    for (const auto& info : program.Metadata().blockInfo) {
        const auto kind = info.terminator.condition;
        if (info.terminator.kind != TerminatorKind::ConditionalBranch || info.condition == nullptr) continue;
        if (kind == BranchCondition::ExecZero || kind == BranchCondition::ExecNonZero || kind == BranchCondition::VccZero || kind == BranchCondition::VccNonZero) continue;
        if (isVarying(info.condition)) result.push_back(&info);
    }
    return result;
}

}
