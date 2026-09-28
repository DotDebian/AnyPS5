#include "ControlFlow/GraphBuilder.hpp"
#include "ControlFlow/RequestSerializer.hpp"
#include "ControlFlow/Structurizer.hpp"
#include "RdnaDecoder/RdnaInstructionDecoder.hpp"
#include "Recompiler.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <exception>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ShaderRecompiler;

struct Program {
    std::string_view name;
    std::string_view source;
    std::vector<std::uint32_t> code;
    std::size_t minimumAddedBlocks;
};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<std::uint32_t> constructBlocks(const ControlFlowGraph& graph, std::uint32_t header, std::uint32_t exclude) {
    std::vector<std::uint32_t> blocks;
    for (const auto& block : graph.blocks) {
        if (graph.Dominates(header, block.id) && (exclude == InvalidControlFlowId || !graph.Dominates(exclude, block.id))) {
            blocks.push_back(block.id);
        }
    }
    return blocks;
}

bool contains(const std::vector<std::uint32_t>& values, std::uint32_t value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

void verifyStructured(const Program& program, const ControlFlowGraph& graph) {
    const std::string prefix = std::string(program.name) + ": ";
    std::map<std::uint32_t, std::uint32_t> mergeOwners;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> loopExits;
    for (const auto& block : graph.blocks) {
        const auto& terminator = block.terminator;
        if (terminator.mergeBlock == InvalidControlFlowId) continue;
        require(mergeOwners.emplace(terminator.mergeBlock, block.id).second, prefix + "block " + std::to_string(terminator.mergeBlock) + " merges two constructs");
        if (terminator.loopHeader) loopExits.emplace_back(terminator.mergeBlock, terminator.continueBlock);
    }
    for (const auto& block : graph.blocks) {
        const auto& terminator = block.terminator;
        if (terminator.loopHeader || terminator.mergeBlock == InvalidControlFlowId) continue;
        for (const auto member : constructBlocks(graph, block.id, terminator.mergeBlock)) {
            for (const auto successor : graph.FindBlock(member).successors) {
                if (successor == terminator.mergeBlock || (graph.Dominates(block.id, successor) && !graph.Dominates(terminator.mergeBlock, successor))) continue;
                const bool loopExit = std::any_of(loopExits.begin(), loopExits.end(), [&](const auto& exits) { return successor == exits.first || successor == exits.second; });
                require(loopExit, prefix + "block " + std::to_string(member) + " of the selection at block " + std::to_string(block.id) + " branches to block " + std::to_string(successor) + " outside the construct");
            }
        }
    }
}

std::uint32_t skipEmptyBranches(const ControlFlowGraph& graph, std::uint32_t blockId) {
    for (std::size_t steps = 0; steps <= graph.blocks.size(); ++steps) {
        const auto& block = graph.FindBlock(blockId);
        if (block.instructionBegin != block.instructionEnd || block.terminator.kind != TerminatorKind::Branch) return blockId;
        blockId = block.terminator.trueBlock;
    }
    throw std::runtime_error("a cycle of empty blocks");
}

void verifySameExecutions(const Program& program, const ControlFlowGraph& original, const ControlFlowGraph& structured) {
    const std::string prefix = std::string(program.name) + ": ";
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pending{{structured.blocks.front().id, original.entryBlock}};
    std::vector<std::pair<std::uint32_t, std::uint32_t>> visited;
    while (!pending.empty()) {
        const auto [structuredId, originalId] = pending.back();
        pending.pop_back();
        const auto pair = std::make_pair(skipEmptyBranches(structured, structuredId), skipEmptyBranches(original, originalId));
        if (std::find(visited.begin(), visited.end(), pair) != visited.end()) continue;
        visited.push_back(pair);
        const auto& copy = structured.FindBlock(pair.first);
        const auto& source = original.FindBlock(pair.second);
        require(copy.instructionBegin == source.instructionBegin && copy.instructionEnd == source.instructionEnd, prefix + "block " + std::to_string(copy.id) + " runs other instructions than block " + std::to_string(source.id));
        require(copy.terminator.kind == source.terminator.kind && copy.terminator.condition == source.terminator.condition, prefix + "block " + std::to_string(copy.id) + " branches differently from block " + std::to_string(source.id));
        if (source.terminator.kind == TerminatorKind::Branch || source.terminator.kind == TerminatorKind::ConditionalBranch) pending.emplace_back(copy.terminator.trueBlock, source.terminator.trueBlock);
        if (source.terminator.kind == TerminatorKind::ConditionalBranch) pending.emplace_back(copy.terminator.falseBlock, source.terminator.falseBlock);
    }
}

std::vector<std::uint32_t> Store(std::vector<std::uint32_t> code) {
    code.insert(code.end(), {0xe0700000u, 0x80000100u, 0xbf810000u});
    return code;
}

std::size_t verifyGraph(const Program& program) {
    const auto decoded = RdnaInstructionDecoder{}.Decode(program.code);
    const auto original = GraphBuilder{}.Build(decoded);
    auto graph = original;
    Structurizer{}.Structurize(graph);
    verifyStructured(program, graph);
    verifySameExecutions(program, original, graph);
    return graph.blocks.size() - original.blocks.size();
}

void verifyRequest(const char* path) {
    std::ifstream file(path, std::ios::binary);
    std::ostringstream text;
    text << file.rdbuf();
    const auto request = RequestSerializer{}.Deserialize(text.str());
    const Program program{path, {}, {request.request.shader.code.begin(), request.request.shader.code.end()}, 0};
    const auto added = verifyGraph(program);
    std::printf("%s: structured with %zu added blocks\n", path, added);
}

void verifyProgram(const Program& program) {
    const auto added = verifyGraph(program);
    require(added >= program.minimumAddedBlocks, std::string(program.name) + ": expected at least " + std::to_string(program.minimumAddedBlocks) + " added blocks, got " + std::to_string(added));

    const std::array<std::uint32_t, 4> userData{0x10000000u, 0x00100000u, 0x40u, 0x00027facu};
    const std::array<std::uint32_t, 2> capabilities{1u, 61u};
    const std::array<std::string_view, 1> extensions{"SPV_KHR_storage_buffer_storage_class"};
    RecompileRequest request{};
    request.shader = {ShaderStage::Compute, 0x20000u, program.code, 0, {}};
    request.context.waveSize = 64;
    request.context.userDataBaseRegister = 0;
    request.context.userData = userData;
    request.context.compute = ShaderComputeStageInfo{{64u, 1u, 1u}, 0u, {false, false, false}, false, 1u};
    request.target.vulkanVersion = 0x00403000u;
    request.target.spirvVersion = 0x00010600u;
    request.target.subgroupSize = 64;
    request.target.bdaAbiVersion = 1;
    request.target.supportedCapabilities = capabilities;
    request.target.supportedExtensions = extensions;
    request.target.maxWorkgroupSize = {1024u, 1024u, 64u};
    request.target.maxWorkgroupInvocations = 1024;
    request.target.maxWorkgroupSharedMemoryBytes = 49152;
    request.layout = {0, 0, 0, 128};
    request.useCache = false;
    const auto result = Recompile(request);
    require(!result.spirv.empty(), std::string(program.name) + ": no SPIR-V");
}

}

int main(int argc, char** argv) {
    if (argc > 1) {
        int failures = 0;
        for (int i = 1; i < argc; ++i) {
            try {
                verifyRequest(argv[i]);
            } catch (const std::exception& error) {
                std::fprintf(stderr, "%s: %s\n", argv[i], error.what());
                ++failures;
            }
        }
        return failures == 0 ? 0 : 1;
    }
    const std::vector<Program> programs{
        {"shared tail", R"(
  v_cmp_gt_u32 vcc, 16, v0
  s_cbranch_vccz outer_else
  v_cmp_gt_u32 vcc, 8, v0
  s_cbranch_vccnz tail
  v_add_nc_u32 v1, 1, v0
  s_branch done
outer_else:
  v_add_nc_u32 v1, 2, v0
tail:
  v_add_nc_u32 v1, 3, v1
done:
  buffer_store_dword v1, off, s[0:3], 0
  s_endpgm)",
         Store({0x7d880090u, 0xbf860004u, 0x7d880088u, 0xbf870003u, 0x4a020081u, 0xbf820002u, 0x4a020082u, 0x4a020283u}), 1},
        {"shared tail with a selection", R"(
  v_cmp_gt_u32 vcc, 16, v0
  s_cbranch_vccz outer_else
  v_cmp_gt_u32 vcc, 8, v0
  s_cbranch_vccnz tail
  v_add_nc_u32 v1, 1, v0
  s_branch done
outer_else:
  v_add_nc_u32 v1, 2, v0
tail:
  v_cmp_gt_u32 vcc, 4, v0
  s_cbranch_vccz tail_join
  v_add_nc_u32 v1, 5, v1
tail_join:
  v_add_nc_u32 v1, 3, v1
done:
  buffer_store_dword v1, off, s[0:3], 0
  s_endpgm)",
         Store({0x7d880090u, 0xbf860004u, 0x7d880088u, 0xbf870003u, 0x4a020081u, 0xbf820005u, 0x4a020082u, 0x7d880084u, 0xbf860001u, 0x4a020285u, 0x4a020283u}), 3},
        {"shared tail in a loop", R"(
  s_mov_b32 s8, 0
  v_mov_b32 v1, 0
loop:
  v_cmp_gt_u32 vcc, s8, v0
  s_cbranch_vccz outer_else
  v_cmp_gt_u32 vcc, 8, v0
  s_cbranch_vccnz tail
  v_add_nc_u32 v1, 1, v1
  s_branch latch
outer_else:
  v_add_nc_u32 v1, 2, v1
tail:
  v_add_nc_u32 v1, 3, v1
latch:
  s_add_u32 s8, s8, 16
  s_cmp_lt_u32 s8, 64
  s_cbranch_scc1 loop
  buffer_store_dword v1, off, s[0:3], 0
  s_endpgm)",
         Store({0xbe880380u, 0x7e020280u, 0x7d880008u, 0xbf860004u, 0x7d880088u, 0xbf870003u, 0x4a020281u, 0xbf820002u, 0x4a020282u, 0x4a020283u, 0x80089008u, 0xbf0ac008u, 0xbf85fff5u}), 1},
        {"shared early exit", R"(
  v_mov_b32 v1, 0
  v_cmp_gt_u32 vcc, 16, v0
  s_and_saveexec_b64 s[8:9], vcc
  s_cbranch_execz join
  v_cmp_gt_u32 vcc, 4, v0
  s_cbranch_vccz early_exit
  v_add_nc_u32 v1, 1, v0
join:
  s_mov_b64 exec, s[8:9]
  v_cmp_gt_u32 vcc, 32, v0
  s_cbranch_vccz early_exit
  buffer_store_dword v1, off, s[0:3], 0
  s_endpgm
early_exit:
  s_mov_b64 exec, 0
  s_endpgm)",
         {0x7e020280u, 0x7d880090u, 0xbe88246au, 0xbf880003u, 0x7d880084u, 0xbf860007u, 0x4a020081u, 0xbefe0408u, 0x7d8800a0u, 0xbf860003u, 0xe0700000u, 0x80000100u, 0xbf810000u, 0xbefe0480u, 0xbf810000u}, 1},
        {"short-circuit condition", R"(
  v_cmp_gt_u32 vcc, 16, v0
  s_cbranch_vccz body
  v_cmp_gt_u32 vcc, 8, v0
  s_cbranch_vccnz done
body:
  v_add_nc_u32 v1, 1, v0
  v_cmp_gt_u32 vcc, 4, v0
  s_cbranch_vccz done
  v_add_nc_u32 v1, 2, v1
done:
  buffer_store_dword v1, off, s[0:3], 0
  s_endpgm)",
         Store({0x7d880090u, 0xbf860002u, 0x7d880088u, 0xbf870004u, 0x4a020081u, 0x7d880084u, 0xbf860001u, 0x4a020282u}), 2},
    };
    int failures = 0;
    for (const auto& program : programs) {
        try {
            verifyProgram(program);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "%s\n%.*s\n", error.what(), static_cast<int>(program.source.size()), program.source.data());
            ++failures;
        }
    }
    if (failures != 0) return 1;
    std::puts("Control flow structurization tests passed");
    return 0;
}
