#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/IndirectDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Report.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include <algorithm>
#include <cstdlib>
#include <limits>

namespace AgcDriver::DriverDetail {

std::uint32_t DrawUserWord(const DrawProgram& program, std::int32_t sgpr) {
    require(sgpr >= 0 && static_cast<std::uint32_t>(sgpr) >= program.firstUserSgpr, "invalid draw offset SGPR");
    const auto index = static_cast<std::uint32_t>(sgpr) - program.firstUserSgpr;
    require(index < program.userData.size(), "draw offset SGPR exceeds user data");
    return program.userData[index];
}

std::optional<std::pair<std::size_t, std::size_t>> LocateDrawUserWord(const std::vector<DrawProgram>& programs, const std::vector<ShaderRecompiler::ProgramRole>& roles, std::uint32_t location) {
    using Role = ShaderRecompiler::ProgramRole;
    if (location == 0x280u) return std::nullopt;
    for (std::size_t i = 0; i < programs.size(); ++i) {
        if (roles[i] == Role::Fragment || roles[i] == Role::GeometryBack || location < programs[i].userDataBase) continue;
        const auto word = location - programs[i].userDataBase + (8u - programs[i].firstUserSgpr);
        if (word < programs[i].userData.size()) return std::make_pair(i, static_cast<std::size_t>(word));
    }
    return std::nullopt;
}

void ResolveIndirectSgprs(const std::vector<DrawProgram>& programs, const std::vector<ShaderRecompiler::ProgramRole>& roles, Pm4::DrawParameters::IndirectDraw& indirect) {
    const auto sgprOf = [&](std::uint32_t location) -> std::int32_t {
        const auto word = LocateDrawUserWord(programs, roles, location);
        if (!word || word->first != 0) return -1;
        return static_cast<std::int32_t>(programs.front().firstUserSgpr + word->second);
    };
    indirect.baseVertexSgpr = sgprOf(indirect.baseVertexLocation);
    indirect.startInstanceSgpr = sgprOf(indirect.startInstanceLocation);
    indirect.drawIndexSgpr = sgprOf(indirect.drawIndexLocation);
}

Pm4::DrawParameters MeshIndexParameters(const Pm4::DrawParameters& draw) {
    if (!draw.indirect) return draw;
    return Pm4::DrawParameters{draw.indexAddress, std::max(draw.indexCount, 1u), draw.indexSize, 1, 0, draw.indexed};
}

std::optional<Graphics::IndirectDrawPath> ClassifyIndirectDraw(const ShaderRecompiler::RecompileResult& result, const Graphics::State& graphics, const DrawProgram& frontProgram, const VulkanDevice::IndirectDrawSupport& support, Pm4::DrawParameters& drawParameters) {
    std::optional<Graphics::IndirectDrawPath> indirectCpu;
    auto& indirect = *drawParameters.indirect;
    using Rule = Pm4::DrawParameters::IndirectDraw::Rule;
    using Path = Graphics::IndirectDrawPath;
    const auto fail = [&](Path reason) { if (!indirectCpu) indirectCpu = reason; };
    static const bool gpuMeshIndirect = std::getenv("APS5_NO_GPU_MESH_INDIRECT") == nullptr;
    constexpr std::uint32_t NoLocation = 0x280u;
    const bool meshPath = graphics.stages.mesh.has_value() && !graphics.stages.tessellation && !graphics.rectList;
    const bool meshGpuSide = gpuMeshIndirect && meshPath && drawParameters.indexed && indirect.count == 1 && !indirect.countIndirect && indirect.baseVertexLocation == NoLocation && indirect.startInstanceLocation == NoLocation && (!indirect.drawIndexEnabled || indirect.drawIndexLocation == NoLocation);
    if ((graphics.stages.path != Graphics::ShaderPath::Vertex || graphics.rectList) && !meshGpuSide) fail(Path::NonVertexPath);
    if (indirect.drawIndexEnabled && indirect.drawIndexSgpr >= 0) fail(Path::DrawIndex);
    if (indirect.countIndirect && !support.count) fail(Path::FeatureGap);
    if (indirect.countIndirect && indirect.count > 1 && !support.multi) fail(Path::FeatureGap);
    const auto dimension = [&](std::int32_t k, std::int32_t p, bool shared, bool conflict, std::uint32_t x, Rule& rule, std::uint32_t& constant) {
        if (p < 0) {
            rule = Rule::Constant;
            const auto offset = k >= 0 ? DrawUserWord(frontProgram, k) : 0u;
            require(offset <= std::numeric_limits<std::uint32_t>::max() - x, "draw vertex offset overflow");
            constant = x + offset;
        } else if (conflict) fail(Path::FetchUnknown);
        else if (k != p || shared) fail(Path::NotFolded);
        else if (x != 0) fail(Path::IndxOffset);
        else rule = Rule::InPlace;
    };
    dimension(result.vertexOffsetSgpr, indirect.baseVertexSgpr, result.vertexOffsetShared, result.vertexOffsetConflict, drawParameters.indexed ? 0u : indirect.indxOffset, indirect.vertexRule, indirect.vertexConstant);
    dimension(result.instanceOffsetSgpr, indirect.startInstanceSgpr, result.instanceOffsetShared, result.instanceOffsetConflict, 0u, indirect.instanceRule, indirect.instanceConstant);
    if (!support.firstInstance && (indirect.instanceRule == Rule::InPlace || indirect.instanceConstant != 0)) fail(Path::FeatureGap);
    if ((indirect.vertexRule == Rule::Constant || indirect.instanceRule == Rule::Constant) && indirect.count > 256) fail(Path::FeatureGap);

    static const std::uint64_t vertexCap = [] { const char* text = std::getenv("APS5_INDIRECT_VERTEX_MIB"); return (text ? std::strtoull(text, nullptr, 10) : 64ull) << 20u; }();
    for (const auto& attribute : result.vertexAttributes) {
        const auto stride = (attribute.resource.fields[1] >> 16u) & 0x3fffu;
        const auto extent = stride == 0 ? static_cast<std::uint64_t>(attribute.resource.fields[2]) : static_cast<std::uint64_t>(attribute.resource.fields[2]) * stride;
        if (extent > vertexCap) fail(Path::VertexRange);
    }
    return indirectCpu;
}

IndirectAhead IndirectAheadRule(const std::optional<Graphics::IndirectDrawPath>& path, bool enabled) {
    if (!enabled) return IndirectAhead::Disabled;
    return path ? IndirectAhead::CpuRecords : IndirectAhead::Prepare;
}

bool AdoptableIndirect(const PreparedDraw& prepared, const VulkanDevice::IndirectDrawSupport& support, bool enabled) {
    if (!prepared.drawParameters.indirect || prepared.decode == nullptr || prepared.programs.empty() || prepared.resultIndex.empty() || prepared.resultIndex.front() >= prepared.results.size()) return false;
    auto parameters = prepared.drawParameters;
    const auto path = ClassifyIndirectDraw(prepared.results[prepared.resultIndex.front()], prepared.decode->state, prepared.programs.front(), support, parameters);
    return IndirectAheadRule(path, enabled) == IndirectAhead::Prepare;
}

bool IndirectDrawAheadEnabled() {
    static const bool enabled = std::getenv("APS5_NO_INDIRECT_DRAW_AHEAD") == nullptr;
    return enabled;
}

std::optional<Graphics::IndirectDrawPath> Driver::classifyIndirectDraw(const ShaderRecompiler::RecompileResult& result, const Graphics::State& graphics, const DrawProgram& frontProgram, const std::shared_ptr<VulkanDevice>& localDevice, Pm4::DrawParameters& drawParameters, bool traceIndirect) {
    const auto indirectCpu = ClassifyIndirectDraw(result, graphics, frontProgram, localDevice->DrawIndirectSupport(), drawParameters);
    if (traceIndirect) {
        using Rule = Pm4::DrawParameters::IndirectDraw::Rule;
        const auto& indirect = *drawParameters.indirect;
        const auto location = [](std::uint32_t value, std::int32_t sgpr) { char text[24]; if (value == 0x280u) std::snprintf(text, sizeof(text), "none"); else std::snprintf(text, sizeof(text), "0x%x(s%d)", value, sgpr); return std::string(text); };
        const auto rule = [](Rule value, std::uint32_t constant) { char text[24]; if (value == Rule::InPlace) std::snprintf(text, sizeof(text), "in-place"); else std::snprintf(text, sizeof(text), "const %u", constant); return std::string(text); };
        std::string decision = indirectCpu ? "cpu (" + std::string(Graphics::IndirectDrawPathName(*indirectCpu)) + ")" : "gpu vertex=" + rule(indirect.vertexRule, indirect.vertexConstant) + " instance=" + rule(indirect.instanceRule, indirect.instanceConstant);
        AgcDriver::ReportLine("[draw] indirect 0x%x args 0x%llx stride %u count %u%s locs base=%s inst=%s idx=%s%s analyzer v=%d i=%d shared=%d/%d conflict=%d/%d indx=%u -> %s\n", indirect.opcode, static_cast<unsigned long long>(indirect.arguments), indirect.stride, indirect.count, indirect.countIndirect ? " (indirect)" : "", location(indirect.baseVertexLocation, indirect.baseVertexSgpr).c_str(), location(indirect.startInstanceLocation, indirect.startInstanceSgpr).c_str(), location(indirect.drawIndexLocation, indirect.drawIndexSgpr).c_str(), indirect.drawIndexEnabled ? " (enabled)" : "", result.vertexOffsetSgpr, result.instanceOffsetSgpr, result.vertexOffsetShared ? 1 : 0, result.instanceOffsetShared ? 1 : 0, result.vertexOffsetConflict ? 1 : 0, result.instanceOffsetConflict ? 1 : 0, indirect.indxOffset, decision.c_str());
    }
    return indirectCpu;
}

}
