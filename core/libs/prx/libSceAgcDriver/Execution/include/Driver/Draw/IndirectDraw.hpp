#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_INDIRECTDRAW_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_INDIRECTDRAW_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace AgcDriver::DriverDetail {

std::uint32_t DrawUserWord(const DrawProgram& program, std::int32_t sgpr);

std::optional<std::pair<std::size_t, std::size_t>> LocateDrawUserWord(const std::vector<DrawProgram>& programs, const std::vector<ShaderRecompiler::ProgramRole>& roles, std::uint32_t location);

void ResolveIndirectSgprs(const std::vector<DrawProgram>& programs, const std::vector<ShaderRecompiler::ProgramRole>& roles, Pm4::DrawParameters::IndirectDraw& indirect);

Pm4::DrawParameters MeshIndexParameters(const Pm4::DrawParameters& draw);

std::optional<Graphics::IndirectDrawPath> ClassifyIndirectDraw(const ShaderRecompiler::RecompileResult& main, const Graphics::State& graphics, const DrawProgram& front, const VulkanDevice::IndirectDrawSupport& support, Pm4::DrawParameters& drawParameters);

}

#endif
