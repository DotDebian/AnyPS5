#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_PREPAREDDRAW_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_PREPAREDDRAW_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Execution/include/ShaderMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderInputState.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

namespace AgcDriver::DriverDetail {

struct PreparedDraw {
    static constexpr std::size_t NoResult = std::numeric_limits<std::size_t>::max();
    std::uint64_t deviceSerial = 0;
    std::uint64_t forgetSerial = 0;
    std::uint64_t drawKey = 0;
    // Driver::drawRegisterKey's state key (0 without APS5_DRAW_PLANS).
    std::uint64_t stateKey = 0;
    bool keyKnown = false;
    std::shared_ptr<const DrawDecode> decode;
    std::vector<DrawProgram> programs;
    Pm4::DrawParameters drawParameters{};
    std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>> vertexInfos;
    std::vector<std::vector<Graphics::DecodeRead>> decodeReads;
    std::unique_ptr<ShaderMemory> shaderMemory;
    std::vector<ShaderRecompiler::MemoryRegion> memory;
    std::vector<ShaderRecompiler::RecompileResult> results;
    std::vector<std::size_t> resultIndex;
    std::vector<StageCapture> stageCaptures;
    std::uint64_t captures = 0;
};

}

#endif
