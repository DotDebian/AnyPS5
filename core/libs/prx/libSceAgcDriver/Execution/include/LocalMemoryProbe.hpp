#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_LOCALMEMORYPROBE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_LOCALMEMORYPROBE_HPP

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "Recompiler.hpp"
#include <cstdint>
#include <optional>
#include <span>

namespace AgcDriver {

struct LocalMemoryProbeContext {
    VkDevice device;
    PFN_vkGetDeviceProcAddr deviceProc;
    VkPipelineCache cache;
};

[[nodiscard]] std::optional<std::uint32_t> MeasureComputeLocalMemory(const LocalMemoryProbeContext& device, std::span<const std::uint32_t> spirv, std::span<const ShaderRecompiler::DescriptorBinding> bindings, std::uint64_t codeAddress);

}

#endif
