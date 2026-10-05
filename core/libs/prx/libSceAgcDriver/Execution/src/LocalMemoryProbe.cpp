#include "prx/libSceAgcDriver/Execution/include/LocalMemoryProbe.hpp"
#include "prx/libSceAgcDriver/Execution/include/Report.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace AgcDriver {
namespace {

void check(VkResult result, const char* operation) {
    if (result != VK_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": Vulkan result " + std::to_string(result));
    }
}

template<typename TFunction>
TFunction deviceFunction(const LocalMemoryProbeContext& device, const char* name) {
    ++Graphics::DeviceProcLookups();
    auto function = reinterpret_cast<TFunction>(device.deviceProc(device.device, name));
    if (function == nullptr) {
        throw std::runtime_error(std::string("Vulkan device function missing: ") + name);
    }
    return function;
}

VkDescriptorType ProbeDescriptorType(ShaderRecompiler::DescriptorKind kind) {
    switch (kind) {
    case ShaderRecompiler::DescriptorKind::UniformBuffer: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case ShaderRecompiler::DescriptorKind::StorageBuffer: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case ShaderRecompiler::DescriptorKind::UniformTexelBuffer: return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    case ShaderRecompiler::DescriptorKind::StorageTexelBuffer: return VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    case ShaderRecompiler::DescriptorKind::SampledImage: return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    case ShaderRecompiler::DescriptorKind::StorageImage: return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case ShaderRecompiler::DescriptorKind::Sampler: return VK_DESCRIPTOR_TYPE_SAMPLER;
    }
    return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
}

}

// The recompiler's local memory probe (SpirvTarget::localMemoryProbe): compiles the compute module
// with a layout made from its bindings, reads the per-invocation local memory from the pipeline's
// statistics (NVIDIA's "Local Memory Size", whose low 32 bits are the bytes; other drivers' spill
// or scratch sizes) and throws the pipeline away. The real pipeline later compiles the module
// again, which the driver's own cache serves. Debug aid: APS5_TRACE_LOCAL_MEMORY_PROBE=1 prints
// every probe's statistics.
std::optional<std::uint32_t> MeasureComputeLocalMemory(const LocalMemoryProbeContext& device, std::span<const std::uint32_t> spirv, std::span<const ShaderRecompiler::DescriptorBinding> bindings, std::uint64_t codeAddress) {
    static const bool trace = std::getenv("APS5_TRACE_LOCAL_MEMORY_PROBE") != nullptr;
    VkShaderModule module = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayout> setLayouts;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const auto release = [&] {
        if (pipeline != VK_NULL_HANDLE) deviceFunction<PFN_vkDestroyPipeline>(device, "vkDestroyPipeline")(device.device, pipeline, nullptr);
        if (layout != VK_NULL_HANDLE) deviceFunction<PFN_vkDestroyPipelineLayout>(device, "vkDestroyPipelineLayout")(device.device, layout, nullptr);
        for (const auto set : setLayouts) deviceFunction<PFN_vkDestroyDescriptorSetLayout>(device, "vkDestroyDescriptorSetLayout")(device.device, set, nullptr);
        if (module != VK_NULL_HANDLE) deviceFunction<PFN_vkDestroyShaderModule>(device, "vkDestroyShaderModule")(device.device, module, nullptr);
    };
    std::optional<std::uint32_t> bytes;
    std::string evidence;
    try {
        VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        moduleInfo.codeSize = spirv.size() * sizeof(std::uint32_t);
        moduleInfo.pCode = spirv.data();
        check(deviceFunction<PFN_vkCreateShaderModule>(device, "vkCreateShaderModule")(device.device, &moduleInfo, nullptr, &module), "vkCreateShaderModule (local memory probe)");
        std::uint32_t sets = 0;
        for (const auto& binding : bindings) sets = std::max(sets, binding.descriptorSet + 1u);
        for (std::uint32_t set = 0; set < sets; ++set) {
            std::vector<VkDescriptorSetLayoutBinding> entries;
            for (const auto& binding : bindings) {
                if (binding.descriptorSet == set) entries.push_back({binding.binding, ProbeDescriptorType(binding.kind), std::max(binding.count, 1u), VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
            }
            VkDescriptorSetLayoutCreateInfo setInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            setInfo.bindingCount = static_cast<std::uint32_t>(entries.size());
            setInfo.pBindings = entries.data();
            setLayouts.push_back(VK_NULL_HANDLE);
            check(deviceFunction<PFN_vkCreateDescriptorSetLayout>(device, "vkCreateDescriptorSetLayout")(device.device, &setInfo, nullptr, &setLayouts.back()), "vkCreateDescriptorSetLayout (local memory probe)");
        }
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, Graphics::PipelinePushConstantBytes};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = static_cast<std::uint32_t>(setLayouts.size());
        layoutInfo.pSetLayouts = setLayouts.data();
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &push;
        check(deviceFunction<PFN_vkCreatePipelineLayout>(device, "vkCreatePipelineLayout")(device.device, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout (local memory probe)");
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.flags = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = module;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = layout;
        check(deviceFunction<PFN_vkCreateComputePipelines>(device, "vkCreateComputePipelines")(device.device, device.cache, 1, &pipelineInfo, nullptr, &pipeline), "vkCreateComputePipelines (local memory probe)");
        const auto executables = deviceFunction<PFN_vkGetPipelineExecutablePropertiesKHR>(device, "vkGetPipelineExecutablePropertiesKHR");
        const auto statistics = deviceFunction<PFN_vkGetPipelineExecutableStatisticsKHR>(device, "vkGetPipelineExecutableStatisticsKHR");
        const VkPipelineInfoKHR info{VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, nullptr, pipeline};
        std::uint32_t count = 0;
        check(executables(device.device, &info, &count, nullptr), "vkGetPipelineExecutablePropertiesKHR");
        for (std::uint32_t index = 0; index < count; ++index) {
            const VkPipelineExecutableInfoKHR executable{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, nullptr, pipeline, index};
            std::uint32_t statisticCount = 0;
            check(statistics(device.device, &executable, &statisticCount, nullptr), "vkGetPipelineExecutableStatisticsKHR");
            std::vector<VkPipelineExecutableStatisticKHR> values(statisticCount, {VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR});
            check(statistics(device.device, &executable, &statisticCount, values.data()), "vkGetPipelineExecutableStatisticsKHR");
            for (const auto& value : values) {
                std::uint64_t number = 0;
                if (value.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR) number = value.value.u64;
                else if (value.format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR) number = value.value.i64 > 0 ? static_cast<std::uint64_t>(value.value.i64) : 0u;
                else continue;
                const std::string_view name(value.name);
                bool local = false;
                if (name == "Local Memory Size") local = true, number &= 0xffffffffull;
                else if (name.find("Spill") != std::string_view::npos || name.find("spill") != std::string_view::npos || name.find("Scratch") != std::string_view::npos) local = true;
                if (trace) evidence += std::string(evidence.empty() ? "" : ", ") + value.name + " " + std::to_string(number);
                if (local) bytes = static_cast<std::uint32_t>(std::min<std::uint64_t>(number + bytes.value_or(0u), 0xffffffffull));
            }
        }
    } catch (const std::exception& error) {
        AgcDriver::ReportLine("[wave64] program 0x%llx: the local memory probe failed (%s)\n", static_cast<unsigned long long>(codeAddress), error.what());
        bytes.reset();
    }
    release();
    if (trace) AgcDriver::ReportLine("[wave64] program 0x%llx: %zu SPIR-V words: %s\n", static_cast<unsigned long long>(codeAddress), spirv.size(), evidence.c_str());
    return bytes;
}

}
