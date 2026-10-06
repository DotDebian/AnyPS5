#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_CONTEXT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_CONTEXT_HPP

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <unordered_map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace AgcDriver::Graphics {

class TextureDetiler;
class GpuColorTransfer;
class BufferPool;
class TextureCache;
class RenderCache;
class DrawQueue;
class GraphicsPipelineCache;
class Recorder;
class DescriptorCache;
class SamplerCache;
class ShaderResources;

inline void Require(bool condition, const std::string& reason) {
    if (!condition) throw std::runtime_error("AGC graphics: " + reason);
}

class DeviceMemoryExhausted : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

inline void Check(VkResult result, const char* operation) {
    if (result == VK_SUCCESS) return;
    const auto message = std::string("AGC graphics: ") + operation + ": Vulkan result " + std::to_string(result);
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) throw DeviceMemoryExhausted(message);
    throw std::runtime_error(message);
}

inline void Require(bool condition, const char* reason) {
    if (!condition) throw std::runtime_error(std::string("AGC graphics: ") + reason);
}

// Device entry points resolved once per device (VulkanDevice's State fills it after setup): the
// loader's vkGetDeviceProcAddr is a name lookup under its global mutex per call, paid at every
// record site otherwise. Null members (tests, APS5_NO_PROC_TABLE=1) resolve per call (Resolved).
struct DeviceFunctions {
    PFN_vkCmdPipelineBarrier cmdPipelineBarrier = nullptr;
    PFN_vkCmdCopyBuffer cmdCopyBuffer = nullptr;
    PFN_vkCmdUpdateBuffer cmdUpdateBuffer = nullptr;
    PFN_vkCmdFillBuffer cmdFillBuffer = nullptr;
    PFN_vkCmdBindPipeline cmdBindPipeline = nullptr;
    PFN_vkCmdBindDescriptorSets cmdBindDescriptorSets = nullptr;
    PFN_vkCmdPushConstants cmdPushConstants = nullptr;
    PFN_vkCmdDispatch cmdDispatch = nullptr;
    PFN_vkCmdDispatchIndirect cmdDispatchIndirect = nullptr;
    PFN_vkCmdBeginRenderPass cmdBeginRenderPass = nullptr;
    PFN_vkCmdEndRenderPass cmdEndRenderPass = nullptr;
    PFN_vkCmdSetViewport cmdSetViewport = nullptr;
    PFN_vkCmdSetScissor cmdSetScissor = nullptr;
    PFN_vkCmdSetDepthBounds cmdSetDepthBounds = nullptr;
    PFN_vkCmdSetDepthBias cmdSetDepthBias = nullptr;
    PFN_vkCmdBindVertexBuffers cmdBindVertexBuffers = nullptr;
    PFN_vkCmdBindIndexBuffer cmdBindIndexBuffer = nullptr;
    PFN_vkCmdDraw cmdDraw = nullptr;
    PFN_vkCmdDrawIndexed cmdDrawIndexed = nullptr;
    PFN_vkCmdDrawIndirect cmdDrawIndirect = nullptr;
    PFN_vkCmdDrawIndexedIndirect cmdDrawIndexedIndirect = nullptr;
    PFN_vkCmdCopyBufferToImage cmdCopyBufferToImage = nullptr;
    PFN_vkCmdCopyImageToBuffer cmdCopyImageToBuffer = nullptr;
    PFN_vkCmdClearColorImage cmdClearColorImage = nullptr;
    PFN_vkUpdateDescriptorSets updateDescriptorSets = nullptr;
    PFN_vkAllocateDescriptorSets allocateDescriptorSets = nullptr;
    PFN_vkGetFenceStatus getFenceStatus = nullptr;
};

// Per-thread count of vkGetDeviceProcAddr lookups made through Context::Function (the [vk] line).
// APS5_TRACE_VRAM (local, not for upstream): how often the entry points that make or destroy
// Vulkan objects were asked for (vkCreate*, vkDestroy*, vkAllocate*, vkFree*), by name, for the
// [vram] line: nearly every such call of the driver resolves its entry point at the call
// (Context::Function), so the lookups are the calls, and an object kind whose makes run ahead of
// its destroys names itself. The few sites that keep the pointer (the buffer pool's and the query
// pool caches' destroys) count their calls themselves. Nothing is recorded without the switch.
namespace VulkanCalls {

inline bool Traced() {
    static const bool traced = std::getenv("APS5_TRACE_VRAM") != nullptr;
    return traced;
}

struct Table {
    std::mutex mutex;
    std::map<std::string, std::uint64_t> counts;
};

inline Table& Counts() {
    static Table table;
    return table;
}

inline void Count(const char* name) {
    if (!Traced() || name == nullptr) return;
    if (std::strncmp(name, "vkCreate", 8) != 0 && std::strncmp(name, "vkDestroy", 9) != 0 && std::strncmp(name, "vkAllocate", 10) != 0 && std::strncmp(name, "vkFree", 6) != 0) return;
    auto& table = Counts();
    std::lock_guard lock(table.mutex);
    ++table.counts[name];
}

// "<Object> made/destroyed" for every object kind seen, so far.
inline std::string Describe() {
    std::map<std::string, std::pair<std::uint64_t, std::uint64_t>> objects;
    {
        auto& table = Counts();
        std::lock_guard lock(table.mutex);
        for (const auto& [name, count] : table.counts) {
            if (name.rfind("vkCreate", 0) == 0) objects[name.substr(8)].first += count;
            else if (name.rfind("vkAllocate", 0) == 0) objects[name.substr(10)].first += count;
            else if (name.rfind("vkDestroy", 0) == 0) objects[name.substr(9)].second += count;
            else objects[name.substr(6)].second += count;
        }
    }
    std::string text;
    for (const auto& [object, counts] : objects) text += " " + object + " " + std::to_string(counts.first) + "/" + std::to_string(counts.second);
    return text;
}

// APS5_COUNT_VK=1 (local, not for upstream): how many vkCmd* commands and vkQueueSubmit calls the
// driver records, by name, for the [vkcount] line (per 10 s and per present). Every call through
// Context::Function, Context::Resolved and the recorder's cached pointers is counted; a counter
// slot per distinct name, found through a per-thread cache keyed by the literal's address.
inline bool CmdCounted() {
    static const bool counted = std::getenv("APS5_COUNT_VK") != nullptr;
    return counted;
}

struct CmdTable {
    static constexpr std::size_t Slots = 96;
    std::mutex mutex;
    std::vector<std::string> names;
    std::atomic<std::uint64_t> counts[Slots]{};
};

inline CmdTable& CmdCounts() {
    static CmdTable table;
    return table;
}

inline void CountCmd(const char* name, std::uint64_t count = 1) {
    if (!CmdCounted() || name == nullptr) return;
    if (std::strncmp(name, "vkCmd", 5) != 0 && std::strcmp(name, "vkQueueSubmit") != 0) return;
    thread_local std::unordered_map<const char*, std::size_t> slots;
    auto it = slots.find(name);
    if (it == slots.end()) {
        auto& table = CmdCounts();
        std::lock_guard lock(table.mutex);
        std::size_t slot = 0;
        while (slot < table.names.size() && table.names[slot] != name) ++slot;
        if (slot == table.names.size()) {
            if (slot >= CmdTable::Slots) return;
            table.names.emplace_back(name);
        }
        it = slots.emplace(name, slot).first;
    }
    CmdCounts().counts[it->second].fetch_add(count, std::memory_order_relaxed);
}

// " name count (per present x)" for every counted name, highest first; clears the counts.
inline std::string TakeCmdCounts(std::uint64_t presents) {
    auto& table = CmdCounts();
    std::vector<std::pair<std::uint64_t, std::string>> rows;
    {
        std::lock_guard lock(table.mutex);
        for (std::size_t i = 0; i < table.names.size(); ++i) rows.emplace_back(table.counts[i].exchange(0, std::memory_order_relaxed), table.names[i]);
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::string text;
    for (const auto& [count, name] : rows) {
        if (count == 0) continue;
        char row[128];
        std::snprintf(row, sizeof(row), " %s %llu (%.1f)", name.c_str() + 2, static_cast<unsigned long long>(count), presents != 0 ? static_cast<double>(count) / static_cast<double>(presents) : 0.0);
        text += row;
    }
    return text;
}

}

inline std::uint64_t& DeviceProcLookups() {
    thread_local std::uint64_t count = 0;
    return count;
}

inline constexpr std::size_t EmptyBufferBytes = 16;

struct Context {
    VkDevice device;
    VkPhysicalDevice physical;
    VkQueue queue;
    VkCommandPool pool;
    PFN_vkGetDeviceProcAddr deviceProc;
    PFN_vkGetPhysicalDeviceFormatProperties formatProperties;
    PFN_vkGetPhysicalDeviceImageFormatProperties imageFormatProperties;
    VkPhysicalDeviceMemoryProperties memory;
    VkPhysicalDeviceLimits limits;
    bool tessellationShader = false;
    bool meshShader = false;
    VkPhysicalDeviceMeshShaderPropertiesEXT meshLimits{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
    bool depthClipControl = false;
    bool depthRangeUnrestricted = false;
    bool bufferDeviceAddress = false;
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    bool fragmentShaderBarycentric = false;
    bool samplerAnisotropy = false;
    bool textureCompressionBC = false;
    TextureDetiler* detiler = nullptr;
    GpuColorTransfer* colorTransfer = nullptr;
    mutable std::shared_ptr<BufferPool> bufferPool;
    TextureCache* textureCache = nullptr;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    bool depthClamp = false;
    // Nonzero when VK_EXT_external_memory_host is enabled: the required host pointer alignment.
    VkDeviceSize hostImportAlignment = 0;
    RenderCache* renderCache = nullptr;
    DrawQueue* drawQueue = nullptr;
    GraphicsPipelineCache* graphicsPipelines = nullptr;
    // Batches GPU work across guest commands (see Recorder); null before the device finished setup.
    Recorder* recorder = nullptr;
    // Per-device caches of the immutable descriptor objects a ShaderResources build needs (set
    // layouts, descriptor pools, samplers); null (tests) means every build makes and destroys its own.
    DescriptorCache* descriptorCache = nullptr;
    SamplerCache* samplerCache = nullptr;
    // Indirect draw features of the device: records with a non-zero first instance, several records
    // per call, and a GPU-side draw count (VK_KHR_draw_indirect_count).
    bool drawIndirectFirstInstance = false;
    bool multiDrawIndirect = false;
    bool drawIndirectCount = false;
    bool occlusionQueryPrecise = false;
    VkBuffer emptyBuffer = VK_NULL_HANDLE;
    // The device's list of recorded dispatches whose copied written buffers await a CPU write-back
    // (VulkanDevice's State::copiedWriters; the draw counterpart is DrawCopiedWriters): an indirect
    // draw whose records one of them produces reads them on the CPU. Null in tests.
    const std::vector<std::shared_ptr<ShaderResources>>* copiedWriters = nullptr;
    // The device's resolved entry points (see DeviceFunctions); null until the device set up.
    const DeviceFunctions* functions = nullptr;
    // VK_EXT_descriptor_indexing with non-uniform sampled/storage image array indexing enabled
    // (bindless image tables in graphics stages).
    bool descriptorIndexing = false;
    bool hostQueryReset = false;
    // VK_EXT_primitive_topology_list_restart: primitive restart on list topologies.
    bool primitiveListRestart = false;
    // The depthBiasClamp feature (PA_SU_POLY_OFFSET_CLAMP).
    bool depthBiasClamp = false;
    bool depthBounds = false;
    // VK_EXT_image_view_min_lod with minLod enabled: sampled views clamp their level of detail as a
    // texture descriptor's MIN_LOD does.
    bool imageViewMinLod = false;
    // The device's GDS buffer (Pm4::GdsBytes, the CP's GDS backing, see Pm4::InstallGdsBacking), which
    // shaders' GDS bindings name; null when the device has none (tests).
    VkBuffer gdsBuffer = VK_NULL_HANDLE;
    PFN_vkGetPhysicalDeviceMemoryProperties2 memoryProperties2 = nullptr;
    bool memoryBudget = false;
    bool pipelineExecutableInfo = false;
    // VK_KHR_maintenance8 enabled: vkCmdCopyImage copies between depth/stencil and color formats.
    bool maintenance8 = false;
    // Storage images are read and written without a format: storage write-backs retile straight
    // from the image and uploads detile straight into it (TextureDetiler::DispatchImage), with no
    // linear buffer and copy between. APS5_NO_SINGLE_PASS_STORAGE=1 keeps the copies.
    bool singlePassStorage = false;

    template<typename TFunction>
    TFunction Function(const char* name) const {
        Require(deviceProc != nullptr, "missing Vulkan device function resolver");
        ++DeviceProcLookups();
        VulkanCalls::Count(name);
        VulkanCalls::CountCmd(name);
        const auto function = reinterpret_cast<TFunction>(deviceProc(device, name));
        if (function == nullptr) throw std::runtime_error(std::string("AGC graphics: missing Vulkan function: ") + name);
        return function;
    }

    // The table's entry point, or a per-call lookup while the table is absent or lacks it.
    template<typename TFunction>
    TFunction Resolved(TFunction DeviceFunctions::*member, const char* name) const {
        if (functions != nullptr && functions->*member != nullptr) {
            VulkanCalls::CountCmd(name);
            return functions->*member;
        }
        return Function<TFunction>(name);
    }

    std::uint32_t MemoryType(std::uint32_t mask, VkMemoryPropertyFlags flags) const {
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if ((mask & (1u << i)) != 0 && (memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
        }
        throw std::runtime_error("AGC graphics: required Vulkan memory type is unavailable");
    }
};

}

#endif
