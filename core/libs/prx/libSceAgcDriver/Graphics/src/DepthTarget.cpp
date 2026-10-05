#include "prx/libSceAgcDriver/Graphics/include/DepthTarget.hpp"
#include "prx/libSceAgcDriver/Execution/include/Report.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace AgcDriver::Graphics {

namespace {

bool attachable(const Context& context, VkFormat format) {
    VkFormatProperties properties{};
    context.formatProperties(context.physical, format, &properties);
    constexpr VkFormatFeatureFlags required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    return (properties.optimalTilingFeatures & required) == required;
}

struct DepthCounts {
    std::mutex mutex;
    std::uint64_t draws = 0;
    std::uint64_t clearDraws = 0;
    std::uint64_t stencilDraws = 0;
    std::uint64_t imageClears = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

DepthCounts& Counts() {
    static DepthCounts counts;
    return counts;
}

bool residentDepthSampling() {
    static const bool disabled = std::getenv("APS5_NO_RESIDENT_DEPTH_SAMPLING") != nullptr;
    return !disabled;
}

std::atomic<std::uint64_t>& HoldingGeneration() {
    static std::atomic<std::uint64_t> generation{1};
    return generation;
}

}

VkFormat DepthAttachmentFormat(const Context& context, const DepthTarget& target) {
    // The answer only depends on the device's format support, so it is remembered per device.
    static std::mutex mutex;
    static std::map<std::tuple<VkPhysicalDevice, std::uint32_t, bool>, VkFormat> known;
    const auto key = std::make_tuple(context.physical, target.zFormat, target.stencil);
    std::lock_guard lock(mutex);
    if (const auto found = known.find(key); found != known.end()) return found->second;
    std::vector<VkFormat> candidates;
    if (!target.stencil) {
        if (target.zFormat == 1) candidates = {VK_FORMAT_D16_UNORM};
        else candidates = {VK_FORMAT_D32_SFLOAT};
    } else if (target.zFormat == 1) {
        candidates = {VK_FORMAT_D16_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
    } else if (target.zFormat == 3) {
        candidates = {VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    } else {
        candidates = {VK_FORMAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT};
    }
    for (const auto format : candidates) {
        if (!attachable(context, format)) continue;
        if (format != candidates.front()) AgcDriver::ReportLine("[gpu] depth format Z%u%s renders as VkFormat %d on this device\n", target.zFormat, target.stencil ? "+S8" : "", static_cast<int>(format));
        known.emplace(key, format);
        return format;
    }
    throw std::runtime_error("AGC graphics: no attachable Vulkan format for depth format " + std::to_string(target.zFormat) + (target.stencil ? " with stencil" : ""));
}

VkImageAspectFlags DepthAspects(VkFormat format) {
    switch (format) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_D32_SFLOAT:
        case VK_FORMAT_X8_D24_UNORM_PACK32:
            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_S8_UINT:
            return VK_IMAGE_ASPECT_STENCIL_BIT;
        default:
            return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    }
}

DepthImage::DepthImage(const Context& context, const DepthTarget& target, VkFormat format) : context(context), target(target), format(format), aspects(DepthAspects(format)) {
    // Cached images outlive their device's teardown; they must not keep its buffer pool alive past it.
    this->context.bufferPool.reset();
    constexpr auto usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImageFormatProperties supported{};
    Check(context.imageFormatProperties(context.physical, format, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL, usage, 0, &supported), "vkGetPhysicalDeviceImageFormatProperties depth");
    Require(target.extent.width <= supported.maxExtent.width && target.extent.height <= supported.maxExtent.height, "depth target exceeds device image limits");
    Require(target.extent.width <= context.limits.maxFramebufferWidth && target.extent.height <= context.limits.maxFramebufferHeight, "depth target exceeds framebuffer limits");
    try {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {target.extent.width, target.extent.height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        Check(context.Function<PFN_vkCreateImage>("vkCreateImage")(context.device, &info, nullptr, &image), "vkCreateImage depth");
        VkMemoryRequirements requirements{};
        context.Function<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(context.device, image, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = context.MemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        Check(context.Function<PFN_vkAllocateMemory>("vkAllocateMemory")(context.device, &allocation, nullptr, &memory), "vkAllocateMemory depth target");
        Check(context.Function<PFN_vkBindImageMemory>("vkBindImageMemory")(context.device, image, memory, 0), "vkBindImageMemory depth");
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = {aspects, 0, 1, 0, 1};
        Check(context.Function<PFN_vkCreateImageView>("vkCreateImageView")(context.device, &viewInfo, nullptr, &view), "vkCreateImageView depth");
    } catch (...) {
        release();
        throw;
    }
}

DepthImage::~DepthImage() {
    release();
}

void DepthImage::release() noexcept {
    if (view) context.Function<PFN_vkDestroyImageView>("vkDestroyImageView")(context.device, view, nullptr);
    if (image) context.Function<PFN_vkDestroyImage>("vkDestroyImage")(context.device, image, nullptr);
    if (memory) context.Function<PFN_vkFreeMemory>("vkFreeMemory")(context.device, memory, nullptr);
    view = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
}

void DepthImage::Abandon() noexcept {
    view = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
}

std::uint64_t DepthHoldingGeneration() {
    return HoldingGeneration().load(std::memory_order_acquire);
}

void DepthImage::noteHolding() {
    const std::uint32_t mask = (Holds(VK_IMAGE_ASPECT_DEPTH_BIT) ? 1u : 0u) | (Holds(VK_IMAGE_ASPECT_STENCIL_BIT) ? 2u : 0u);
    if (holding.exchange(mask, std::memory_order_acq_rel) != mask) HoldingGeneration().fetch_add(1, std::memory_order_acq_rel);
}

std::shared_ptr<StorageTexture> DepthImage::Writer() const {
    std::lock_guard lock(writerMutex);
    return writer.lock();
}

void DepthImage::NoteStored(const std::shared_ptr<StorageTexture>& storage, VkImageAspectFlags aspect) {
    {
        std::lock_guard lock(writerMutex);
        writer = storage;
    }
    writerAspect.store(aspect, std::memory_order_release);
    holder.store(Holder::Memory, std::memory_order_release);
    noteHolding();
}

void DepthImage::NoteWritten() {
    holder.store(Holder::Image, std::memory_order_release);
    noteHolding();
    version.fetch_add(1, std::memory_order_acq_rel);
}

void DepthImage::RecordCopyToBuffer(VkCommandBuffer commands, VkBuffer buffer, VkImageAspectFlags aspect) const {
    Require(initialized, "a depth image is copied before anything was recorded into it");
    const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
    VkImageMemoryBarrier before{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    before.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    before.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.image = image;
    before.subresourceRange = {aspects, 0, 1, 0, 1};
    barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &before);
    VkBufferImageCopy region{};
    region.imageSubresource = {aspect, 0, 0, 1};
    region.imageExtent = {target.extent.width, target.extent.height, 1};
    context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, image, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &region);
    VkImageMemoryBarrier after = before;
    after.srcAccessMask = 0;
    after.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &after);
}

void DepthImage::RecordCopyFromBuffer(VkCommandBuffer commands, VkBuffer buffer, VkImageAspectFlags aspect) {
    const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
    VkBufferMemoryBarrier staged{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    staged.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    staged.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    staged.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    staged.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    staged.buffer = buffer;
    staged.size = VK_WHOLE_SIZE;
    VkImageMemoryBarrier before{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    before.srcAccessMask = initialized ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT : 0u;
    before.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    before.oldLayout = initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
    before.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.image = image;
    before.subresourceRange = {aspects, 0, 1, 0, 1};
    barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &staged, 1, &before);
    initialized = true;
    VkBufferImageCopy region{};
    region.imageSubresource = {aspect, 0, 0, 1};
    region.imageExtent = {target.extent.width, target.extent.height, 1};
    context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, buffer, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    VkImageMemoryBarrier after = before;
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    after.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &after);
    clearPending.store(false, std::memory_order_release);
    fillClear.store(false, std::memory_order_release);
    holder.store(Holder::Both, std::memory_order_release);
    noteHolding();
    version.fetch_add(1, std::memory_order_acq_rel);
}

bool DepthImage::RecordPendingClear(VkCommandBuffer commands, const DepthState& state) {
    if (!clearPending.exchange(false, std::memory_order_acq_rel)) return false;
    if (fillClear.exchange(false, std::memory_order_acq_rel)) {
        holder.store(Holder::Image, std::memory_order_release);
        noteHolding();
        version.fetch_add(1, std::memory_order_acq_rel);
    }
    {
        auto& counts = Counts();
        std::lock_guard lock(counts.mutex);
        ++counts.imageClears;
    }
    const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
    VkImageMemoryBarrier before{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    before.srcAccessMask = initialized ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT : 0u;
    before.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    before.oldLayout = initialized ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED;
    before.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before.image = image;
    before.subresourceRange = {aspects, 0, 1, 0, 1};
    barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &before);
    initialized = true;
    // Depth clear values outside [0, 1] are clamped by the clear (a float format would keep them,
    // but vkCmdClearDepthStencilImage requires the range without VK_EXT_depth_range_unrestricted).
    const VkClearDepthStencilValue value{std::clamp(state.depthClearValue, 0.0f, 1.0f), state.stencilClearValue};
    const VkImageSubresourceRange range{aspects, 0, 1, 0, 1};
    context.Function<PFN_vkCmdClearDepthStencilImage>("vkCmdClearDepthStencilImage")(commands, image, VK_IMAGE_LAYOUT_GENERAL, &value, 1, &range);
    VkImageMemoryBarrier after = before;
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    after.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr, 1, &after);
    return true;
}

namespace {

struct DepthStore {
    struct Entry {
        VkDevice device;
        // The device's buffer pool at insertion (see PipelineStore in Pipeline.cpp): tells a device
        // instance apart from a later one the loader gave the same handle.
        std::weak_ptr<BufferPool> pool;
        std::uint64_t address;
        std::uint64_t stencilAddress;
        std::uint32_t slice;
        std::uint32_t width;
        std::uint32_t height;
        VkFormat format;
        std::shared_ptr<DepthImage> image;
    };
    struct Sampled {
        struct Layer {
            std::weak_ptr<DepthImage> image;
            std::uint64_t imageVersion = ~0ull;
            std::weak_ptr<StorageTexture> storage;
            std::uint64_t storageVersion = ~0ull;
        };
        VkDevice device;
        std::weak_ptr<Texture> texture;
        std::uint64_t address;
        VkImageAspectFlags aspect;
        std::uint32_t width;
        std::uint32_t height;
        std::vector<Layer> layers;
        std::uint64_t generation = 0;
        std::shared_ptr<DeviceBuffer> scratch;
    };
    std::mutex mutex;
    // Least recently used first.
    std::list<Entry> entries;
    // HTILE buffers already reported.
    std::set<std::uint64_t> reportedFills;
    std::atomic<std::size_t> count{0};
    std::uint64_t generation = 1;
    std::vector<Sampled> sampled;
    std::atomic<std::size_t> sampledCount{0};
};

// Never destroyed: the images belong to a device that may already be gone when statics die.
DepthStore& Store() {
    static auto* store = new DepthStore();
    return *store;
}

bool alive(const DepthStore::Entry& entry, const Context& context) {
    if (entry.device != context.device) return false;
    return context.bufferPool == nullptr || entry.pool.lock() == context.bufferPool;
}

VkImageAspectFlags surfaceAspect(const DepthStore::Entry& entry, std::uint64_t address) {
    const auto& target = entry.image->Target();
    if (target.zFormat != 0 && entry.address == address) return VK_IMAGE_ASPECT_DEPTH_BIT;
    if (target.stencil && (entry.stencilAddress == address || (target.zFormat == 0 && entry.address == address))) return VK_IMAGE_ASPECT_STENCIL_BIT;
    return 0;
}

// The HTILE bytes of a surface at least: one dword per 8x8 tile.
std::uint64_t htileBytes(const DepthTarget& target) {
    return static_cast<std::uint64_t>((target.extent.width + 7u) / 8u) * ((target.extent.height + 7u) / 8u) * 4u;
}

}

std::shared_ptr<DepthImage> CachedDepthImage(const Context& context, const DepthTarget& target) {
    const auto format = DepthAttachmentFormat(context, target);
    auto& store = Store();
    std::lock_guard lock(store.mutex);
    // Images of a device the driver replaced went with it: forgotten, not destroyed (a recorded
    // draw that still holds one then destroys nothing).
    for (auto it = store.entries.begin(); it != store.entries.end();) {
        if (alive(*it, context)) {
            ++it;
            continue;
        }
        it->image->Abandon();
        it = store.entries.erase(it);
        ++store.generation;
    }
    for (auto it = store.entries.begin(); it != store.entries.end(); ++it) {
        if (!alive(*it, context) || it->address != target.address || it->stencilAddress != target.stencilAddress || it->slice != target.slice || it->width != target.extent.width || it->height != target.extent.height || it->format != format) continue;
        store.entries.splice(store.entries.end(), store.entries, it);
        auto image = store.entries.back().image;
        // The HTILE address can move under the same surface; the latest is the one fills are matched
        // against. Same for the clear values: they come with each draw.
        image->SetHtileAddress(target.htileAddress);
        return image;
    }
    auto image = std::make_shared<DepthImage>(context, target, format);
    static const bool trace = std::getenv("APS5_TRACE_DEPTH") != nullptr;
    if (trace || store.entries.size() < 16) AgcDriver::ReportLine("[depth] resident depth image for 0x%llx layer %u (stencil 0x%llx, HTILE 0x%llx) %ux%u VkFormat %d, %zu resident\n", static_cast<unsigned long long>(target.address), target.slice, static_cast<unsigned long long>(target.stencilAddress), static_cast<unsigned long long>(target.htileAddress), target.extent.width, target.extent.height, static_cast<int>(format), store.entries.size() + 1);
    store.entries.push_back({context.device, context.bufferPool, target.address, target.stencilAddress, target.slice, target.extent.width, target.extent.height, format, image});
    ++store.generation;
    // Beyond the bound the least recently used image no recorded draw holds goes.
    constexpr std::size_t bound = 24;
    while (store.entries.size() > bound) {
        const auto victim = std::find_if(store.entries.begin(), store.entries.end(), [](const DepthStore::Entry& entry) { return entry.image.use_count() == 1; });
        if (victim == store.entries.end()) break;
        store.entries.erase(victim);
        ++store.generation;
    }
    store.count.store(store.entries.size(), std::memory_order_release);
    return image;
}

bool DepthImageCached(const Context& context, const DepthImage* image) {
    auto& store = Store();
    std::lock_guard lock(store.mutex);
    for (auto it = store.entries.begin(); it != store.entries.end(); ++it) {
        if (it->image.get() != image) continue;
        if (!alive(*it, context)) return false;
        store.entries.splice(store.entries.end(), store.entries, it);
        return true;
    }
    return false;
}

std::size_t NoteDepthMetadataFill(std::uint64_t address, std::size_t bytes, std::uint32_t pattern) {
    auto& store = Store();
    if (store.count.load(std::memory_order_acquire) == 0) return 0;
    std::lock_guard lock(store.mutex);
    std::size_t marked = 0;
    for (auto& entry : store.entries) {
        const auto& target = entry.image->Target();
        const auto htile = target.htileAddress;
        if (htile == 0 || address + bytes <= htile || htile + htileBytes(target) <= address) continue;
        // ZMASK (bits 0-3) 0 is a cleared tile in both HTILE layouts; 0xf an expanded one.
        const bool covers = address <= htile && address + bytes >= htile + htileBytes(target);
        const bool clear = covers && (pattern & 0xfu) == 0;
        if (clear) {
            entry.image->RequestClear(true);
            ++marked;
        }
        if (store.reportedFills.insert(htile).second) AgcDriver::ReportLine("[depth] fill 0x%llx+0x%zx with 0x%08x over the HTILE 0x%llx (0x%llx bytes) of depth 0x%llx: %s\n", static_cast<unsigned long long>(address), bytes, pattern, static_cast<unsigned long long>(htile), static_cast<unsigned long long>(htileBytes(target)), static_cast<unsigned long long>(target.address), clear ? "cleared" : covers ? "not a clear pattern, ignored" : "partial, ignored");
    }
    return marked;
}

std::uint32_t DepthTexelBytes(VkFormat format, VkImageAspectFlags aspect, VkFormat textureFormat) {
    if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) {
        const bool stencil = format == VK_FORMAT_S8_UINT || format == VK_FORMAT_D16_UNORM_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT;
        return stencil && (textureFormat == VK_FORMAT_R8_UINT || textureFormat == VK_FORMAT_R8_UNORM || textureFormat == VK_FORMAT_S8_UINT) ? 1u : 0u;
    }
    if (aspect != VK_IMAGE_ASPECT_DEPTH_BIT) return 0;
    if (format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D32_SFLOAT_S8_UINT) return textureFormat == VK_FORMAT_R32_SFLOAT || textureFormat == VK_FORMAT_D32_SFLOAT ? 4u : 0u;
    if (format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D16_UNORM_S8_UINT) return textureFormat == VK_FORMAT_R16_UNORM || textureFormat == VK_FORMAT_D16_UNORM ? 2u : 0u;
    return 0;
}

bool WritesDepthImage(const DepthState& state) {
    if (!state.attached) return false;
    if (state.depthWrite || state.clearDepth || state.clearStencil) return true;
    const auto stores = [](const VkStencilOpState& face) { return face.writeMask != 0 && (face.failOp != VK_STENCIL_OP_KEEP || face.passOp != VK_STENCIL_OP_KEEP || face.depthFailOp != VK_STENCIL_OP_KEEP); };
    return state.stencilTest && (stores(state.front) || stores(state.back));
}

VkImageAspectFlags SampledDepthAspect(const Context& context, const GuestTextureResource& resource) {
    auto& store = Store();
    if (!residentDepthSampling() || store.count.load(std::memory_order_acquire) == 0) return 0;
    VkImageAspectFlags aspect = 0;
    VkFormat format = VK_FORMAT_UNDEFINED;
    bool held = false;
    {
        std::lock_guard lock(store.mutex);
        for (const auto& entry : store.entries) {
            if (!alive(entry, context) || entry.width != resource.width || entry.height != resource.height) continue;
            const auto sampled = surfaceAspect(entry, resource.baseAddress);
            if (sampled == 0) continue;
            aspect = sampled;
            format = entry.format;
            held = held || entry.image->Holds(sampled);
        }
    }
    if (!held) return 0;
    const auto textureFormat = ResolveTextureFormat(resource.format);
    if (resource.dimension == TextureDimension::k3D || DepthTexelBytes(format, aspect, textureFormat) == 0) {
        char text[256];
        std::snprintf(text, sizeof(text), "AGC graphics: texture 0x%llx (%ux%u, guest format %u, VkFormat %d, dimension %d) samples the %s of resident depth images in VkFormat %d as other texels", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<int>(textureFormat), static_cast<int>(resource.dimension), aspect == VK_IMAGE_ASPECT_DEPTH_BIT ? "depth" : "stencil", static_cast<int>(format));
        throw std::runtime_error(text);
    }
    return aspect;
}

void SampleDepthSurface(const Context& context, const std::shared_ptr<Texture>& texture, const GuestTextureResource& resource, VkImageAspectFlags aspect) {
    Require(texture != nullptr && texture->Image() != VK_NULL_HANDLE && texture->ImageLayers() != 0, "a texture of a resident depth surface has no image of its own");
    if (texture->SamplesResidentDepth()) return;
    auto& store = Store();
    std::lock_guard lock(store.mutex);
    if (texture->SamplesResidentDepth()) return;
    texture->MarkResidentDepth();
    DepthStore::Sampled sampled;
    sampled.device = context.device;
    sampled.texture = texture;
    sampled.address = resource.baseAddress;
    sampled.aspect = aspect;
    sampled.width = resource.width;
    sampled.height = resource.height;
    sampled.layers.resize(texture->ImageLayers());
    store.sampled.push_back(std::move(sampled));
    store.sampledCount.store(store.sampled.size(), std::memory_order_release);
}

namespace {

struct LayerCopy {
    std::shared_ptr<Texture> texture;
    std::shared_ptr<DeviceBuffer> scratch;
    VkImageAspectFlags aspect;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t layer;
    std::uint32_t texelBytes;
    std::shared_ptr<DepthImage> image;
    std::uint64_t imageVersion;
    std::shared_ptr<StorageTexture> storage;
    std::uint64_t storageVersion;
};

void recordLayerCopy(const Context& context, Recorder& recorder, const LayerCopy& copy) {
    const auto commands = recorder.Commands();
    const auto timing = recorder.BeginGpuTiming(Recorder::CommandClass::Copy);
    recorder.Keep(copy.texture);
    recorder.Keep(copy.scratch);
    const auto barrier = context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier");
    const VkExtent3D extent{copy.width, copy.height, 1};
    if (copy.image != nullptr) {
        recorder.Keep(copy.image);
        copy.image->RecordCopyToBuffer(commands, copy.scratch->Handle(), copy.aspect);
    } else {
        recorder.Keep(copy.storage);
        const VkMemoryBarrier written{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
        barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &written, 0, nullptr, 0, nullptr);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, copy.layer, 1};
        region.imageExtent = extent;
        context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, copy.storage->Image(), VK_IMAGE_LAYOUT_GENERAL, copy.scratch->Handle(), 1, &region);
    }
    VkBufferMemoryBarrier staged{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    staged.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    staged.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    staged.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    staged.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    staged.buffer = copy.scratch->Handle();
    staged.size = VK_WHOLE_SIZE;
    VkImageMemoryBarrier destination{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    destination.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    destination.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    destination.oldLayout = copy.texture->Layout();
    destination.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    destination.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    destination.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    destination.image = copy.texture->Image();
    destination.subresourceRange = {copy.texture->ImageAspect(), 0, 1, copy.layer, 1};
    barrier(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &staged, 1, &destination);
    VkBufferImageCopy region{};
    region.imageSubresource = {copy.texture->ImageAspect(), 0, copy.layer, 1};
    region.imageExtent = extent;
    context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands, copy.scratch->Handle(), copy.texture->Image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    VkImageMemoryBarrier sampled = destination;
    sampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    sampled.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    sampled.newLayout = copy.texture->Layout();
    barrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &sampled);
    Recorder::CountBarriers(Recorder::CommandClass::Copy, 4);
    recorder.EndGpuTiming(timing, static_cast<std::uint64_t>(copy.width) * copy.height * copy.texelBytes);
}

}

// Technical debt: an HTILE clear still pending (no draw attached the image since the fill) is not in
// the copies; they take the image as it was before the fill.
void SyncDepthSurfaceTextures(const Context& context, const DepthImage* writing, std::span<const std::shared_ptr<Texture>> bound) {
    auto& store = Store();
    if (store.sampledCount.load(std::memory_order_acquire) == 0) return;
    if (std::none_of(bound.begin(), bound.end(), [](const std::shared_ptr<Texture>& texture) { return texture != nullptr && texture->SamplesResidentDepth(); })) return;
    Require(GuestMemory::GpuMutex().HeldByThisThread(), "textures of resident depth surfaces are brought up to date outside the GPU mutex");
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "textures of resident depth surfaces are brought up to date without a recorder");
    std::vector<LayerCopy> copies;
    {
        std::lock_guard lock(store.mutex);
        std::erase_if(store.sampled, [](const DepthStore::Sampled& sampled) { return sampled.texture.expired(); });
        store.sampledCount.store(store.sampled.size(), std::memory_order_release);
        for (auto& sampled : store.sampled) {
            if (sampled.device != context.device) continue;
            auto texture = sampled.texture.lock();
            if (texture == nullptr || std::find(bound.begin(), bound.end(), texture) == bound.end()) continue;
            if (sampled.generation != store.generation) {
                for (std::uint32_t layer = 0; layer < sampled.layers.size(); ++layer) {
                    const auto found = std::find_if(store.entries.begin(), store.entries.end(), [&](const DepthStore::Entry& entry) {
                        return alive(entry, context) && entry.slice == layer && entry.width == sampled.width && entry.height == sampled.height && surfaceAspect(entry, sampled.address) == sampled.aspect;
                    });
                    auto& state = sampled.layers[layer];
                    const auto image = found == store.entries.end() ? nullptr : found->image;
                    if (state.image.lock() != image) state = {};
                    state.image = image;
                }
                sampled.generation = store.generation;
            }
            for (std::uint32_t layer = 0; layer < sampled.layers.size(); ++layer) {
                auto& state = sampled.layers[layer];
                auto image = state.image.lock();
                if (image == nullptr) continue;
                LayerCopy copy{texture, nullptr, sampled.aspect, sampled.width, sampled.height, layer, DepthTexelBytes(image->Format(), sampled.aspect, texture->ImageFormat()), nullptr, 0, nullptr, 0};
                Require(copy.texelBytes != 0, "a texture of a resident depth surface changed its format");
                if (image->ContentHolder() == DepthImage::Holder::Memory && !image->Holds(sampled.aspect)) {
                    auto storage = image->Writer();
                    if (storage == nullptr || storage->Version() == state.storageVersion) continue;
                    const auto& described = storage->Descriptor();
                    Require(described.width == sampled.width && described.height == sampled.height && storage->ImageLayers() > layer && BytesPerElement(described.format) == copy.texelBytes, "the storage image that wrote a resident depth surface has other texels");
                    copy.storage = std::move(storage);
                    copy.storageVersion = copy.storage->Version();
                } else {
                    if (image.get() == writing || image->Version() == state.imageVersion) continue;
                    copy.imageVersion = image->Version();
                    copy.image = std::move(image);
                }
                const auto bytes = static_cast<std::size_t>(copy.width) * copy.height * copy.texelBytes;
                if (sampled.scratch == nullptr || sampled.scratch->Size() < bytes) sampled.scratch = std::make_shared<DeviceBuffer>(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                copy.scratch = sampled.scratch;
                if (copy.image != nullptr) {
                    state.imageVersion = copy.imageVersion;
                    state.storage.reset();
                    state.storageVersion = ~0ull;
                } else {
                    state.storage = copy.storage;
                    state.storageVersion = copy.storageVersion;
                    state.imageVersion = ~0ull;
                }
                copies.push_back(std::move(copy));
            }
        }
    }
    for (const auto& copy : copies) recordLayerCopy(context, *recorder, copy);
}

void PrepareDepthAttachment(const Context& context, DepthImage& image, const DepthState& state) {
    if (!residentDepthSampling() || image.ContentHolder() != DepthImage::Holder::Memory || image.FillClearPending()) return;
    auto storage = image.Writer();
    if (storage == nullptr) return;
    const auto& target = image.Target();
    const auto aspect = image.WriterAspect();
    const auto& described = storage->Descriptor();
    const auto texelBytes = DepthTexelBytes(image.Format(), aspect, ResolveTextureFormat(described.format));
    if (described.width != target.extent.width || described.height != target.extent.height || storage->ImageLayers() <= target.slice || texelBytes == 0) {
        char text[256];
        std::snprintf(text, sizeof(text), "AGC graphics: the depth target 0x%llx layer %u (%ux%u, VkFormat %d) takes the texels of the storage image 0x%llx (%ux%u, %u layers, guest format %u) that wrote it, which are other texels", static_cast<unsigned long long>(target.address), target.slice, target.extent.width, target.extent.height, static_cast<int>(image.Format()), static_cast<unsigned long long>(described.baseAddress), described.width, described.height, storage->ImageLayers(), described.format);
        throw std::runtime_error(text);
    }
    Require(GuestMemory::GpuMutex().HeldByThisThread(), "a depth target takes storage texels outside the GPU mutex");
    auto* recorder = Recorder::Active();
    Require(recorder != nullptr, "a depth target takes storage texels without a recorder");
    auto scratch = std::make_shared<DeviceBuffer>(context, static_cast<std::size_t>(target.extent.width) * target.extent.height * texelBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    const auto commands = recorder->Commands();
    const auto timing = recorder->BeginGpuTiming(Recorder::CommandClass::Copy);
    recorder->Keep(scratch);
    recorder->Keep(storage);
    if (image.ClearPending()) image.RecordPendingClear(commands, state);
    const VkMemoryBarrier written{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    context.Resolved(&DeviceFunctions::cmdPipelineBarrier, "vkCmdPipelineBarrier")(commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &written, 0, nullptr, 0, nullptr);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, target.slice, 1};
    region.imageExtent = {target.extent.width, target.extent.height, 1};
    context.Function<PFN_vkCmdCopyImageToBuffer>("vkCmdCopyImageToBuffer")(commands, storage->Image(), VK_IMAGE_LAYOUT_GENERAL, scratch->Handle(), 1, &region);
    image.RecordCopyFromBuffer(commands, scratch->Handle(), aspect);
    Recorder::CountBarriers(Recorder::CommandClass::Copy, 3);
    recorder->EndGpuTiming(timing, static_cast<std::uint64_t>(target.extent.width) * target.extent.height * texelBytes);
}

// Technical debt: the resident images never reach guest memory or the surface's storage images, so a
// CPU read, a copy or a storage access of a surface an image holds sees older bytes, and a storage
// write over part of such a surface leaves the rest of it stale.
void NoteDepthSurfaceStored(const std::shared_ptr<StorageTexture>& storage) {
    auto& store = Store();
    if (storage == nullptr || store.count.load(std::memory_order_acquire) == 0) return;
    const auto& described = storage->Descriptor();
    std::lock_guard lock(store.mutex);
    for (const auto& entry : store.entries) {
        if (entry.width != described.width || entry.height != described.height || entry.slice >= storage->ImageLayers()) continue;
        const auto aspect = surfaceAspect(entry, described.baseAddress);
        if (aspect == 0) continue;
        const bool texels = DepthTexelBytes(entry.format, aspect, ResolveTextureFormat(described.format)) != 0;
        entry.image->NoteStored(texels ? storage : nullptr, aspect);
    }
}

void CountDepthDraw(const DepthState& state) {
    auto& counts = Counts();
    std::lock_guard lock(counts.mutex);
    ++counts.draws;
    if (state.clearDepth || state.clearStencil) ++counts.clearDraws;
    if (state.stencilTest) ++counts.stencilDraws;
    const auto now = std::chrono::steady_clock::now();
    if (now - counts.lastReport < std::chrono::seconds(10)) return;
    counts.lastReport = now;
    AgcDriver::ReportLine("[depth] %llu draws with a depth attachment in 10 s (%llu clear draws, %llu with a stencil test), %llu whole-image clears\n", static_cast<unsigned long long>(counts.draws), static_cast<unsigned long long>(counts.clearDraws), static_cast<unsigned long long>(counts.stencilDraws), static_cast<unsigned long long>(counts.imageClears));
    counts.draws = counts.clearDraws = counts.stencilDraws = counts.imageClears = 0;
}

void ClearDepthImages(VkDevice device) {
    auto& store = Store();
    std::lock_guard lock(store.mutex);
    std::erase_if(store.entries, [&](const DepthStore::Entry& entry) { return entry.device == device; });
    store.count.store(store.entries.size(), std::memory_order_release);
    std::erase_if(store.sampled, [&](const DepthStore::Sampled& sampled) { return sampled.device == device; });
    store.sampledCount.store(store.sampled.size(), std::memory_order_release);
    ++store.generation;
}

}
