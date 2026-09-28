#include "prx/libSceAgcDriver/Graphics/include/DepthTarget.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <mutex>
#include <set>
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
        if (format != candidates.front()) std::fprintf(stderr, "[gpu] depth format Z%u%s renders as VkFormat %d on this device\n", target.zFormat, target.stencil ? "+S8" : "", static_cast<int>(format));
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

bool DepthImage::RecordPendingClear(VkCommandBuffer commands, const DepthState& state) {
    if (!clearPending.exchange(false, std::memory_order_acq_rel)) return false;
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
    std::mutex mutex;
    // Least recently used first.
    std::list<Entry> entries;
    // HTILE buffers and sampled surfaces already reported.
    std::set<std::uint64_t> reportedFills;
    std::set<std::uint64_t> reportedSamples;
    std::atomic<std::size_t> count{0};
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
    if (trace || store.entries.size() < 16) std::fprintf(stderr, "[depth] resident depth image for 0x%llx layer %u (stencil 0x%llx, HTILE 0x%llx) %ux%u VkFormat %d, %zu resident\n", static_cast<unsigned long long>(target.address), target.slice, static_cast<unsigned long long>(target.stencilAddress), static_cast<unsigned long long>(target.htileAddress), target.extent.width, target.extent.height, static_cast<int>(format), store.entries.size() + 1);
    store.entries.push_back({context.device, context.bufferPool, target.address, target.stencilAddress, target.slice, target.extent.width, target.extent.height, format, image});
    // Beyond the bound the least recently used image no recorded draw holds goes.
    constexpr std::size_t bound = 24;
    while (store.entries.size() > bound) {
        const auto victim = std::find_if(store.entries.begin(), store.entries.end(), [](const DepthStore::Entry& entry) { return entry.image.use_count() == 1; });
        if (victim == store.entries.end()) break;
        store.entries.erase(victim);
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
            entry.image->RequestClear();
            ++marked;
        }
        if (store.reportedFills.insert(htile).second) std::fprintf(stderr, "[depth] fill 0x%llx+0x%zx with 0x%08x over the HTILE 0x%llx (0x%llx bytes) of depth 0x%llx: %s\n", static_cast<unsigned long long>(address), bytes, pattern, static_cast<unsigned long long>(htile), static_cast<unsigned long long>(htileBytes(target)), static_cast<unsigned long long>(target.address), clear ? "cleared" : covers ? "not a clear pattern, ignored" : "partial, ignored");
    }
    return marked;
}

void NoteDepthSurfaceSampled(std::uint64_t address) {
    auto& store = Store();
    if (store.count.load(std::memory_order_acquire) == 0) return;
    std::lock_guard lock(store.mutex);
    for (const auto& entry : store.entries) {
        if (entry.address != address && entry.stencilAddress != address) continue;
        if (store.reportedSamples.insert(address).second) std::fprintf(stderr, "[depth] a texture samples the %s surface 0x%llx of a resident depth image: it reads guest memory, which the depth image does not update\n", entry.address == address && entry.image->Target().zFormat != 0 ? "depth" : "stencil", static_cast<unsigned long long>(address));
        return;
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
    std::fprintf(stderr, "[depth] %llu draws with a depth attachment in 10 s (%llu clear draws, %llu with a stencil test), %llu whole-image clears\n", static_cast<unsigned long long>(counts.draws), static_cast<unsigned long long>(counts.clearDraws), static_cast<unsigned long long>(counts.stencilDraws), static_cast<unsigned long long>(counts.imageClears));
    counts.draws = counts.clearDraws = counts.stencilDraws = counts.imageClears = 0;
}

void ClearDepthImages(VkDevice device) {
    auto& store = Store();
    std::lock_guard lock(store.mutex);
    std::erase_if(store.entries, [&](const DepthStore::Entry& entry) { return entry.device == device; });
    store.count.store(store.entries.size(), std::memory_order_release);
}

}
