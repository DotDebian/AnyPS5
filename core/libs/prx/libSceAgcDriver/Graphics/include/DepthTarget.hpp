#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEPTHTARGET_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_DEPTHTARGET_HPP

#include "prx/libSceAgcDriver/Graphics/include/GuestTextureResource.hpp"
#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>

namespace AgcDriver::Graphics {

class StorageTexture;
class Texture;

// The Vulkan format a depth target renders in on this device: Z_16 is D16_UNORM and Z_32_FLOAT is
// D32_SFLOAT; with STENCIL_8 they become D16_UNORM_S8_UINT / D32_SFLOAT_S8_UINT, else the first of
// D24_UNORM_S8_UINT and D32_SFLOAT_S8_UINT the device can attach, and a stencil-only surface is
// S8_UINT with the same fallbacks. Throws when none can be attached.
VkFormat DepthAttachmentFormat(const Context& context, const DepthTarget& target);
VkImageAspectFlags DepthAspects(VkFormat format);

// The GPU-resident contents of one depth/stencil surface. Nothing writes the image back to the
// surface's memory; textures sampling the surface take the image's texels (SampleDepthSurface). The
// image stays in the GENERAL layout, so draws need no transitions; its contents start undefined and
// are cleared to the first user's DB_DEPTH_CLEAR / DB_STENCIL_CLEAR before anything renders into it.
class DepthImage {
public:
    DepthImage(const Context& context, const DepthTarget& target, VkFormat format);
    ~DepthImage();
    DepthImage(const DepthImage&) = delete;
    DepthImage& operator=(const DepthImage&) = delete;
    VkImage Image() const { return image; }
    VkImageView View() const { return view; }
    VkFormat Format() const { return format; }
    VkImageAspectFlags Aspects() const { return aspects; }
    const DepthTarget& Target() const { return target; }
    // The HTILE buffer fills are matched against (the latest draw's; under the store's mutex).
    void SetHtileAddress(std::uint64_t address) { target.htileAddress = address; }
    // Whether the next draw clears the image first: its contents are undefined, or a fill of the
    // surface's HTILE buffer (NoteDepthMetadataFill) cleared the surface.
    bool ClearPending() const { return clearPending.load(std::memory_order_acquire); }
    bool FillClearPending() const { return ClearPending() && fillClear.load(std::memory_order_acquire); }
    void RequestClear(bool fill = false) {
        fillClear.store(fill, std::memory_order_release);
        clearPending.store(true, std::memory_order_release);
    }
    enum class Holder : std::uint8_t { Memory, Image, Both };
    Holder ContentHolder() const { return holder.load(std::memory_order_acquire); }
    std::shared_ptr<StorageTexture> Writer() const;
    VkImageAspectFlags WriterAspect() const { return writerAspect.load(std::memory_order_acquire); }
    bool Holds(VkImageAspectFlags aspect) const {
        const auto written = WriterAspect();
        return ContentHolder() == Holder::Image || (written != 0 && aspect != written);
    }
    std::uint64_t Version() const { return version.load(std::memory_order_acquire); }
    void NoteWritten();
    void NoteStored(const std::shared_ptr<StorageTexture>& storage, VkImageAspectFlags aspect);
    void RecordCopyToBuffer(VkCommandBuffer commands, VkBuffer buffer, VkImageAspectFlags aspect) const;
    void RecordCopyFromBuffer(VkCommandBuffer commands, VkBuffer buffer, VkImageAspectFlags aspect);
    // Records the pending clear of the whole image to the state's clear values, outside a render
    // pass, between barriers that order it after earlier attachment and transfer work and before
    // later depth tests. Returns whether a clear was recorded.
    bool RecordPendingClear(VkCommandBuffer commands, const DepthState& state);
    // Forgets the Vulkan objects without destroying them (the device is already gone).
    void Abandon() noexcept;

private:
    void release() noexcept;
    Context context;
    DepthTarget target;
    VkFormat format;
    VkImageAspectFlags aspects;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    // UNDEFINED until the first clear moved the image to GENERAL.
    bool initialized = false;
    std::atomic<bool> clearPending{true};
    std::atomic<bool> fillClear{false};
    std::atomic<Holder> holder{Holder::Memory};
    std::atomic<std::uint64_t> version{0};
    mutable std::mutex writerMutex;
    std::weak_ptr<StorageTexture> writer;
    std::atomic<VkImageAspectFlags> writerAspect{0};
    std::atomic<std::uint32_t> holding{0};
    void noteHolding();
};

std::uint32_t DepthTexelBytes(VkFormat format, VkImageAspectFlags aspect, VkFormat textureFormat);
std::uint64_t DepthHoldingGeneration();
bool WritesDepthImage(const DepthState& state);

// The resident image of a depth target, shared by draws and frames: per device, keyed by the surface
// (depth and stencil base, extent) and the attachment format, least recently used first out beyond a
// bound (only images no recorded draw holds). Used under GuestMemory::GpuMutex like the draws.
std::shared_ptr<DepthImage> CachedDepthImage(const Context& context, const DepthTarget& target);
// Whether the image is still the store's (a draw recipe's stored image): otherwise the lookup would
// return another one.
bool DepthImageCached(const Context& context, const DepthImage* image);
// A fill of guest memory (the fill HLE; `pattern` its first dword) that covers the HTILE buffer of a
// resident depth image: a ZMASK of 0 in the HTILE word is the DB's "cleared" state, so the image's
// clear becomes pending (the values are the next user's DB_DEPTH_CLEAR / DB_STENCIL_CLEAR, which is
// what the DB reads for cleared tiles). Returns how many images were marked; the first fill per
// HTILE buffer is reported on stderr.
std::size_t NoteDepthMetadataFill(std::uint64_t address, std::size_t bytes, std::uint32_t pattern);
VkImageAspectFlags SampledDepthAspect(const Context& context, const GuestTextureResource& resource);
void SampleDepthSurface(const Context& context, const std::shared_ptr<Texture>& texture, const GuestTextureResource& resource, VkImageAspectFlags aspect);
void SyncDepthSurfaceTextures(const Context& context, const DepthImage* writing, std::span<const std::shared_ptr<Texture>> bound);
void PrepareDepthAttachment(const Context& context, DepthImage& image, const DepthState& state);
void NoteDepthSurfaceStored(const std::shared_ptr<StorageTexture>& storage);
// A draw recorded with a depth attachment, for the "[depth]" line printed every 10 s while such
// draws happen: draws, clear draws (DB_RENDER_CONTROL) and whole-image clears (new images, HTILE fills).
void CountDepthDraw(const DepthState& state);
// Destroys the resident depth images of a device; to be called before the device goes away.
void ClearDepthImages(VkDevice device);

}

#endif
