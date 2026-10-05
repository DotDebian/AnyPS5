#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESOURCES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_RESOURCES_HPP

#include "prx/libSceAgcDriver/Graphics/include/State.hpp"
#include <span>

namespace AgcDriver::Graphics {

class Buffer {
public:
    Buffer(const Context& context, std::size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    ~Buffer();
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    VkBuffer Handle() const;
    VkDeviceAddress DeviceAddress() const;
    // Host-visible buffers only: a device-local one (properties without HOST_VISIBLE, the staging
    // shadows of GuestBufferMemory) has no mapping and its bytes move by GPU copies alone.
    std::span<std::byte> Bytes();
    bool Mapped() const { return mapping != nullptr; }
    // Whether the buffer's memory was allocated from a mappable video memory type: false for one
    // asked for with VideoMemoryProperties() that fell back to system memory.
    bool InVideoMemory() const { return video; }
    void Invalidate();

private:
    void initializeAddress(VkBufferUsageFlags usage);
    void release() noexcept;
    Context context;
    VkDeviceAddress deviceAddress = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapping = nullptr;
    std::size_t size;
    // The VkBuffer's own size, `size` rounded up to its pool class (see BufferPool::Capacity).
    std::size_t capacity;
    // Fully made (or taken from the pool), so release returns it to the pool instead of destroying
    // the handles a failed construction left.
    bool ready = false;
    VkDeviceSize allocationBytes = 0;
    VkDeviceSize offset = 0;
    bool slab = false;
    bool video = false;
    VkBufferUsageFlags usage;
    VkMemoryPropertyFlags properties;
    std::shared_ptr<BufferPool> cache;
};

// APS5_VRAM_BUFFERS=<mask> (local experiment, not for upstream): buffers the shaders read are made in
// video memory the CPU can still map (DEVICE_LOCAL | HOST_VISIBLE, the whole heap with a resizable
// BAR) instead of cached system memory, which the GPU reads across PCIe. The CPU then writes them
// through a write-combined mapping and must not read them back in bulk. One bit per kind, so a run
// tells which kind the GPU time depends on; Everything moves every default host buffer. A kind whose
// bit is clear, or a device without such a memory type, keeps the default.
enum class GpuReadKind : unsigned { BdaTable = 1, DrawInput = 2, StorageCopy = 4, RegionCopy = 8, Mirror = 16, Everything = 128 };
VkMemoryPropertyFlags GpuReadProperties(GpuReadKind kind);
// The properties of such video memory, whatever the mask says: for a buffer placed there by its own
// switch (APS5_VRAM_MIRROR, see GuestBufferMemory.cpp). It falls back to system memory like the others.
VkMemoryPropertyFlags VideoMemoryProperties();

// Device-local scratch memory for GPU-side layout conversion. The detiler reads and writes scattered
// elements, which crawls across PCIe, so guest bytes move between host and device buffers with DMA
// copies and are only swizzled in video memory.
class DeviceBuffer {
public:
    DeviceBuffer(const Context& context, std::size_t size, VkBufferUsageFlags usage);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    VkBuffer Handle() const;
    std::size_t Size() const;

private:
    void release() noexcept;
    Context context;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    std::size_t size;
    std::size_t capacity;
    VkDeviceSize allocationBytes = 0;
    VkBufferUsageFlags usage;
    std::shared_ptr<BufferPool> cache;
};

// Records a whole-range buffer copy.
void CopyBuffer(const Context& context, VkCommandBuffer commands, VkBuffer source, VkDeviceSize sourceOffset, VkBuffer destination, VkDeviceSize destinationOffset, VkDeviceSize bytes);

// Records a global memory barrier.
void RecordMemoryBarrier(const Context& context, VkCommandBuffer commands, VkPipelineStageFlags sourceStage, VkPipelineStageFlags destinationStage, VkAccessFlags sourceAccess, VkAccessFlags destinationAccess);

// Resolves every DeviceFunctions entry point of the context's device (once, at device setup).
void FillDeviceFunctions(const Context& context, DeviceFunctions& functions);

class RenderTarget {
public:
    RenderTarget(const Context& context, const ColorTarget& target, bool blending);
    ~RenderTarget();
    RenderTarget(const RenderTarget&) = delete;
    RenderTarget& operator=(const RenderTarget&) = delete;
    VkImage Image() const;
    VkImageView View() const;

private:
    void release() noexcept;
    Context context;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

class CommandBatch {
public:
    explicit CommandBatch(const Context& context);
    ~CommandBatch();
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;
    VkCommandBuffer Handle() const;
    void SubmitAndWait();
    void Submit();
    void Wait();

private:
    void release() noexcept;
    Context context;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false;
    bool submitted = false;
};

}

#endif
