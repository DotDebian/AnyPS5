#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINECACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_PIPELINECACHE_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <mutex>
#include <thread>
#include <vector>

namespace AgcDriver::Graphics {

// The device's VkPipelineCache, persisted across runs beside the shader disk cache
// (ShaderCacheDirectory() in core/shader/recompiler/ShaderCacheDirectory.hpp): the driver's
// compiled pipelines for a device are loaded at device creation and saved at teardown and, since a
// run usually ends by a kill, every SaveIntervalSeconds while the cache grew. The file is named by
// the vendor, device, driver version and pipelineCacheUUID, and its data is only handed to Vulkan
// when its checksum and its Vulkan header match this device. APS5_NO_SHADER_DISK_CACHE=1 disables
// the persistence (the cache then starts empty, as before).
class PipelineCache {
public:
    static constexpr int SaveIntervalSeconds = 10;

    PipelineCache(const Context& context, const VkPhysicalDeviceProperties& properties);
    ~PipelineCache();

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;
    VkPipelineCache Handle() const { return cache; }

private:
    void load(std::vector<std::byte>& initialData);
    void save(bool final);
    void run();

    Context context;
    VkPhysicalDeviceProperties properties{};
    VkPipelineCache cache = VK_NULL_HANDLE;
    std::filesystem::path path;
    // The size of the data last written (or loaded); a save only writes a cache that changed.
    std::size_t savedBytes = 0;
    std::mutex mutex;
    std::condition_variable wake;
    bool stopping = false;
    std::thread saver;
};

}

#endif
