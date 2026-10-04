#include <cstdint>
#include <cstddef>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "SceTypes.hpp"
#include "prx/libc/include/General.hpp"

namespace {

constexpr std::uint32_t SystemMemoryDescriptorId = 0x010107d4;
constexpr std::size_t SystemCpuMemoryBytes = 64;
constexpr std::uint32_t RenderParamDescriptorId = 0x010107d6;

enum class ObjectKind { System, Material, Room, Portal, Source };

struct Attribute {
    std::uint32_t id;
    std::uint32_t reserved;
    const void* data;
    std::size_t size;
    std::uint32_t flags;
    std::uint32_t padding;
};
static_assert(sizeof(Attribute) == 0x20);

struct Object {
    ObjectKind kind;
    AudioPropagationHandle system;
    std::map<std::uint32_t, std::vector<std::uint8_t>> attributes;
};

struct State {
    std::mutex mutex;
    AudioPropagationHandle next = 1;
    std::map<AudioPropagationHandle, Object> objects;
};

State& Objects() {
    static State state;
    return state;
}

[[noreturn]] void Fail(const char* function, const char* reason) {
    throw std::invalid_argument(std::string(function) + ": " + reason);
}

Object& Find(State& state, AudioPropagationHandle handle, ObjectKind kind, const char* function) {
    const auto found = state.objects.find(handle);
    if (found == state.objects.end() || found->second.kind != kind) Fail(function, "invalid handle");
    return found->second;
}

int32_t Create(AudioPropagationHandle system, ObjectKind kind, AudioPropagationHandle* out, const char* function) {
    if (out == nullptr) Fail(function, "null output handle");
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    Find(state, system, ObjectKind::System, function);
    const auto handle = state.next++;
    state.objects.emplace(handle, Object{kind, system, {}});
    *out = handle;
    return 0;
}

int32_t Destroy(AudioPropagationHandle system, AudioPropagationHandle handle, ObjectKind kind, const char* function) {
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    Find(state, system, ObjectKind::System, function);
    if (Find(state, handle, kind, function).system != system) Fail(function, "object belongs to another system");
    state.objects.erase(handle);
    return 0;
}

int32_t SetAttributes(AudioPropagationHandle handle, ObjectKind kind, const Attribute* attributes, uint32_t count, const char* function) {
    if (attributes == nullptr && count != 0) Fail(function, "null attributes");
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    auto& object = Find(state, handle, kind, function);
    for (uint32_t i = 0; i < count; ++i) {
        const auto& attribute = attributes[i];
        if (attribute.data == nullptr && attribute.size != 0) Fail(function, "null attribute data");
        const auto* bytes = static_cast<const std::uint8_t*>(attribute.data);
        object.attributes[attribute.id].assign(bytes, bytes + attribute.size);
    }
    return 0;
}

}

extern "C" {

int32_t APS5_VABI sceAudioPropagationSystemQueryMemory(const void* options, AudioPropagationSystemMemory* out_memory) {
    if (options == nullptr || out_memory == nullptr) Fail(__func__, "null argument");
    if (out_memory->desc.id != SystemMemoryDescriptorId || out_memory->desc.size != sizeof(AudioPropagationSystemMemory)) Fail(__func__, "unknown memory descriptor");
    out_memory->size_cpu_mem = SystemCpuMemoryBytes;
    out_memory->size_gpu_mem = 0;
    return 0;
}

int32_t APS5_VABI sceAudioPropagationSystemCreate(const void* options, AudioPropagationSystemMemory* memory, AudioPropagationHandle* out_system_handle) {
    if (options == nullptr || memory == nullptr || out_system_handle == nullptr) Fail(__func__, "null argument");
    if (memory->desc.id != SystemMemoryDescriptorId || memory->p_cpu_mem == nullptr || memory->size_cpu_mem < SystemCpuMemoryBytes) Fail(__func__, "system memory was not provided");
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    const auto handle = state.next++;
    state.objects.emplace(handle, Object{ObjectKind::System, handle, {}});
    *out_system_handle = handle;
    return 0;
}

int32_t APS5_VABI sceAudioPropagationSystemDestroy(AudioPropagationHandle system) {
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    Find(state, system, ObjectKind::System, __func__);
    std::erase_if(state.objects, [&](const auto& entry) { return entry.second.system == system; });
    return 0;
}

int32_t APS5_VABI sceAudioPropagationSystemSetAttributes(AudioPropagationHandle system, const Attribute* attributes, uint32_t count) {
    return SetAttributes(system, ObjectKind::System, attributes, count, __func__);
}

int32_t APS5_VABI sceAudioPropagationSystemRegisterMaterial(AudioPropagationHandle system, const void* material, AudioPropagationHandle* out_material) {
    if (material == nullptr) Fail(__func__, "null material");
    return Create(system, ObjectKind::Material, out_material, __func__);
}

int32_t APS5_VABI sceAudioPropagationSystemUnregisterMaterial(AudioPropagationHandle material) {
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    Find(state, material, ObjectKind::Material, __func__);
    state.objects.erase(material);
    return 0;
}

int32_t APS5_VABI sceAudioPropagationRoomCreate(AudioPropagationHandle system, AudioPropagationHandle* out_room) {
    return Create(system, ObjectKind::Room, out_room, __func__);
}

int32_t APS5_VABI sceAudioPropagationRoomDestroy(AudioPropagationHandle system, AudioPropagationHandle room) {
    return Destroy(system, room, ObjectKind::Room, __func__);
}

int32_t APS5_VABI sceAudioPropagationPortalCreate(AudioPropagationHandle system, const void* params, AudioPropagationHandle* out_portal) {
    if (params == nullptr) Fail(__func__, "null portal parameters");
    return Create(system, ObjectKind::Portal, out_portal, __func__);
}

int32_t APS5_VABI sceAudioPropagationPortalDestroy(AudioPropagationHandle system, AudioPropagationHandle portal) {
    return Destroy(system, portal, ObjectKind::Portal, __func__);
}

int32_t APS5_VABI sceAudioPropagationPortalSetAttributes(AudioPropagationHandle portal, const Attribute* attributes, uint32_t count) {
    return SetAttributes(portal, ObjectKind::Portal, attributes, count, __func__);
}

int32_t APS5_VABI sceAudioPropagationSourceCreate(AudioPropagationHandle system, AudioPropagationHandle* out_source) {
    return Create(system, ObjectKind::Source, out_source, __func__);
}

int32_t APS5_VABI sceAudioPropagationSourceDestroy(AudioPropagationHandle system, AudioPropagationHandle source) {
    return Destroy(system, source, ObjectKind::Source, __func__);
}

int32_t APS5_VABI sceAudioPropagationSourceSetAttributes(AudioPropagationHandle source, const Attribute* attributes, uint32_t count) {
    return SetAttributes(source, ObjectKind::Source, attributes, count, __func__);
}

int32_t APS5_VABI sceAudioPropagationSystemGetRays(AudioPropagationHandle system, void* rays, uint32_t* count) {
    if (rays == nullptr || count == nullptr) Fail(__func__, "null argument");
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    Find(state, system, ObjectKind::System, __func__);
    *count = 0;
    return 0;
}

int32_t APS5_VABI sceAudioPropagationSystemSetRays(AudioPropagationHandle system, const void* rays, uint32_t count) {
    if (count != 0) Fail(__func__, "rays were returned that were never requested");
    (void)rays;
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    Find(state, system, ObjectKind::System, __func__);
    return 0;
}

int32_t APS5_VABI sceAudioPropagationSourceGetRays(AudioPropagationHandle source, void* rays, uint32_t* count) {
    if (rays == nullptr || count == nullptr) Fail(__func__, "null argument");
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    Find(state, source, ObjectKind::Source, __func__);
    *count = 0;
    return 0;
}

int32_t APS5_VABI sceAudioPropagationSourceGetAudioPathCount(AudioPropagationHandle source, uint32_t* count) {
    if (count == nullptr) Fail(__func__, "null argument");
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    Find(state, source, ObjectKind::Source, __func__);
    *count = 0;
    return 0;
}

int32_t APS5_VABI sceAudioPropagationSourceGetAudioPath(AudioPropagationHandle source, uint32_t index, AudioPropagationHandle* out_path) {
    (void)index;
    (void)out_path;
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    Find(state, source, ObjectKind::Source, __func__);
    Fail(__func__, "path index out of range");
}

int32_t APS5_VABI sceAudioPropagationSourceCalculateAudioPaths(AudioPropagationHandle source, const void* rays, uint32_t ray_count, uint32_t flags, void* paths, uint32_t path_count) {
    (void)rays;
    (void)flags;
    (void)paths;
    if (ray_count != 0 || path_count != 0) Fail(__func__, "rays or paths were passed that were never requested");
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    Find(state, source, ObjectKind::Source, __func__);
    return 0;
}

int32_t APS5_VABI sceAudioPropagationSourceSetAudioPaths(AudioPropagationHandle source, const void* paths, uint32_t count) {
    (void)paths;
    if (count != 0) Fail(__func__, "paths were passed that were never requested");
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    Find(state, source, ObjectKind::Source, __func__);
    return 0;
}

struct RenderParam {
    std::uint32_t id;
    std::uint32_t reserved;
    std::uint64_t size;
    AudioPropagationHandle source;
    void* output;
    std::uint64_t outputBytes;
    std::uint32_t format;
    std::uint32_t padding;
};
static_assert(sizeof(RenderParam) == 0x30);

int32_t APS5_VABI sceAudioPropagationSourceRender(AudioPropagationHandle system, const RenderParam* params, uint32_t count) {
    if (params == nullptr && count != 0) Fail(__func__, "null parameters");
    auto& state = Objects();
    std::lock_guard lock(state.mutex);
    Find(state, system, ObjectKind::System, __func__);
    for (uint32_t i = 0; i < count; ++i) {
        const auto& param = params[i];
        if (param.id != RenderParamDescriptorId || param.size != sizeof(RenderParam)) Fail(__func__, "unknown render descriptor");
        if (Find(state, param.source, ObjectKind::Source, __func__).system != system) Fail(__func__, "source belongs to another system");
        if (param.output == nullptr && param.outputBytes != 0) Fail(__func__, "null output");
        std::memset(param.output, 0, param.outputBytes);
    }
    return 0;
}


}
