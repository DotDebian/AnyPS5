#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_VRAMLEDGER_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_VRAMLEDGER_HPP

#include "prx/libSceAgcDriver/Graphics/include/Context.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// APS5_TRACE_VRAM=1 (local, not for upstream): what device memory the driver holds, for the [vram]
// line (Recorder.cpp prints it every 30 s). Two views that do not depend on each other:
// - the ledger: every VkDeviceMemory the driver allocates is entered with its size, its memory
//   type and the class of the allocation site, and removed when freed; the totals per class are
//   split by heap kind (video: a device-local heap; system: the others, which host imports and
//   host-visible buffers use), so their sum can be set against VK_EXT_memory_budget's heapUsage;
// - the live Buffer objects by purpose (a staging shadow, a batch arena block, anything else),
//   counted when a Buffer is made and when it goes: a Buffer's memory may come from the pool, so
//   the ledger alone cannot say what the buffers alive are for.
// The caches' own figures (what their budgets count) are added by Recorder.cpp from each cache.
// Without the switch nothing is recorded: every function returns at once.
namespace AgcDriver::Graphics::Vram {

enum class Class : std::uint8_t { BufferMemory, SlabBlock, DeviceBuffer, RenderTarget, Depth, SampledImage, StorageImage, UnitShadowSlab, HostImport, Count };
inline constexpr const char* ClassNames[static_cast<std::size_t>(Class::Count)] = {"buffer allocations", "slab blocks", "device buffers", "render targets", "depth images", "sampled images", "storage images", "unit shadow slabs", "host imports"};
enum class Purpose : std::uint8_t { Other, StagingShadow, ArenaBlock, Count };
inline constexpr const char* PurposeNames[static_cast<std::size_t>(Purpose::Count)] = {"other", "staging shadows", "arena blocks"};

inline bool Traced() {
    static const bool traced = std::getenv("APS5_TRACE_VRAM") != nullptr;
    return traced;
}

struct Totals {
    std::atomic<std::uint64_t> objects{0}, videoBytes{0}, systemBytes{0};
};

struct Ledger {
    struct Entry {
        std::uint64_t bytes;
        Class owner;
        bool video;
    };
    std::mutex mutex;
    std::unordered_map<VkDeviceMemory, Entry> entries;
    std::array<Totals, static_cast<std::size_t>(Class::Count)> classes{};
    std::array<Totals, static_cast<std::size_t>(Purpose::Count)> purposes{};
    std::atomic<std::uint64_t> allocations{0}, frees{0};
    // The holders' own segments of the line, registered by each holder's source file (so the
    // recorder links without the holders a test leaves out).
    std::vector<std::string (*)(const Context&)> sections;
};

inline Ledger& State() {
    static Ledger ledger;
    return ledger;
}

struct SectionRegistration {
    explicit SectionRegistration(std::string (*section)(const Context&)) {
        auto& ledger = State();
        std::lock_guard lock(ledger.mutex);
        ledger.sections.push_back(section);
    }
};

inline void Allocated(const Context& context, VkDeviceMemory memory, VkDeviceSize bytes, std::uint32_t memoryType, Class owner) {
    if (!Traced() || memory == VK_NULL_HANDLE) return;
    const bool video = memoryType < context.memory.memoryTypeCount && (context.memory.memoryHeaps[context.memory.memoryTypes[memoryType].heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    auto& ledger = State();
    {
        std::lock_guard lock(ledger.mutex);
        ledger.entries[memory] = {bytes, owner, video};
    }
    auto& totals = ledger.classes[static_cast<std::size_t>(owner)];
    totals.objects.fetch_add(1, std::memory_order_relaxed);
    (video ? totals.videoBytes : totals.systemBytes).fetch_add(bytes, std::memory_order_relaxed);
    ledger.allocations.fetch_add(1, std::memory_order_relaxed);
}

// Before vkFreeMemory. A memory the ledger does not hold (allocated by a site it does not cover)
// is ignored.
inline void Freed(VkDeviceMemory memory) {
    if (!Traced() || memory == VK_NULL_HANDLE) return;
    auto& ledger = State();
    Ledger::Entry entry{};
    {
        std::lock_guard lock(ledger.mutex);
        const auto found = ledger.entries.find(memory);
        if (found == ledger.entries.end()) return;
        entry = found->second;
        ledger.entries.erase(found);
    }
    auto& totals = ledger.classes[static_cast<std::size_t>(entry.owner)];
    totals.objects.fetch_sub(1, std::memory_order_relaxed);
    (entry.video ? totals.videoBytes : totals.systemBytes).fetch_sub(entry.bytes, std::memory_order_relaxed);
    ledger.frees.fetch_add(1, std::memory_order_relaxed);
}

inline Purpose& ThreadPurpose() {
    thread_local Purpose purpose = Purpose::Other;
    return purpose;
}

// The purpose of the Buffers the calling thread makes inside the scope.
class PurposeScope {
public:
    explicit PurposeScope(Purpose purpose) : previous(ThreadPurpose()) { ThreadPurpose() = purpose; }
    ~PurposeScope() { ThreadPurpose() = previous; }
    PurposeScope(const PurposeScope&) = delete;
    PurposeScope& operator=(const PurposeScope&) = delete;

private:
    Purpose previous;
};

inline void BufferLives(Purpose purpose, std::uint64_t bytes, bool video, bool born) {
    if (!Traced()) return;
    auto& totals = State().purposes[static_cast<std::size_t>(purpose)];
    if (born) {
        totals.objects.fetch_add(1, std::memory_order_relaxed);
        (video ? totals.videoBytes : totals.systemBytes).fetch_add(bytes, std::memory_order_relaxed);
    } else {
        totals.objects.fetch_sub(1, std::memory_order_relaxed);
        (video ? totals.videoBytes : totals.systemBytes).fetch_sub(bytes, std::memory_order_relaxed);
    }
}

// "<name> objects/video MiB/system MiB" for every class or purpose with an object, and the sums.
template <std::size_t N>
std::string Describe(const std::array<Totals, N>& totals, const char* const* names, std::uint64_t* video = nullptr, std::uint64_t* system = nullptr) {
    std::string text;
    char piece[96];
    for (std::size_t index = 0; index < N; ++index) {
        const auto objects = totals[index].objects.load(std::memory_order_relaxed);
        const auto videoBytes = totals[index].videoBytes.load(std::memory_order_relaxed);
        const auto systemBytes = totals[index].systemBytes.load(std::memory_order_relaxed);
        if (video != nullptr) *video += videoBytes;
        if (system != nullptr) *system += systemBytes;
        if (objects == 0) continue;
        std::snprintf(piece, sizeof(piece), " %s %llu/%.0f/%.0f", names[index], static_cast<unsigned long long>(objects), videoBytes / 1048576.0, systemBytes / 1048576.0);
        text += piece;
    }
    return text;
}

}

#endif
