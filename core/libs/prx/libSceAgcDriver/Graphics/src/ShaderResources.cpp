#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include "prx/libSceAgcDriver/Execution/include/Report.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthTarget.hpp"
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureTiling.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DccMetadata.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureResidency.hpp"
#include "prx/libc/include/General.hpp"
#include "Optimization/include/Optimization/ShaderStageInputInfo.hpp"
#include "RdnaDecoder/include/RdnaDecoder/RdnaDescriptorFormat.hpp"
#include <cstring>
#include <limits>
#include <list>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include "prx/libc/include/GuestAllocations.hpp"

namespace AgcDriver::Graphics {
namespace {

bool overlap(std::uint64_t first, std::size_t firstSize, std::uint64_t second, std::size_t secondSize) {
    return first < second + secondSize && second < first + firstSize;
}

VkComponentSwizzle ComponentSwizzleFor(std::uint8_t dstSel) {
    switch (dstSel) {
        case 0: return VK_COMPONENT_SWIZZLE_ZERO;
        case 1: return VK_COMPONENT_SWIZZLE_ONE;
        case 4: return VK_COMPONENT_SWIZZLE_R;
        case 5: return VK_COMPONENT_SWIZZLE_G;
        case 6: return VK_COMPONENT_SWIZZLE_B;
        case 7: return VK_COMPONENT_SWIZZLE_A;
        default: throw std::runtime_error("AGC graphics: guest texture descriptor has an invalid destination channel selector " + std::to_string(dstSel));
    }
}

// Sampled textures are reused across draws and dispatches while their guest bytes are unchanged; a
// byte copy of the guest surface validates each reuse, so CPU or GPU writes to it force a re-upload.
// The key is everything a lookup matches: the device, the eight descriptor words and the swizzle.
struct TextureKey {
    VkDevice device;
    std::array<std::uint32_t, 8> words;
    std::array<std::uint32_t, 4> components;
    bool depthCompare = false;
    bool operator==(const TextureKey&) const = default;
};

std::uint64_t hashWords(std::uint64_t hash, const std::uint32_t* words, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) hash = (hash ^ words[i]) * 1099511628211ull;
    return hash;
}

struct TextureKeyHash {
    std::size_t operator()(const TextureKey& key) const noexcept {
        auto hash = 14695981039346656037ull ^ reinterpret_cast<std::uintptr_t>(key.device);
        hash = hashWords(hash, key.words.data(), key.words.size()) ^ static_cast<std::uint64_t>(key.depthCompare);
        return static_cast<std::size_t>(hashWords(hash, key.components.data(), key.components.size()));
    }
};

TextureKey MakeTextureKey(VkDevice device, std::span<const std::uint32_t> words, VkComponentMapping components, bool depthCompare = false) {
    TextureKey key{device, {}, {static_cast<std::uint32_t>(components.r), static_cast<std::uint32_t>(components.g), static_cast<std::uint32_t>(components.b), static_cast<std::uint32_t>(components.a)}};
    key.depthCompare = depthCompare;
    std::copy(words.begin(), words.end(), key.words.begin());
    return key;
}

struct CachedTexture {
    TextureKey key;
    std::uint64_t address;
    std::vector<std::byte> bytes;
    std::shared_ptr<Texture> texture;
    // A fast-cleared surface is cached as its clear texels; it stays valid while the keys are unchanged.
    DccKeys keys = DccKeys::Uncompressed;
    // Write generation the snapshot is known current at (see GuestMemory::CollectWrites).
    std::uint64_t generation = 0;
    // A texture viewing a storage image on the GPU has no snapshot; it stays valid while that image
    // is still its surface's cached image and nothing wrote the guest memory since it matched.
    std::shared_ptr<StorageTexture> source;
    std::uint64_t sourceVersion = 0;
    std::uint64_t accounted = 0;
    std::uint64_t lastUse = 0;
};

// Entries in use order (front = most recent) with a hash index by key: a lookup is O(1) and the
// eviction takes the back. APS5_NO_TEXTURE_HASH=1 finds entries by scanning the list (the index is
// still kept), the linear lookup of before.
struct TextureCache {
    std::mutex mutex;
    std::list<CachedTexture> entries;
    std::unordered_map<TextureKey, std::list<CachedTexture>::iterator, TextureKeyHash> index;
    std::uint64_t bytes = 0;
    std::uint64_t hostBytes = 0;
    std::uint64_t sweptDepartures = 0;
    std::uint64_t maintainedFrame = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t erasures = 0;
};

TextureCache& Textures() {
    static TextureCache cache;
    return cache;
}

bool TextureHashEnabled() {
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_HASH") != nullptr;
    return !disabled;
}

std::list<CachedTexture>::iterator findTexture(TextureCache& cache, const TextureKey& key) {
    if (TextureHashEnabled()) {
        const auto found = cache.index.find(key);
        return found == cache.index.end() ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key == key) return it;
    }
    return cache.entries.end();
}

// Sampled textures dropped by their cache so far (see Texture::NoteDeparted).
std::atomic<std::uint64_t>& SampledDepartures() {
    static std::atomic<std::uint64_t> departures{0};
    return departures;
}

void eraseTexture(TextureCache& cache, std::list<CachedTexture>::iterator it) {
    if (it->texture != nullptr) it->texture->NoteDeparted();
    SampledDepartures().fetch_add(1, std::memory_order_relaxed);
    cache.bytes -= it->accounted;
    cache.hostBytes -= it->bytes.size();
    cache.index.erase(it->key);
    cache.entries.erase(it);
    ++cache.erasures;
}

std::uint64_t dropDeadViews(TextureCache& cache) {
    std::uint64_t dropped = 0;
    for (auto it = cache.entries.begin(); it != cache.entries.end();) {
        if (it->source == nullptr || it->source->Cached()) {
            ++it;
            continue;
        }
        eraseTexture(cache, it++);
        ++dropped;
    }
    return dropped;
}

// Moves an entry to the front (most recently used).
void touchTexture(TextureCache& cache, std::list<CachedTexture>::iterator it) {
    it->lastUse = ResidencyClock::Now();
    cache.entries.splice(cache.entries.begin(), cache.entries, it);
}

bool DedupeImages() {
    static const bool enabled = std::getenv("APS5_NO_IMAGE_DEDUPE") == nullptr;
    return enabled;
}

bool ImageMemoEnabled() {
    static const bool enabled = std::getenv("APS5_NO_IMAGE_MEMO") == nullptr;
    return enabled;
}

struct ImageMemoEntry {
    VkDevice device = VK_NULL_HANDLE;
    std::array<std::uint32_t, 8> words{};
    GuestTextureResource resource{};
    std::uint64_t guestBytes = 0;
    VkComponentMapping components{};
    bool hasEntry = false;
    std::list<CachedTexture>::iterator entry{};
    std::uint64_t erasures = 0;
};

constexpr std::size_t ImageMemoSlots = 1024;
thread_local std::unique_ptr<std::array<ImageMemoEntry, ImageMemoSlots>> imageMemo;

ImageMemoEntry& imageMemoSlot(std::span<const std::uint32_t> words) {
    if (imageMemo == nullptr) imageMemo = std::make_unique<std::array<ImageMemoEntry, ImageMemoSlots>>();
    std::uint64_t hash = 14695981039346656037ull;
    for (const auto word : words) hash = (hash ^ word) * 1099511628211ull;
    return (*imageMemo)[(hash ^ (hash >> 29u)) % ImageMemoSlots];
}

const ImageMemoEntry* decodedImage(VkDevice device, std::span<const std::uint32_t> words) {
    auto& slot = imageMemoSlot(words);
    if (slot.device == device && std::equal(words.begin(), words.end(), slot.words.begin())) return &slot;
    slot.device = VK_NULL_HANDLE;
    slot.resource = DecodeTextureResource(words);
    slot.guestBytes = DescribeSurface(slot.resource).guestBytes;
    slot.components = {ComponentSwizzleFor(slot.resource.dstSelX), ComponentSwizzleFor(slot.resource.dstSelY), ComponentSwizzleFor(slot.resource.dstSelZ), ComponentSwizzleFor(slot.resource.dstSelW)};
    std::copy(words.begin(), words.end(), slot.words.begin());
    slot.hasEntry = false;
    slot.device = device;
    return &slot;
}

std::list<CachedTexture>::iterator findSampledEntry(TextureCache& cache, VkDevice device, const std::array<std::uint32_t, 8>& words, VkComponentMapping components) {
    if (!ImageMemoEnabled()) return findTexture(cache, MakeTextureKey(device, words, components));
    auto& slot = imageMemoSlot(words);
    const bool same = slot.device == device && slot.words == words && slot.components.r == components.r && slot.components.g == components.g && slot.components.b == components.b && slot.components.a == components.a;
    if (same && slot.hasEntry && slot.erasures == cache.erasures) return slot.entry;
    const auto it = findTexture(cache, MakeTextureKey(device, words, components));
    if (same) {
        slot.hasEntry = it != cache.entries.end();
        slot.entry = it;
        slot.erasures = cache.erasures;
    }
    return it;
}

// APS5_PROFILE_DRAW: what the sampled-texture and storage-image lookups did, printed as [textures]
// every 10 s (cumulative). Storage-sourced sampled textures need no CPU read; snapshots do, and a
// pending read is a snapshot compare or read made while recorded work still wrote the surface (the
// flush hook may find that work already signaled, so this bounds the hook syncs from above; the
// [hooksync] line attributes the actual waits).
struct TextureCounters {
    std::atomic<std::uint64_t> fromStorage{0};
    std::atomic<std::uint64_t> snapshots{0};
    std::atomic<std::uint64_t> pendingReads{0};
    std::atomic<std::uint64_t> storageFallbacks{0};
    std::atomic<std::uint64_t> records{0};
    std::atomic<std::uint64_t> fastHits{0};
    std::atomic<std::uint64_t> fastMisses{0};
    std::atomic<std::uint64_t> storageHits{0};
    std::atomic<std::uint64_t> storageCreated{0};
    // Storage images a Revalidate refreshed directly instead of through the lookups (T1, see
    // ShaderResources::refreshOwnObjects).
    std::atomic<std::uint64_t> ownRefreshes{0};
    std::atomic<std::uint64_t> sampledEvicted{0};
    std::atomic<std::uint64_t> sampledEvictedBytes{0};
    std::atomic<std::uint64_t> sampledRescued{0};
    std::atomic<std::uint64_t> deadViews{0};
    std::atomic<std::uint64_t> storageEvicted{0};
    std::atomic<std::uint64_t> storageEvictedBytes{0};
    std::atomic<std::uint64_t> sampledBytes{0};
    std::atomic<std::uint64_t> sampledHostBytes{0};
    std::atomic<std::uint64_t> storageBytes{0};
    std::atomic<std::uint64_t> sampledSoft{0};
    std::atomic<std::uint64_t> sampledHard{0};
    std::atomic<std::uint64_t> storageSoft{0};
    std::atomic<std::uint64_t> storageHard{0};
    std::atomic<std::uint64_t> exhaustedRetries{0};
    std::atomic<std::int64_t> lastReport{0};
};

TextureCounters& TextureCounts() {
    static TextureCounters counters;
    return counters;
}

struct ResidencyConfig {
    ResidencyPolicy policy;
    std::uint64_t minIdleTicks = 16;
    std::uint32_t minIdleFrames = 2;
    std::optional<std::uint64_t> sampledOverride;
    std::optional<std::uint64_t> storageOverride;
    std::uint64_t hostSoft = 0;
};

const ResidencyConfig& Residency() {
    static const ResidencyConfig config = [] {
        ResidencyConfig made;
        const char* mode = std::getenv("APS5_TEXTURE_EVICTION");
        made.policy.strictLru = mode != nullptr && std::strcmp(mode, "lru") == 0;
        if (const char* text = std::getenv("APS5_TEXTURE_IDLE_SUBMISSIONS")) made.minIdleTicks = std::max<std::uint64_t>(1, std::strtoull(text, nullptr, 10));
        if (const char* text = std::getenv("APS5_TEXTURE_IDLE_FRAMES")) made.minIdleFrames = static_cast<std::uint32_t>(std::max<std::uint64_t>(1, std::strtoull(text, nullptr, 10)));
        made.sampledOverride = MebibytesFromEnvironment("APS5_TEXTURE_CACHE_MIB");
        made.storageOverride = MebibytesFromEnvironment("APS5_STORAGE_CACHE_MIB");
        made.hostSoft = HostSoftLimit(PhysicalMemoryBytes(), MebibytesFromEnvironment("APS5_TEXTURE_HOST_MIB"));
        if (made.policy.strictLru) {
            constexpr std::uint64_t oldBudget = 2048ull << 20u;
            if (!made.sampledOverride.has_value()) made.sampledOverride = oldBudget;
            if (!made.storageOverride.has_value()) made.storageOverride = oldBudget;
        }
        return made;
    }();
    return config;
}

DeviceMemoryBudget SampleDeviceBudget(const Context& context, std::uint64_t& heapBytes) {
    struct Sample {
        std::mutex mutex;
        std::uint64_t frame = std::numeric_limits<std::uint64_t>::max();
        std::chrono::steady_clock::time_point at{};
        DeviceMemoryBudget budget;
        std::uint64_t heapBytes = 0;
    };
    static Sample sample;
    std::lock_guard lock(sample.mutex);
    const auto now = std::chrono::steady_clock::now();
    const auto frame = ResidencyClock::Frame();
    if (frame != sample.frame || now - sample.at > std::chrono::seconds(1)) {
        sample.budget = QueryDeviceMemoryBudget(context);
        sample.heapBytes = DeviceLocalHeapBytes(context);
        sample.frame = frame;
        sample.at = now;
    }
    heapBytes = sample.heapBytes;
    return sample.budget;
}

ResidencyLimits CacheLimits(const Context& context, std::uint64_t cacheDeviceBytes, std::uint64_t numerator, std::uint64_t denominator, std::optional<std::uint64_t> deviceOverride, bool holdsHostCopies) {
    constexpr auto unlimited = std::numeric_limits<std::uint64_t>::max();
    const auto& config = Residency();
    ResidencyLimits limits{unlimited, unlimited, unlimited, unlimited};
    if (config.policy.strictLru) {
        limits.deviceSoft = *deviceOverride;
        return limits;
    }
    std::uint64_t heapBytes = 0;
    const auto budget = SampleDeviceBudget(context, heapBytes);
    limits.deviceHard = DeviceHardLimit(budget, heapBytes, cacheDeviceBytes);
    limits.deviceSoft = DeviceSoftLimit(budget, heapBytes, numerator, denominator, deviceOverride);
    if (holdsHostCopies) {
        limits.hostSoft = config.hostSoft;
        limits.hostHard = config.hostSoft * 2;
    }
    return limits;
}

ResidencyPolicy ExhaustedPolicy() {
    ResidencyPolicy policy;
    policy.criticalEvictionsPerPass = std::numeric_limits<std::size_t>::max();
    policy.scansPerPass = std::numeric_limits<std::size_t>::max();
    return policy;
}

bool ChargeTextureViews() {
    static const bool charge = std::getenv("APS5_CHARGE_TEXTURE_VIEWS") != nullptr;
    return charge;
}

std::atomic<std::uint64_t>& StorageDepartures() {
    static std::atomic<std::uint64_t> departures{0};
    return departures;
}

bool TextureCountersReported() {
    static const bool reported = std::getenv("APS5_PROFILE_DRAW") != nullptr || std::getenv("APS5_TEXTURE_STATS") != nullptr;
    return reported;
}

void reportTextureCounters() {
    if (!TextureCountersReported()) return;
    auto& counters = TextureCounts();
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load();
    if (nowMs - last < 10000 || !counters.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto count = [](const std::atomic<std::uint64_t>& value) { return static_cast<unsigned long long>(value.load(std::memory_order_relaxed)); };
    AgcDriver::ReportLine("[textures] sampled created: %llu from storage images, %llu snapshots (%llu snapshot reads over recorded writes inside cachedTexture, %llu storage-path fallbacks); stage-A records %llu: %llu fast hits, %llu full lookups; storage images %llu hits, %llu created, %llu own-object refreshes; residency: sampled %llu MiB held (%llu MiB host copies, soft %llu, hard %llu), %llu evicted (%llu MiB), %llu rescued, %llu dead views dropped; storage %llu MiB held (soft %llu, hard %llu), %llu evicted (%llu MiB); %llu retries after device memory ran out\n", count(counters.fromStorage), count(counters.snapshots), count(counters.pendingReads), count(counters.storageFallbacks), count(counters.records), count(counters.fastHits), count(counters.fastMisses), count(counters.storageHits), count(counters.storageCreated), count(counters.ownRefreshes), count(counters.sampledBytes) >> 20u, count(counters.sampledHostBytes) >> 20u, count(counters.sampledSoft) >> 20u, count(counters.sampledHard) >> 20u, count(counters.sampledEvicted), count(counters.sampledEvictedBytes) >> 20u, count(counters.sampledRescued), count(counters.deadViews), count(counters.storageBytes) >> 20u, count(counters.storageSoft) >> 20u, count(counters.storageHard) >> 20u, count(counters.storageEvicted), count(counters.storageEvictedBytes) >> 20u, count(counters.exhaustedRetries));
}

// What the sampled-texture lookups on this thread proved their returned objects current against,
// taken by object into the ShaderResources being built or revalidated (captureValidation), which
// then repeats the proof from write stamps on its next Revalidate. The lookups run inside one build
// on one thread; records nobody takes (a build that threw, a resident target looked up outside a
// build, a thread that never captures) are dropped by the next capture or by the bound below.
struct LookupRecord {
    const void* object;
    GuestTextureResource resource;
    std::uint64_t bytes;
    DccKeys keys;
    std::uint64_t generation;
    const StorageTexture* source;
};

thread_local std::vector<LookupRecord> lookupLog;

void logLookup(const LookupRecord& record) {
    // The bound drops the oldest half, never everything: one build's lookups (a few dozen at most)
    // are the newest records, and a capture must find them even when the bound trips mid-build.
    constexpr std::size_t bound = 1024;
    if (lookupLog.size() >= bound) lookupLog.erase(lookupLog.begin(), lookupLog.begin() + bound / 2);
    lookupLog.push_back(record);
}

std::shared_ptr<StorageTexture> cachedStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, std::uint32_t mip, std::uint64_t guestBytes = 0);
bool MetadataMoved(const StorageTexture& image, const GuestTextureResource& resource);

// Whether a sampled texture over `resource` can be a view of the surface's cached storage image
// instead of a CPU snapshot (see cachedTexture): the format has a storage form and is not block
// compressed, and the surface lives in host-imported memory, where the image uploads and refreshes
// GPU-direct (a snapshot of such a surface reads and compares its bytes on the CPU, waiting for
// the recorded work that wrote them). APS5_NO_SAMPLED_FROM_STORAGE=1 keeps every sampled texture a
// snapshot as before.
bool SampledFromStorageEligible(const Context& context, std::uint32_t format, std::uint64_t address, std::uint64_t guestBytes) {
    static const bool disabled = std::getenv("APS5_NO_SAMPLED_FROM_STORAGE") != nullptr;
    if (disabled || IsBlockCompressed(format) || !StorageFormatAvailable(context, format)) return false;
    return HostImportCovers(context, address, static_cast<std::size_t>(guestBytes));
}

bool SampledFromStorageEligible(const Context& context, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    return SampledFromStorageEligible(context, resource.format, resource.baseAddress, guestBytes);
}

// Whether a sampled texture over a fast-cleared, storage-eligible surface views the cleared storage
// image (see cachedTexture). APS5_NO_CLEARED_VIEW=1 keeps such surfaces snapshots, as before.
bool ClearedViewEnabled() {
    static const bool disabled = std::getenv("APS5_NO_CLEARED_VIEW") != nullptr;
    return !disabled;
}

// Surfaces whose storage image could not be made (no writable committed pages, say): remembered
// so the sampled lookup does not throw and fall back on every use. A failure is keyed by the surface
// (address and size: the heap reuses addresses) and holds only while the guest mappings are what
// they were when it failed (the allocation generation): pages committed later, or another surface
// at the address, get a fresh attempt.
struct StorageFailures {
    std::mutex mutex;
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::uint64_t> generations;
};

StorageFailures& StorageFailed() {
    static StorageFailures failures;
    return failures;
}

// The cached storage image for a sampled descriptor, or null when it cannot be made (then the
// snapshot path serves the descriptor, as before). Every null return counts as a fallback.
std::shared_ptr<StorageTexture> sampledStorageSource(const Context& context, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    auto& counters = TextureCounts();
    auto& failures = StorageFailed();
    const auto surface = std::make_pair(resource.baseAddress, guestBytes);
    const auto generation = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    {
        std::lock_guard lock(failures.mutex);
        if (const auto it = failures.generations.find(surface); it != failures.generations.end()) {
            if (it->second == generation) {
                counters.storageFallbacks.fetch_add(1, std::memory_order_relaxed);
                return nullptr;
            }
            failures.generations.erase(it);
        }
    }
    try {
        return cachedStorageTexture(context, {}, resource, 0, guestBytes);
    } catch (const std::exception& error) {
        counters.storageFallbacks.fetch_add(1, std::memory_order_relaxed);
        std::lock_guard lock(failures.mutex);
        if (failures.generations.size() < 4096 && failures.generations.emplace(surface, generation).second) AgcDriver::ReportLine("[textures] sampled texture 0x%llx (%ux%u format %u) keeps the snapshot path: %s\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, error.what());
        return nullptr;
    }
}

// `guestBytes` is the surface size when the caller described the surface already (0: described here).
std::shared_ptr<Texture> cachedTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components, std::uint64_t guestBytes = 0, bool depthCompare = false);

EvictionOutcome evictSampled(TextureCache& cache, ResidencyUsage& usage, const ResidencyLimits& limits, const ResidencyPolicy& policy) {
    const auto& config = Residency();
    const auto window = ResidencyClock::Window(config.minIdleTicks, config.minIdleFrames);
    const auto outcome = RunEvictionPass(
            cache.entries, usage, limits, window, policy,
            [](const CachedTexture& entry) { return ResidencyEntryState{entry.lastUse, std::max(entry.lastUse, entry.texture->ResidencyUse()), entry.accounted, entry.bytes.size()}; },
            [&](std::list<CachedTexture>::iterator it) {
                it->lastUse = std::max(it->lastUse, it->texture->ResidencyUse());
                cache.entries.splice(cache.entries.begin(), cache.entries, it);
            },
            [&](std::list<CachedTexture>::iterator it) { eraseTexture(cache, it); });
    auto& counters = TextureCounts();
    counters.sampledEvicted.fetch_add(outcome.evicted, std::memory_order_relaxed);
    counters.sampledEvictedBytes.fetch_add(outcome.deviceBytes, std::memory_order_relaxed);
    counters.sampledRescued.fetch_add(outcome.rescued, std::memory_order_relaxed);
    return outcome;
}

template<typename Make>
auto makeWithSampledMemory(TextureCache& cache, Make make) {
    try {
        return make();
    } catch (const DeviceMemoryExhausted&) {
        ResidencyUsage usage{cache.bytes, cache.hostBytes};
        evictSampled(cache, usage, ResidencyLimits{}, ExhaustedPolicy());
        TextureCounts().exhaustedRetries.fetch_add(1, std::memory_order_relaxed);
        return make();
    }
}

std::shared_ptr<Texture> cachedTextureLookup(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components, std::uint64_t guestBytes, bool depthCompare, VkImageAspectFlags depthAspect) {
    CaptureTrace::Log("sampled-lookup address=%llx width=%u height=%u dcc=%llx", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, static_cast<unsigned long long>(resource.dccAddress));
    const auto depthBitsWidth = words.size() >= 4 ? ShaderRecompiler::DepthBitsTextureWidth(words[1], words[3]) : 0u;
    if (depthBitsWidth == 32u) {
        char text[160];
        std::snprintf(text, sizeof(text), "AGC graphics: 32-bit integer read of the depth-layout texture 0x%llx is not implemented", static_cast<unsigned long long>(resource.baseAddress));
        throw std::runtime_error(text);
    }
    constexpr auto unorm16 = static_cast<std::uint32_t>(ShaderRecompiler::IrBufferFormat::Format16UNorm);
    if (depthBitsWidth == 16u && resource.format != unorm16) {
        auto normalized = resource;
        normalized.format = unorm16;
        return cachedTexture(context, words, normalized, components, guestBytes, depthCompare);
    }
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    const bool profile = LookupOutcomes::Profiled();
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto scanKeys = [&] {
        const auto scanStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        const auto keys = TextureClearKeys(resource, guestBytes);
        if (profile && resource.dccAddress != 0) LookupOutcomes::Add(LookupOutcomes::DccScan, scanStart);
        return keys;
    };
    if (guestBytes == 0) guestBytes = DescribeSurface(resource).guestBytes;
    // A comparison sample reads an R32 float or R16 unorm surface through a native depth image
    // (Vulkan compares only depth formats). Other formats and 3D surfaces keep the binding they had
    // before: their color image under the comparison sampler, whose result Vulkan leaves to the
    // driver, rather than a skipped draw. Said once per format.
    if (depthCompare) {
        const auto format = ResolveTextureFormat(resource.format);
        if ((format != VK_FORMAT_R32_SFLOAT && format != VK_FORMAT_R16_UNORM) || resource.dimension == TextureDimension::k3D) {
            static std::mutex reportedMutex;
            static std::set<std::uint32_t> reported;
            std::lock_guard lock(reportedMutex);
            if (reported.insert(resource.format).second) AgcDriver::ReportLine("[gpu] comparison sampling of guest format %u (VkFormat %d%s) goes through its color image\n", resource.format, static_cast<int>(format), resource.dimension == TextureDimension::k3D ? ", 3D" : "");
            depthCompare = false;
        }
    }
    auto& counters = TextureCounts();
    const auto address = resource.baseAddress;
    const auto bytes = static_cast<std::size_t>(guestBytes);
    // A storage image whose results for this surface (or for a mip chain containing it) are still on
    // the GPU supplies the texture by a view of it; anything else needs those results in guest
    // memory first.
    auto source = StorageTexture::FindPending(address, guestBytes);
    if (source != nullptr && (depthAspect != 0 || depthCompare || !Texture::CanCopyFrom(*source, resource) || MetadataMoved(*source, resource))) source.reset();
    std::optional<DccKeys> keys;
    bool clearThroughKeys = false;
    if (source != nullptr && resource.dccAddress != 0 && IsDccClear(source->FilledKeys())) {
        keys = scanKeys();
        if (*keys == source->FilledKeys()) {
            std::array<std::byte, 16> probe{};
            if (!FillDccClear(ResolveTextureFormat(resource.format), *keys, resource.dccAlphaOnMsb, probe)) {
                char text[256];
                std::snprintf(text, sizeof(text), "AGC graphics: sampled texture 0x%llx (%ux%u format %u, dcc 0x%llx) reads %s DCC keys filled over its pending image, a clear value the format has no encoding for", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<unsigned long long>(resource.dccAddress), DccKeysName(*keys));
                throw std::runtime_error(text);
            }
            static const bool traceKeys = std::getenv("APS5_TRACE_DCC_KEYS") != nullptr;
            if (traceKeys) AgcDriver::ReportLine("[dcc-keys] sampled 0x%llx through keys 0x%llx reads the %s fill, not the pending image\n", static_cast<unsigned long long>(address), static_cast<unsigned long long>(resource.dccAddress), DccKeysName(*keys));
            source.reset();
            clearThroughKeys = true;
        }
    }
    // Otherwise a surface in host-imported memory is viewed through its cached storage image (made
    // here when there is none): its refresh after a CPU or GPU write is a GPU-direct detile from the
    // import, recorded behind the producer, so no bytes are read or compared on the CPU and nothing
    // waits for the producer. A fast-cleared surface (keys) is viewed only through an image whose own
    // descriptor carries the DCC address (below); it stays a snapshot otherwise, its texels not read.
    if (depthAspect == 0 && !depthCompare && source == nullptr && !clearThroughKeys && SampledFromStorageEligible(context, resource, guestBytes)) {
        keys = scanKeys();
        if (*keys == DccKeys::Uncompressed) {
            source = sampledStorageSource(context, resource, guestBytes);
        } else if (ClearedViewEnabled() && StorageClearAvailable(context, resource.format, *keys)) {
            // A fast-cleared surface is viewed as well, through its image cleared on the GPU (the
            // lookup's Refresh, see StorageTexture::upload), when the image's own descriptor names
            // the same DCC metadata so the refresh sees the keys. A snapshot of the clear texels
            // (filled and uploaded on the CPU, then replaced by a view once results are pending: two
            // 4K uploads per clear at the movie stage) serves the surface otherwise, as before.
            auto candidate = sampledStorageSource(context, resource, guestBytes);
            if (candidate != nullptr && candidate->Descriptor().dccAddress == resource.dccAddress) source = std::move(candidate);
        }
        // The lookup's Refresh may have flushed another image's results over the memory (which
        // marks the keys uncompressed): a surface that stays a snapshot re-reads them behind that
        // flush, as it does behind its own. APS5_NO_KEYS_RESCAN=1 keeps the keys scanned above.
        static const bool rescan = std::getenv("APS5_NO_KEYS_RESCAN") == nullptr;
        if (rescan && source == nullptr) keys.reset();
    }
    // Pages of the surface written since they were last collected are stamped now, before any
    // UnchangedSince (here and on a cache hit) looks at them: the checks only compare stamped blocks,
    // so a write the game made since the last walk is invisible until a collect stamps its page. The
    // walk is memoized per packet, so a texture viewed from a storage image repeats it for free.
    auto generation = GuestMemory::CollectWrites(address, bytes);
    if (source != nullptr && !GuestMemory::UnchangedSince(address, bytes, source->Generation())) {
        // The CPU wrote the memory while results were pending: Refresh merges them the usual way.
        source->Refresh();
    }
    // Results stored to guest memory just now are stamped newer than `generation`; a snapshot read
    // after them is current at the generation of a second (memoized) collect. The store marks the
    // surface's DCC keys uncompressed, so the keys are read after it.
    const auto flushStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (source == nullptr && !clearThroughKeys && StorageTexture::FlushPending(address, bytes, nullptr, "sampled texture")) {
        if (profile) LookupOutcomes::Add(LookupOutcomes::PendingFlush, flushStart);
        generation = GuestMemory::CollectWrites(address, bytes);
        keys.reset();
    }
    if (!keys.has_value()) keys = scanKeys();
    if (disabled) {
        if (source != nullptr) return std::make_shared<Texture>(context, source, resource, components);
        std::vector<std::byte> snapshot(bytes);
        ReadTextureSurface(resource, *keys, snapshot);
        return std::make_shared<Texture>(context, *context.detiler, resource, components, snapshot, depthCompare);
    }
    auto& cache = Textures();
    const auto key = MakeTextureKey(context.device, words, components, depthCompare);
    std::lock_guard lock(cache.mutex);
    if (const auto frame = ResidencyClock::Frame(); frame != cache.maintainedFrame && !Residency().policy.strictLru) {
        cache.maintainedFrame = frame;
        ResidencyUsage usage{cache.bytes, cache.hostBytes};
        evictSampled(cache, usage, CacheLimits(context, cache.bytes, 3, 8, Residency().sampledOverride, true), Residency().policy);
    }
    if (auto it = findTexture(cache, key); it != cache.entries.end()) {
        if (it->source != nullptr) {
            // The view follows the storage image, whatever the GPU wrote to it since; guest memory
            // written meanwhile is taken in by refreshing the image. A fast clear the image cannot
            // see (keys, and no pending results to prefer) ends the view: a snapshot holds the clear.
            if (depthAspect == 0 && (source == nullptr || source == it->source) && (source != nullptr || *keys == DccKeys::Uncompressed) && StorageImageServesKeys(*it->source, resource.dccAddress)) {
                if (!GuestMemory::UnchangedSince(address, bytes, it->source->Generation())) it->source->Refresh();
                it->keys = *keys;
                touchTexture(cache, it);
                logLookup({it->texture.get(), resource, guestBytes, *keys, 0, it->source.get()});
                reportTextureCounters();
                if (profile) LookupOutcomes::Add(*keys != DccKeys::Uncompressed ? LookupOutcomes::SampledHitClearedView : LookupOutcomes::SampledHitView, start);
                return it->texture;
            }
        } else if (source == nullptr && it->bytes.size() == guestBytes && it->keys == *keys) {
            // Unwritten pages need no comparison; partially resident textures compare only committed
            // pages. The compare goes through the flush hook: it waits for recorded work over the
            // surface (counted, and named for the [hooksync] line).
            const auto equalsCommitted = [&] {
                if (Recorder::SnapshotWriteOverlaps(address, bytes)) counters.pendingReads.fetch_add(1, std::memory_order_relaxed);
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::TextureCompare);
                return GuestMemory::EqualsCommitted(address, it->bytes);
            };
            const bool residentDepth = depthAspect != 0 && it->texture->SamplesResidentDepth();
            if (*keys != DccKeys::Uncompressed || residentDepth || GuestMemory::UnchangedSince(address, it->bytes.size(), it->generation) || equalsCommitted()) {
                it->generation = generation;
                touchTexture(cache, it);
                logLookup({it->texture.get(), resource, guestBytes, *keys, generation, nullptr});
                reportTextureCounters();
                if (profile) LookupOutcomes::Add(LookupOutcomes::SampledHitSnapshot, start);
                return it->texture;
            }
        }
        eraseTexture(cache, it);
    }
    CachedTexture entry{key, address, std::vector<std::byte>(source != nullptr ? 0u : bytes), nullptr, *keys, generation};
    entry.accounted = source != nullptr && !ChargeTextureViews() ? 0u : guestBytes;
    if (source != nullptr) {
        entry.source = source;
        entry.sourceVersion = source->Version();
        entry.texture = makeWithSampledMemory(cache, [&] { return std::make_shared<Texture>(context, source, resource, components); });
        counters.fromStorage.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Snapshot before the upload so a write racing with it is caught by the next comparison.
        if (*keys == DccKeys::Uncompressed && Recorder::SnapshotWriteOverlaps(address, bytes)) counters.pendingReads.fetch_add(1, std::memory_order_relaxed);
        ReadTextureSurface(resource, *keys, entry.bytes);
        static const bool traceTextures = std::getenv("APS5_TRACE_TEXTURES") != nullptr;
        if (traceTextures) {
            std::size_t nonzero = 0;
            for (std::size_t i = 0; i < entry.bytes.size(); i += 64) nonzero += entry.bytes[i] != std::byte{0};
            AgcDriver::ReportLine("[texture] 0x%llx %ux%u format %u tile %d: %zu of %zu sampled bytes nonzero\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), nonzero, entry.bytes.size() / 64);
        }
        entry.texture = makeWithSampledMemory(cache, [&] { return std::make_shared<Texture>(context, *context.detiler, resource, components, entry.bytes, depthCompare); });
        counters.snapshots.fetch_add(1, std::memory_order_relaxed);
    }
    const auto& residency = Residency();
    if (const auto departures = StorageDepartures().load(std::memory_order_relaxed); !residency.policy.strictLru && departures != cache.sweptDepartures) {
        cache.sweptDepartures = departures;
        counters.deadViews.fetch_add(dropDeadViews(cache), std::memory_order_relaxed);
    }
    const auto limits = CacheLimits(context, cache.bytes, 3, 8, residency.sampledOverride, true);
    ResidencyUsage usage{cache.bytes + entry.accounted, cache.hostBytes + entry.bytes.size()};
    evictSampled(cache, usage, limits, residency.policy);
    cache.bytes += entry.accounted;
    cache.hostBytes += entry.bytes.size();
    entry.lastUse = ResidencyClock::Now();
    counters.sampledBytes.store(cache.bytes, std::memory_order_relaxed);
    counters.sampledHostBytes.store(cache.hostBytes, std::memory_order_relaxed);
    counters.sampledSoft.store(limits.deviceSoft, std::memory_order_relaxed);
    counters.sampledHard.store(limits.deviceHard, std::memory_order_relaxed);
    auto texture = entry.texture;
    logLookup({texture.get(), resource, guestBytes, *keys, generation, source.get()});
    cache.entries.push_front(std::move(entry));
    cache.index[key] = cache.entries.begin();
    reportTextureCounters();
    if (profile) LookupOutcomes::Add(source != nullptr ? LookupOutcomes::SampledMadeView : LookupOutcomes::SampledMadeSnapshot, start);
    return texture;
}

std::shared_ptr<Texture> cachedTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& resource, VkComponentMapping components, std::uint64_t guestBytes, bool depthCompare) {
    const auto depthAspect = SampledDepthAspect(context, resource);
    auto texture = cachedTextureLookup(context, words, resource, components, guestBytes, depthCompare, depthAspect);
    if (depthAspect != 0) SampleDepthSurface(context, texture, resource, depthAspect);
    return texture;
}

// Storage images stay on the GPU between dispatches: while guest memory still holds what an image was
// last uploaded from or written back as, its next use skips the upload and detile.
struct StorageKey {
    VkDevice device;
    std::array<std::uint32_t, 8> words;
    bool operator==(const StorageKey&) const = default;
};

struct StorageKeyHash {
    std::size_t operator()(const StorageKey& key) const noexcept {
        return static_cast<std::size_t>(hashWords(14695981039346656037ull ^ reinterpret_cast<std::uintptr_t>(key.device), key.words.data(), key.words.size()));
    }
};

struct CachedStorageTexture {
    StorageKey key;
    std::uint32_t mip;
    std::shared_ptr<StorageTexture> texture;
    std::uint64_t lastUse = 0;
};

// As TextureCache: use order with a hash index by key, plus one by image for StorageImageCached.
struct StorageTextureCache {
    std::mutex mutex;
    std::list<CachedStorageTexture> entries;
    std::unordered_map<StorageKey, std::list<CachedStorageTexture>::iterator, StorageKeyHash> index;
    std::unordered_map<const StorageTexture*, std::list<CachedStorageTexture>::iterator> byImage;
    std::uint64_t bytes = 0;
    std::uint64_t maintainedFrame = std::numeric_limits<std::uint64_t>::max();
};

StorageTextureCache& StorageTextures() {
    static StorageTextureCache cache;
    return cache;
}

std::list<CachedStorageTexture>::iterator findStorage(StorageTextureCache& cache, const StorageKey& key) {
    if (TextureHashEnabled()) {
        const auto found = cache.index.find(key);
        return found == cache.index.end() ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key == key) return it;
    }
    return cache.entries.end();
}

std::list<CachedStorageTexture>::iterator findStorageByImage(StorageTextureCache& cache, VkDevice device, const StorageTexture* image) {
    if (TextureHashEnabled()) {
        const auto found = cache.byImage.find(image);
        return found == cache.byImage.end() || found->second->key.device != device ? cache.entries.end() : found->second;
    }
    for (auto it = cache.entries.begin(); it != cache.entries.end(); ++it) {
        if (it->key.device == device && it->texture.get() == image) return it;
    }
    return cache.entries.end();
}

// Evicts an entry: its pending results go to guest memory first (the image may die with the entry).
void evictStorage(StorageTextureCache& cache, std::list<CachedStorageTexture>::iterator it) {
    StorageDepartures().fetch_add(1, std::memory_order_relaxed);
    it->texture->SetCached(false);
    it->texture->Flush();
    cache.bytes -= it->texture->GuestBytes();
    cache.index.erase(it->key);
    cache.byImage.erase(it->texture.get());
    cache.entries.erase(it);
}

EvictionOutcome evictStorageImages(StorageTextureCache& cache, ResidencyUsage& usage, const ResidencyLimits& limits, const ResidencyPolicy& policy) {
    const auto& config = Residency();
    const auto window = ResidencyClock::Window(config.minIdleTicks, config.minIdleFrames);
    const auto outcome = RunEvictionPass(
            cache.entries, usage, limits, window, policy,
            [](const CachedStorageTexture& entry) { return ResidencyEntryState{entry.lastUse, entry.lastUse, entry.texture->GuestBytes(), 0}; },
            [](std::list<CachedStorageTexture>::iterator) {},
            [&](std::list<CachedStorageTexture>::iterator it) { evictStorage(cache, it); });
    auto& counters = TextureCounts();
    counters.storageEvicted.fetch_add(outcome.evicted, std::memory_order_relaxed);
    counters.storageEvictedBytes.fetch_add(outcome.deviceBytes, std::memory_order_relaxed);
    return outcome;
}

template<typename Make>
auto makeWithStorageMemory(StorageTextureCache& cache, Make make) {
    try {
        return make();
    } catch (const DeviceMemoryExhausted&) {
        ResidencyUsage usage{cache.bytes, 0};
        evictStorageImages(cache, usage, ResidencyLimits{}, ExhaustedPolicy());
        TextureCounts().exhaustedRetries.fetch_add(1, std::memory_order_relaxed);
        return make();
    }
}

// Whether a descriptor names DCC metadata other than the keys `image` follows. The image stands
// for the surface as read through its own descriptor's keys (its refresh scans them, a fast clear of
// them clears it, its write-back marks them uncompressed), so a surface whose metadata moved (the
// title reallocated it, or its memory was another surface's before) must not keep the old keys: the
// title's fast clears of the new ones would never reach the image, and its write-backs would store
// into metadata the surface no longer owns. A descriptor without metadata reads the texels as
// stored and keeps whatever image the surface has. APS5_NO_MOVED_DCC_CHECK=1 keeps the first
// descriptor's keys for good, as before.
bool MetadataMoved(const StorageTexture& image, const GuestTextureResource& resource) {
    static const bool disabled = std::getenv("APS5_NO_MOVED_DCC_CHECK") != nullptr;
    static const bool unshared = std::getenv("APS5_NO_SHARED_DCC_KEYS") != nullptr;
    return !disabled && resource.dccAddress != 0 && image.Descriptor().dccAddress != resource.dccAddress && (unshared || !image.ServesKeysAt(resource.dccAddress));
}

// Storage images are shared by every descriptor of one surface (address, extent, layers, format, tile
// mode): the image holds the whole mip chain, and render targets in the same memory attach to it.
// The image follows one DCC key range, the newest a descriptor named (see MetadataMoved).
std::array<std::uint32_t, 8> SurfaceKey(const Context& context, const GuestTextureResource& resource) {
    // Guest formats that store in the same Vulkan format share the image (views carry the difference).
    return {static_cast<std::uint32_t>(resource.baseAddress), static_cast<std::uint32_t>(resource.baseAddress >> 32u), resource.width, resource.height, (resource.depthOrLastArray << 16u) | (resource.mipCount & 0xffffu), (static_cast<std::uint32_t>(resource.tileMode) << 12u) | (static_cast<std::uint32_t>(resource.dimension) << 20u), resource.baseArray, static_cast<std::uint32_t>(StorageFormatForGuest(context, resource.format))};
}

struct ExtendedSurfaces {
    std::mutex mutex;
    std::map<std::array<std::uint32_t, 8>, std::uint32_t> levels;
    std::atomic<bool> any{false};
};

ExtendedSurfaces& ExtendedChains() {
    static ExtendedSurfaces surfaces;
    return surfaces;
}

std::uint32_t AllocatedLevels(const GuestTextureResource& resource) {
    return resource.allocatedMipCount != 0 ? resource.allocatedMipCount : resource.mipCount;
}

GuestTextureResource StorageSurface(const Context& context, const GuestTextureResource& viewed) {
    auto& surfaces = ExtendedChains();
    const bool extended = viewed.mipCount > AllocatedLevels(viewed);
    if (!extended && !surfaces.any.load(std::memory_order_acquire)) return viewed;
    auto surface = viewed;
    surface.mipCount = AllocatedLevels(viewed);
    const auto identity = SurfaceKey(context, surface);
    std::lock_guard lock(surfaces.mutex);
    if (extended) {
        auto& levels = surfaces.levels[identity];
        levels = std::max(levels, viewed.mipCount);
        surfaces.any.store(true, std::memory_order_release);
        surface.mipCount = levels;
    } else if (const auto it = surfaces.levels.find(identity); it != surfaces.levels.end()) {
        surface.mipCount = it->second;
    } else {
        surface.mipCount = viewed.mipCount;
    }
    return surface;
}

// `guestBytes` is the surface size when the caller described the surface already (0: described here).
std::shared_ptr<StorageTexture> cachedStorageTexture(const Context& context, std::span<const std::uint32_t> words, const GuestTextureResource& viewed, std::uint32_t mip, std::uint64_t guestBytes) {
    static const bool disabled = std::getenv("APS5_NO_TEXTURE_CACHE") != nullptr;
    if (disabled) return std::make_shared<StorageTexture>(context, *context.detiler, viewed, mip);
    static_cast<void>(words);
    const bool profile = LookupOutcomes::Profiled();
    auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto& counters = TextureCounts();
    const auto resource = StorageSurface(context, viewed);
    const StorageKey key{context.device, SurfaceKey(context, resource)};
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    if (const auto frame = ResidencyClock::Frame(); frame != cache.maintainedFrame && !Residency().policy.strictLru) {
        cache.maintainedFrame = frame;
        ResidencyUsage usage{cache.bytes, 0};
        evictStorageImages(cache, usage, CacheLimits(context, cache.bytes, 1, 4, Residency().storageOverride, false), Residency().policy);
    }
    if (resource.mipCount > AllocatedLevels(resource)) {
        auto allocated = resource;
        allocated.mipCount = AllocatedLevels(resource);
        if (const auto it = findStorage(cache, {context.device, SurfaceKey(context, allocated)}); it != cache.entries.end()) evictStorage(cache, it);
    }
    if (auto it = findStorage(cache, key); it != cache.entries.end() && MetadataMoved(*it->texture, resource)) {
        // The old image leaves with its pending results stored (as the hardware's rendering left the
        // memory), and a new one is made below from memory under the keys the descriptor names; views
        // and recipes holding the old one see it gone from the cache (StorageImageCached).
        static std::atomic<int> reports{0};
        if (reports.fetch_add(1, std::memory_order_relaxed) < 8) AgcDriver::ReportLine("[gpu] storage image 0x%llx (%ux%u format %u): DCC keys moved from 0x%llx to 0x%llx; the image is remade under the new keys\n", static_cast<unsigned long long>(resource.baseAddress), resource.width, resource.height, resource.format, static_cast<unsigned long long>(it->texture->Descriptor().dccAddress), static_cast<unsigned long long>(resource.dccAddress));
        evictStorage(cache, it);
    } else if (it != cache.entries.end()) {
        it->texture->Refresh();
        it->lastUse = ResidencyClock::Now();
        cache.entries.splice(cache.entries.begin(), cache.entries, it);
        if (TextureCountersReported()) counters.storageHits.fetch_add(1, std::memory_order_relaxed);
        if (profile) LookupOutcomes::Add(LookupOutcomes::StorageHit, start);
        return it->texture;
    }
    // Results other images hold over this memory reach it before the new image reads it: a hit's
    // Refresh flushes them, the constructor's upload does not, and a GPU-direct upload reads the
    // import buffer without the flush hook. The flush is recorded ahead of the upload in the batch.
    if (guestBytes == 0) guestBytes = DescribeSurface(resource).guestBytes;
    if (StorageTexture::FlushPending(resource.baseAddress, static_cast<std::size_t>(guestBytes), nullptr, "storage image creation", PublishScope::None) && profile) start = LookupOutcomes::Add(LookupOutcomes::PendingFlush, start);
    CachedStorageTexture entry{key, mip, makeWithStorageMemory(cache, [&] { return std::make_shared<StorageTexture>(context, *context.detiler, resource, mip); })};
    // The constructor's upload may have recorded into the open batch (a GPU clear, a direct
    // detile) before the image could keep itself (no weak_from_this yet): the batch keeps it here,
    // so an eviction or a failed view before it ran cannot destroy a referenced image.
    // APS5_NO_KEEP_NEW_STORAGE=1 leaves the image to its cache entry alone, as before.
    static const bool keepNew = std::getenv("APS5_NO_KEEP_NEW_STORAGE") == nullptr;
    if (auto* recorder = Recorder::Active(); keepNew && recorder != nullptr && GuestMemory::GpuMutex().HeldByThisThread() && recorder->Recording()) recorder->Keep(entry.texture);
    const auto& residency = Residency();
    const auto limits = CacheLimits(context, cache.bytes, 1, 4, residency.storageOverride, false);
    ResidencyUsage usage{cache.bytes + entry.texture->GuestBytes(), 0};
    evictStorageImages(cache, usage, limits, residency.policy);
    cache.bytes += entry.texture->GuestBytes();
    entry.lastUse = ResidencyClock::Now();
    counters.storageBytes.store(cache.bytes, std::memory_order_relaxed);
    counters.storageSoft.store(limits.deviceSoft, std::memory_order_relaxed);
    counters.storageHard.store(limits.deviceHard, std::memory_order_relaxed);
    auto texture = entry.texture;
    cache.entries.push_front(std::move(entry));
    cache.index[key] = cache.entries.begin();
    cache.byImage[texture.get()] = cache.entries.begin();
    texture->SetCached(true);
    counters.storageCreated.fetch_add(1, std::memory_order_relaxed);
    if (profile) LookupOutcomes::Add(LookupOutcomes::StorageMade, start);
    return texture;
}

}

void FlushCachedTextures(VkDevice device) {
    Require(device != VK_NULL_HANDLE, "cannot flush textures without a Vulkan device");
    GuestMemory::AssertGpuLockHeld("FlushCachedTextures");
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    for (const auto& entry : cache.entries) {
        if (entry.key.device == device) entry.texture->Flush();
    }
}

void ClearCachedTextures(VkDevice device) {
    Require(device != VK_NULL_HANDLE, "cannot clear textures without a Vulkan device");
    auto& sampled = Textures();
    {
        std::lock_guard lock(sampled.mutex);
        for (auto it = sampled.entries.begin(); it != sampled.entries.end();) {
            if (it->key.device == device) eraseTexture(sampled, it++);
            else ++it;
        }
    }
    auto& storage = StorageTextures();
    std::lock_guard lock(storage.mutex);
    for (auto it = storage.entries.begin(); it != storage.entries.end();) {
        if (it->key.device != device) {
            ++it;
            continue;
        }
        it->texture->SetCached(false);
        storage.bytes -= it->texture->GuestBytes();
        storage.index.erase(it->key);
        storage.byImage.erase(it->texture.get());
        it = storage.entries.erase(it);
    }
}

bool StorageImageCached(const Context& context, const StorageTexture* image) {
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    const auto it = findStorageByImage(cache, context.device, image);
    if (it == cache.entries.end()) return false;
    it->lastUse = ResidencyClock::Now();
    cache.entries.splice(cache.entries.begin(), cache.entries, it);
    return true;
}

namespace {

// StorageImageCached for several images under one acquisition of the cache mutex.
bool StorageImagesCached(const Context& context, std::span<const StorageTexture* const> images) {
    if (images.empty()) return true;
    auto& cache = StorageTextures();
    std::lock_guard lock(cache.mutex);
    for (const auto* image : images) {
        const auto it = findStorageByImage(cache, context.device, image);
        if (it == cache.entries.end()) return false;
        it->lastUse = ResidencyClock::Now();
        cache.entries.splice(cache.entries.begin(), cache.entries, it);
    }
    return true;
}

}

std::shared_ptr<StorageTexture> CachedStorageSurface(const Context& context, const GuestTextureResource& resource) {
    return cachedStorageTexture(context, {}, resource, 0);
}

bool StorageImageServesKeys(const StorageTexture& image, std::uint64_t dccAddress) {
    GuestTextureResource resource{};
    resource.dccAddress = dccAddress;
    return !MetadataMoved(image, resource);
}

namespace {

const char* roleName(ShaderRecompiler::DescriptorRole role) {
    switch (role) {
        case ShaderRecompiler::DescriptorRole::GuestBuffers: return "GuestBuffers";
        case ShaderRecompiler::DescriptorRole::GuestImages: return "GuestImages";
        case ShaderRecompiler::DescriptorRole::GuestSamplers: return "GuestSamplers";
        case ShaderRecompiler::DescriptorRole::Gds: return "Gds";
        case ShaderRecompiler::DescriptorRole::BdaPagetable: return "BdaPagetable";
        case ShaderRecompiler::DescriptorRole::FaultBuffer: return "FaultBuffer";
        case ShaderRecompiler::DescriptorRole::FlattenedSrt: return "FlattenedSrt";
        case ShaderRecompiler::DescriptorRole::ShaderData: return "ShaderData";
    }
    throw std::runtime_error("AGC graphics: unknown descriptor role");
}

const char* kindName(ShaderRecompiler::DescriptorKind kind) {
    switch (kind) {
        case ShaderRecompiler::DescriptorKind::UniformBuffer: return "UniformBuffer";
        case ShaderRecompiler::DescriptorKind::StorageBuffer: return "StorageBuffer";
        case ShaderRecompiler::DescriptorKind::UniformTexelBuffer: return "UniformTexelBuffer";
        case ShaderRecompiler::DescriptorKind::StorageTexelBuffer: return "StorageTexelBuffer";
        case ShaderRecompiler::DescriptorKind::SampledImage: return "SampledImage";
        case ShaderRecompiler::DescriptorKind::StorageImage: return "StorageImage";
        case ShaderRecompiler::DescriptorKind::Sampler: return "Sampler";
    }
    throw std::runtime_error("AGC graphics: unknown descriptor kind");
}

// Consecutive identical storage descriptors address successive mips of one texture (dynamic-mip
// storage writes), so the second and later ones share the first one's image lookup.
// APS5_NO_STORAGE_DEDUPE=1 looks each one up (and refreshes the image) separately as before.
bool StorageDedupeEnabled() {
    static const bool disabled = std::getenv("APS5_NO_STORAGE_DEDUPE") != nullptr;
    return !disabled;
}

// Whether storage element `element` of `binding` repeats the previous element's eight words.
bool SameAsPreviousStorageElement(const ShaderRecompiler::DescriptorBinding& binding, std::uint32_t element) {
    if (element == 0) return false;
    const auto words = binding.guestDescriptor.begin() + static_cast<std::size_t>(element) * 8u;
    return std::equal(words, words + 8, words - 8);
}

}

ShaderResources::ShaderResources(const Context& context, const ShaderRecompiler::RecompileResult& vertex, const ShaderRecompiler::RecompileResult& fragment, const ColorTarget& target, std::uint64_t indexAddress, std::size_t indexBytes) : ShaderResources(context, std::array<CompiledShader, 2>{{{ShaderRecompiler::ShaderStage::Vertex, &vertex, 0}, {ShaderRecompiler::ShaderStage::Fragment, &fragment, static_cast<std::uint32_t>(vertex.pushConstants.size())}}}, target, indexAddress, indexBytes) {}

ShaderResources::ShaderResources(const Context& context, std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint64_t indexAddress, std::size_t indexBytes, std::span<const GuestMemorySnapshot> snapshots, bool stageWrites) : context(context), guestMemory(context) {
    if (stageWrites) guestMemory.AllowDrawStaging();
    // Only a draw's build is shared (see ReuseAddressDraws): a dispatch's address-based build
    // stages and refreshes nothing a template could keep, and its path never shares a lease.
    addressReuse = ReuseAddressDraws();
    prepareAddressBindings(shaders, snapshots);
    build(shaders, &target, indexAddress, indexBytes);
}

ShaderResources::ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots) : ShaderResources(context, compute, snapshots, false) {}

namespace {

// Whether a compute stage maps registered guest memory through BDA tables (see prepareAddressBindings).
bool UsesAddressTables(const CompiledShader& compute) {
    Require(compute.program != nullptr, "missing compiled shader");
    return std::any_of(compute.program->bindings.begin(), compute.program->bindings.end(), [](const auto& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable; });
}

}

ShaderResources::ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots, bool deferred, std::uint64_t dispatchThreads) : context(context), guestMemory(context), dispatchThreads(dispatchThreads), deferredCompute(compute), deferredSnapshots(snapshots) {
    Require(compute.stage == ShaderRecompiler::ShaderStage::Compute, "compute resources require a compute shader");
    // Every use of a compute build is a recorded dispatch that calls MarkGpuWrites, which staged
    // buffers need (a synchronous draw's use would not).
    guestMemory.AllowDeviceStaging();
    const std::span<const CompiledShader> shaders(&deferredCompute, 1);
    if (!deferred) {
        prepareAddressBindings(shaders, snapshots);
        build(shaders, nullptr, 0, 0);
        forgetDeferredInputs();
        return;
    }
    // Address-based shaders pin and mirror registered memory in prepareAddressBindings (reconciling
    // imports retires buffers to the recorder, refreshing mirrors waits for recorded work): that is
    // device-lock work, so their whole build waits for Complete(). Without tables the call only
    // checks the bindings.
    lockedBuild = UsesAddressTables(compute);
    if (lockedBuild) return;
    unlockedPrepare = true;
    prepareAddressBindings(shaders, snapshots);
    buildPrepare(shaders, nullptr, 0, 0);
}

void ShaderResources::Complete() {
    Require(!completed, "shader resources were already completed");
    const std::span<const CompiledShader> shaders(&deferredCompute, 1);
    if (lockedBuild) {
        prepareAddressBindings(shaders, deferredSnapshots);
        buildPrepare(shaders, nullptr, 0, 0);
    }
    buildComplete();
    forgetDeferredInputs();
}

void ShaderResources::forgetDeferredInputs() {
    // The compiled shader and the captured regions belong to the dispatch, which returns while this
    // object lives on in the resource cache and the recorder: nothing may reach for them after the
    // build, so they are dropped rather than left dangling.
    deferredCompute = {};
    deferredSnapshots = {};
}

void ShaderResources::build(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint64_t indexAddress, std::size_t indexBytes) {
    buildPrepare(shaders, target, indexAddress, indexBytes);
    buildComplete();
}

namespace {

// APS5_PROFILE_DRAW: the [resources] phase totals. Each thread accumulates its builds' phases in
// arrays of its own and merges them into the shared totals every 1000 of its builds (and when it
// ends), so no build takes the shared mutex per phase; the totals lag by up to 999 builds per
// worker.
constexpr std::size_t BuildPhaseCount = static_cast<std::size_t>(ShaderResources::BuildPhase::Count);
constexpr std::array<const char*, BuildPhaseCount> BuildPhaseNames{"bindings", "precollect", "guest memory upload", "descriptors", "stage A", "images", "bda", "stage B"};

struct BuildProfile {
    std::mutex mutex;
    std::array<double, BuildPhaseCount> ms{};
    std::uint64_t builds = 0;
};

BuildProfile& Builds() {
    static BuildProfile profile;
    return profile;
}

bool BuildProfiled() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    return profile;
}

struct ThreadBuildProfile {
    std::array<double, BuildPhaseCount> ms{};
    std::uint64_t builds = 0;
    ~ThreadBuildProfile() { merge(); }
    void merge() {
        auto& profile = Builds();
        std::lock_guard lock(profile.mutex);
        for (std::size_t i = 0; i < BuildPhaseCount; ++i) profile.ms[i] += ms[i];
        profile.builds += builds;
        ms = {};
        if (builds == 0) return;
        builds = 0;
        std::string report;
        for (std::size_t i = 0; i < BuildPhaseCount; ++i) report += " " + std::string(BuildPhaseNames[i]) + "=" + std::to_string(static_cast<long long>(profile.ms[i])) + "ms";
        AgcDriver::ReportLine("[resources] %llu builds, phase totals:%s\n", static_cast<unsigned long long>(profile.builds), report.c_str());
    }
};

ThreadBuildProfile& ThreadBuilds() {
    thread_local ThreadBuildProfile profile;
    return profile;
}

void addBuildPhase(ShaderResources::BuildPhase which, double ms) {
    ThreadBuilds().ms[static_cast<std::size_t>(which)] += ms;
}

// APS5_PROFILE_DRAW: guest buffer elements bound by descriptor (addGuestBuffer) and how many the
// recompiler proved read-only, plus the pending-write notes their uses skipped (MarkGpuWrites),
// printed as [buffers] every 10 s from MarkGpuWrites. Cumulative.
struct BufferWriteCounters {
    std::atomic<std::uint64_t> elements{0};
    std::atomic<std::uint64_t> readOnly{0};
    std::atomic<std::uint64_t> notesSkipped{0};
    std::atomic<std::int64_t> lastReport{0};
};

BufferWriteCounters& BufferWrites() {
    static BufferWriteCounters counters;
    return counters;
}

}

double ShaderResources::phase(BuildPhase which) {
    if (!BuildProfiled()) return 0.0;
    const auto now = std::chrono::steady_clock::now();
    const auto ms = std::chrono::duration<double, std::milli>(now - phaseStart).count();
    addBuildPhase(which, ms);
    if (ms > 50) AgcDriver::ReportLine("[resources] %s took %.0f ms (%zu textures, %zu storage images, %zu buffers, bda %d)\n", BuildPhaseNames[static_cast<std::size_t>(which)], ms, textures.size(), storageTextures.size(), allocations.size(), usesBda ? 1 : 0);
    phaseStart = now;
    return ms;
}

void ShaderResources::buildPrepare(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint64_t indexAddress, std::size_t indexBytes) {
    const GuestMemory::CollectSiteScope collectSite(GuestMemory::CollectSite::Build, false);
    const auto stageStart = std::chrono::steady_clock::now();
    phaseStart = stageStart;
    if (BuildProfiled()) {
        auto& profile = ThreadBuilds();
        if (++profile.builds % 1000 == 0) profile.merge();
    }
    try {
        Require(!shaders.empty() && context.limits.maxBoundDescriptorSets >= 1, "shader descriptor set exceeds device limits");
        std::size_t plannedBindings = 0;
        std::size_t plannedAllocations = 0;
        for (const auto& shader : shaders) {
            Require(shader.program != nullptr, "missing compiled shader");
            plannedBindings += shader.program->bindings.size();
            for (const auto& binding : shader.program->bindings) {
                if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages && binding.role != ShaderRecompiler::DescriptorRole::GuestSamplers) plannedAllocations += binding.count;
            }
        }
        bindings.reserve(plannedBindings);
        allocations.reserve(plannedAllocations);
        std::vector<std::uint32_t> occupied;
        occupied.reserve(plannedBindings);
        std::vector<std::size_t> offsetsInData;
        for (const auto& shader : shaders) {
            bdaWrites = bdaWrites || shader.program->bdaWrites;
            const VkShaderStageFlags flags = VulkanStage(shader.stage);
            std::uint64_t stageDescriptors = 0;
            offsetsInData.clear();
            std::int64_t shaderData = -1;
            for (const auto& binding : shader.program->bindings) {
                Require(binding.descriptorSet == 0, "unexpected descriptor set: every shader resource must use descriptor set zero");
                Require(std::find(occupied.begin(), occupied.end(), binding.binding) == occupied.end(), "duplicate shader binding");
                occupied.push_back(binding.binding);
                const bool addressRole = binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable || binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer;
                const bool gdsRole = binding.role == ShaderRecompiler::DescriptorRole::Gds;
                const bool bufferRole = addressRole || gdsRole || binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers || binding.role == ShaderRecompiler::DescriptorRole::ShaderData || binding.role == ShaderRecompiler::DescriptorRole::FlattenedSrt;
                const bool imageRole = binding.role == ShaderRecompiler::DescriptorRole::GuestImages || binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers;
                if (imageRole) {
                    addImageBinding(binding, flags);
                    continue;
                }
                if (!bufferRole) Require(false, std::string("unsupported descriptor role ") + roleName(binding.role));
                if (binding.kind != ShaderRecompiler::DescriptorKind::StorageBuffer) Require(false, std::string("unsupported descriptor kind ") + kindName(binding.kind) + " for role " + roleName(binding.role) + ": only StorageBuffer is supported");
                Require(!binding.readOnly, "read-only descriptors are unsupported because the recompiler emits no NonWritable decoration");
                Require(binding.count != 0, "empty descriptor binding");
                stageDescriptors += binding.count;
                storageBuffers += binding.count;
                Require(stageDescriptors <= context.limits.maxPerStageDescriptorStorageBuffers && stageDescriptors <= context.limits.maxPerStageResources, "shader descriptors exceed per-stage limits");
                Binding item{{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, binding.count, flags, nullptr}, {}};
                item.allocations.reserve(binding.count);
                if (binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers) {
                    Require(binding.guestDescriptor.size() == static_cast<std::uint64_t>(binding.count) * 4, "guest buffer descriptor must contain four DWORDs per array element");
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        // An element the recompiler did not classify (a producer without the
                        // vector) counts as written, like imageWritten below.
                        const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                        const bool atomic = element < binding.bufferAtomic.size() && binding.bufferAtomic[element];
                        const bool read = element < binding.bufferRead.size() && binding.bufferRead[element];
                        const auto index = addGuestBuffer(std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 4, 4), target, indexAddress, indexBytes, written, atomic, read);
                        const auto& push = shader.program->pushConstants;
                        if (!push.empty()) {
                            const auto position = shader.program->memoryOffsetDword * 4u + element;
                            Require(position < push.size(), "guest buffer offset lies outside the shader's push constants");
                            allocations[index].pushByte = static_cast<std::int32_t>(shader.pushConstantOffset + position);
                        } else {
                            allocations[index].dataByte = shader.program->memoryOffsetDword * 4u + element;
                            offsetsInData.push_back(index);
                        }
                        item.allocations.push_back(index);
                    }
                } else if (addressRole) {
                    item.allocations.push_back(allocations.size());
                    allocations.push_back({0, 0, false, nullptr, binding.role});
                } else if (gdsRole) {
                    // The device's GDS buffer, the backing the CP's DMA_DATA reaches too.
                    Require(binding.count == 1, "a GDS descriptor must not be an array");
                    Require(context.gdsBuffer != VK_NULL_HANDLE, "the device has no GDS buffer to bind (APS5_NO_GDS=1 is set, or another device holds the GDS)");
                    item.allocations.push_back(allocations.size());
                    allocations.push_back({0, Pm4::GdsBytes, false, nullptr, binding.role});
                    usesGds = true;
                } else {
                    Require(binding.count == 1, "shader data and flattened SRT descriptors must not be arrays");
                    Require(!binding.guestDescriptor.empty(), "empty shader data descriptor");
                    item.allocations.push_back(addDataBuffer(binding.guestDescriptor));
                    if (binding.role == ShaderRecompiler::DescriptorRole::ShaderData) shaderData = static_cast<std::int64_t>(item.allocations.back());
                }
                bindings.push_back(std::move(item));
            }
            for (const auto index : offsetsInData) allocations[index].dataAllocation = shaderData;
        }
        Require(storageBuffers <= context.limits.maxDescriptorSetStorageBuffers, "pipeline descriptors exceed device limits");
        textures.reserve(plannedSampledImages);
        textureFirstLayer.reserve(plannedSampledImages);
        storageTextures.reserve(plannedStorageImages);
        storageMips.reserve(plannedStorageImages);
        storageKeys.reserve(plannedStorageImages);
        storageFirstLayer.reserve(plannedStorageImages);
        storageWritten.reserve(plannedStorageImages);
        describedRanges.reserve(plannedSampledImages + plannedStorageImages);
        timing.bindingsMs = phase(BuildPhase::Bindings);
        // For every build, locked ones included: their stage B then takes the fast path too, and the
        // collects cost the same wherever they run.
        if (precollectImages()) phase(BuildPhase::Precollect);
        guestMemory.UploadPrepare(usesBda);
        timing.uploadMs = phase(BuildPhase::Upload);
        std::vector<VkDescriptorSetLayoutBinding> description;
        description.reserve(bindings.size());
        layoutKey.reserve(bindings.size() * 4);
        for (const auto& binding : bindings) {
            description.push_back(binding.layout);
            layoutKey.insert(layoutKey.end(), {binding.layout.binding, static_cast<std::uint32_t>(binding.layout.descriptorType), binding.layout.descriptorCount, binding.layout.stageFlags});
        }
        static const bool noLayoutCache = std::getenv("APS5_NO_LAYOUT_CACHE") != nullptr;
        static const bool noPoolCache = std::getenv("APS5_NO_POOL_CACHE") != nullptr;
        if (context.descriptorCache != nullptr && !noLayoutCache) {
            _layout = context.descriptorCache->Layout(layoutKey, description);
        } else {
            VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            info.bindingCount = static_cast<std::uint32_t>(description.size());
            info.pBindings = description.data();
            Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &info, nullptr, &_layout), "vkCreateDescriptorSetLayout");
            ownsLayout = true;
        }
        if (!bindings.empty()) {
            // The set is sized from the plan: every image element becomes exactly one descriptor
            // when stage B looks it up.
            std::vector<VkDescriptorPoolSize> sizes;
            if (storageBuffers != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, static_cast<std::uint32_t>(storageBuffers)});
            if (plannedSampledImages != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, plannedSampledImages});
            if (plannedStorageImages != 0) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, plannedStorageImages});
            if (!samplers.empty()) sizes.push_back({VK_DESCRIPTOR_TYPE_SAMPLER, static_cast<std::uint32_t>(samplers.size())});
            if (context.descriptorCache != nullptr && !noPoolCache) {
                const auto allocated = context.descriptorCache->Allocate(_layout, sizes);
                _set = allocated.set;
                cachePool = allocated.pool;
            }
            if (_set == VK_NULL_HANDLE) {
                VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
                poolInfo.maxSets = 1;
                poolInfo.poolSizeCount = static_cast<std::uint32_t>(sizes.size());
                poolInfo.pPoolSizes = sizes.data();
                Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
                VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
                allocation.descriptorPool = pool;
                allocation.descriptorSetCount = 1;
                allocation.pSetLayouts = &_layout;
                Check(context.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(context.device, &allocation, &_set), "vkAllocateDescriptorSets");
            }
        }
        timing.descriptorsMs = phase(BuildPhase::Descriptors);
        if (BuildProfiled()) {
            timing.prepareMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            addBuildPhase(BuildPhase::StageA, timing.prepareMs);
        }
    } catch (...) {
        release();
        throw;
    }
}

void ShaderResources::buildComplete() {
    const GuestMemory::CollectSiteScope collectSite(GuestMemory::CollectSite::Build, false);
    const auto stageStart = std::chrono::steady_clock::now();
    phaseStart = stageStart;
    try {
        // The lookups run in plan order: Revalidate walks the bindings the same way, and consecutive
        // storage elements of one mip chain share the previous element's image.
        previousSampled = {};
        for (const auto& deferred : deferredImages) resolveImageBinding(*deferred.binding, bindings[deferred.index]);
        previousSampled = {};
        deferredImages.clear();
        // The records served their purpose: each holds the cache entry's objects as of stage A,
        // which would otherwise keep a replaced texture or an evicted storage image (and its device
        // memory) alive, outside the caches' budgets, for as long as this object is cached.
        imageRecords.clear();
        imageRecords.shrink_to_fit();
        nextImageRecord = 0;
        Require(textures.size() == plannedSampledImages && storageTextures.size() == plannedStorageImages, "image lookups disagree with the descriptor plan");
        timing.bindingsMs += phase(BuildPhase::Images);
        guestMemory.UploadFinish(usesBda);
        timing.uploadMs += phase(BuildPhase::Upload);
        if (usesBda) bda = std::make_unique<BdaResources>(context, guestMemory, bdaWrites);
        else if (usesFaultBuffer) bda = std::make_unique<BdaResources>(context, bdaWrites);
        phase(BuildPhase::Bda);
        if (_set != VK_NULL_HANDLE) {
            // One update call for the whole set: the info arrays are sized up front so every write's
            // pointer into them stays valid until the call.
            std::size_t bufferCount = 0;
            std::size_t imageCount = 0;
            for (const auto& binding : bindings) {
                bufferCount += binding.allocations.size();
                imageCount += binding.imageAllocations.size();
            }
            std::vector<VkDescriptorBufferInfo> buffers;
            std::vector<VkDescriptorImageInfo> images;
            buffers.reserve(bufferCount);
            images.reserve(imageCount);
            std::vector<VkWriteDescriptorSet> writes;
            writes.reserve(bindings.size());
            for (const auto& binding : bindings) {
                VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                write.dstSet = _set;
                write.dstBinding = binding.layout.binding;
                write.descriptorCount = binding.layout.descriptorCount;
                write.descriptorType = binding.layout.descriptorType;
                switch (binding.layout.descriptorType) {
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
                        write.pBufferInfo = buffers.data() + buffers.size();
                        for (const auto index : binding.allocations) buffers.push_back(descriptor(allocations[index]));
                        break;
                    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
                        write.pImageInfo = images.data() + images.size();
                        for (const auto index : binding.imageAllocations) images.push_back({VK_NULL_HANDLE, textureFirstLayer[index] ? textures[index]->FirstLayerView() : textures[index]->View(), textures[index]->Layout()});
                        break;
                    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                        write.pImageInfo = images.data() + images.size();
                        for (const auto index : binding.imageAllocations) images.push_back({VK_NULL_HANDLE, storageFirstLayer[index] ? storageTextures[index]->FirstLayerView(storageMips[index]) : storageTextures[index]->View(storageMips[index]), VK_IMAGE_LAYOUT_GENERAL});
                        break;
                    case VK_DESCRIPTOR_TYPE_SAMPLER:
                        write.pImageInfo = images.data() + images.size();
                        for (const auto index : binding.imageAllocations) images.push_back({samplers[index]->Handle(), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED});
                        break;
                    default: throw std::runtime_error("AGC graphics: ShaderResources encountered an unknown descriptor type while writing the descriptor set");
                }
                writes.push_back(write);
            }
            context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets")(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
        for (const auto& allocation : allocations) {
            if (allocation.adjustment == 0) continue;
            if (allocation.pushByte >= 0) {
                pushPatches.emplace_back(static_cast<std::uint32_t>(allocation.pushByte), allocation.adjustment);
                continue;
            }
            Require(allocation.dataAllocation >= 0, "a guest buffer off the storage buffer offset alignment in a shader without shader data is not implemented");
            auto& data = allocations[static_cast<std::size_t>(allocation.dataAllocation)];
            Require(data.buffer != nullptr && allocation.dataByte < data.size, "guest buffer offset lies outside the shader's data buffer");
            data.buffer->Bytes()[allocation.dataByte] = static_cast<std::byte>(allocation.adjustment);
            dataPatches.push_back({static_cast<std::size_t>(allocation.dataAllocation), allocation.dataByte, allocation.adjustment});
        }
        timing.descriptorsMs += phase(BuildPhase::Descriptors);
        noteReusable();
        completed = true;
        if (BuildProfiled()) {
            timing.completeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stageStart).count();
            addBuildPhase(BuildPhase::StageB, timing.completeMs);
            reportDescriptorCaches();
        }
    } catch (...) {
        release();
        throw;
    }
}

// Whether this build can serve later dispatches of the same content (see ContentKey): nothing to do
// once the GPU completed (no BDA, no copied written buffers, no lease) and every guest buffer bound
// in place through a host import (or staged in device memory from one, see
// GuestBufferMemory::DirectRegions) whose identity is recorded for Revalidate.
namespace {

// APS5_NO_TEMPLATE_DATA_REFRESH=1: compute keys keep the ShaderData/FlattenedSrt words and no
// template refreshes its data buffers (see ShaderResources::ContentKey).
bool TemplateDataRefresh() {
    static const bool enabled = std::getenv("APS5_NO_TEMPLATE_DATA_REFRESH") == nullptr;
    return enabled;
}

bool DataRole(ShaderRecompiler::DescriptorRole role) {
    return role == ShaderRecompiler::DescriptorRole::ShaderData || role == ShaderRecompiler::DescriptorRole::FlattenedSrt;
}

// The largest data buffer vkCmdUpdateBuffer refreshes; a template with a bigger one is not reused.
constexpr std::size_t MaxRefreshBytes = 65536;

// FNV-1a over one data buffer's words, its count first (DataWordsHash).
constexpr std::uint64_t FnvOffset = 14695981039346656037ull;
constexpr std::uint64_t FnvPrime = 1099511628211ull;
void mixDataWords(std::uint64_t& hash, std::span<const std::uint32_t> words) {
    hash = (hash ^ static_cast<std::uint64_t>(words.size())) * FnvPrime;
    for (const auto word : words) hash = (hash ^ word) * FnvPrime;
}

// ResourceCache::Find and Touch calls (the [rescache] line).
std::atomic<std::uint64_t> resourceCacheFinds{0};
std::atomic<std::uint64_t> resourceCacheTouches{0};

}

void ShaderResources::noteReusable() {
    // Taken whether or not the object turns out reusable: the lookups' records are this build's.
    captureValidation();
    reusable = false;
    directRegions.clear();
    if (HoldsLease()) {
        // A draw's address-based build is reusable when it can be shared (see ReuseAddressDraws);
        // it records no direct region: a use proves the address space itself instead.
        if (!addressReuse) return;
        addressRefusal = sharingRefusal();
        reusable = addressRefusal == AddressRefusal::None;
        return;
    }
    if (NeedsCompletion()) return;
    if (TemplateDataRefresh() && std::any_of(allocations.begin(), allocations.end(), [](const Allocation& allocation) { return allocation.buffer != nullptr && !allocation.guest && allocation.size > MaxRefreshBytes; })) return;
    const auto regions = guestMemory.DirectRegions();
    if (!regions.has_value()) return;
    for (const auto& [begin, end] : *regions) {
        // No reconcile here: the upload just took these imports, and the set is about to be recorded
        // against them.
        auto serial = HostImportSerial(context, begin, static_cast<std::size_t>(end - begin), false);
        // Read-only image mirrors (exe ranges) are as stable as imports; their serials have the top
        // bit set, so the two spaces never collide.
        if (serial == 0) serial = ImageMirrorSerial(context, begin, static_cast<std::size_t>(end - begin));
        if (serial == 0) return;
        directRegions.push_back({begin, end, serial});
    }
    reusable = true;
}

bool ShaderResources::keepsTemplateRecords() const {
    static const bool always = std::getenv("APS5_KEEP_TEMPLATE_RECORDS") != nullptr;
    return always || (addressReuse && usesBda) || (!usesBda && !usesFaultBuffer);
}

bool ShaderResources::ReuseAddressDraws() {
    // A lease synced as soon as its work is recorded (APS5_SYNC_LEASE_DISPATCH) is the behaviour
    // before deferred release, which sharing builds on.
    static const bool enabled = std::getenv("APS5_REUSE_ADDRESS_DRAWS") != nullptr && !SyncLeaseWork();
    return enabled;
}

ShaderResources::AddressRefusal ShaderResources::sharingRefusal() const {
    // The written-page slots of the fault buffer are filled by the GPU and cleared by the
    // completion that read them: two uses in flight would lose each other's pages.
    if (bda == nullptr || bda->ScansWrittenPages()) return AddressRefusal::StoresByAddress;
    switch (guestMemory.Shareable()) {
        case GuestBufferMemory::ShareRefusal::None: return AddressRefusal::None;
        case GuestBufferMemory::ShareRefusal::OwnRegions: return AddressRefusal::OwnRegions;
        case GuestBufferMemory::ShareRefusal::MirrorWrites: return AddressRefusal::MirrorWrites;
        default: return AddressRefusal::NoSpace;
    }
}

std::shared_ptr<const DrawRecipe> ShaderResources::FindPlan(std::uint64_t planKey) const {
    for (const auto& [key, plan] : plans) {
        if (key == planKey) return plan;
    }
    return nullptr;
}

bool ShaderResources::AttachPlan(std::uint64_t planKey, std::shared_ptr<const DrawRecipe> plan) {
    // One template serves a material's draws into a handful of passes (the state keys differ by
    // target, viewport or blend); eight covers them without the list becoming a search.
    // APS5_DRAW_PLAN_SLOTS=<n> (1 to 64) keeps n instead.
    static const std::size_t MaxPlans = [] {
        const char* text = std::getenv("APS5_DRAW_PLAN_SLOTS");
        const auto value = text != nullptr ? std::strtoul(text, nullptr, 0) : 8ul;
        return static_cast<std::size_t>(std::clamp(value, 1ul, 64ul));
    }();
    plans.erase(std::remove_if(plans.begin(), plans.end(), [&](const auto& entry) { return entry.first == planKey; }), plans.end());
    plans.insert(plans.begin(), {planKey, std::move(plan)});
    if (plans.size() <= MaxPlans) return false;
    plans.pop_back();
    return true;
}

ShaderResources::SharedLease ShaderResources::ShareLease() {
    Require(reusable && guestMemory.HoldsLease() && !guestMemory.Shared(), "only a reusable address-based build shares its lease");
    return guestMemory.Share();
}

ShaderResources::SharedLease ShaderResources::AcquireSharedLease(std::span<const GuestMemorySnapshot> snapshots, SharedMiss& miss) {
    miss = SharedMiss::Faulted;
    if (faulted.load(std::memory_order_relaxed)) return nullptr;
    auto failure = GuestBufferMemory::SharedFailure::None;
    auto lease = guestMemory.AcquireShared(snapshots, failure);
    miss = failure == GuestBufferMemory::SharedFailure::None ? SharedMiss::None : failure == GuestBufferMemory::SharedFailure::Snapshot ? SharedMiss::Snapshot : SharedMiss::Space;
    return lease;
}

bool ShaderResources::NeverReusable(std::span<const CompiledShader> shaders) {
    return std::any_of(shaders.begin(), shaders.end(), [](const CompiledShader& shader) {
        return shader.program != nullptr && std::any_of(shader.program->bindings.begin(), shader.program->bindings.end(), [](const ShaderRecompiler::DescriptorBinding& binding) { return binding.role == ShaderRecompiler::DescriptorRole::BdaPagetable || binding.role == ShaderRecompiler::DescriptorRole::FaultBuffer; });
    });
}

// APS5_PROFILE_DRAW: the per-device descriptor caches' counters, every 10 s.
void ShaderResources::reportDescriptorCaches() const {
    static std::mutex reportMutex;
    static auto lastReport = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(reportMutex);
    if (now - lastReport < std::chrono::seconds(10)) return;
    lastReport = now;
    const auto descriptors = context.descriptorCache != nullptr ? context.descriptorCache->Counters() : DescriptorCache::Stats{};
    const auto samplerHits = context.samplerCache != nullptr ? context.samplerCache->Hits() : 0;
    const auto samplerMisses = context.samplerCache != nullptr ? context.samplerCache->Misses() : 0;
    AgcDriver::ReportLine("[descriptors] layouts %llu hits / %llu created, sets %llu from %llu pools, %llu recycled, samplers %llu hits / %llu created\n", static_cast<unsigned long long>(descriptors.layoutHits), static_cast<unsigned long long>(descriptors.layoutMisses), static_cast<unsigned long long>(descriptors.sets), static_cast<unsigned long long>(descriptors.pools), static_cast<unsigned long long>(descriptors.recycled), static_cast<unsigned long long>(samplerHits), static_cast<unsigned long long>(samplerMisses));
}

std::vector<std::uint32_t> ShaderResources::ContentKey(const CompiledShader& shader, bool dataWords, bool movableBuffers) {
    std::vector<std::uint32_t> key;
    AppendContentKey(key, shader, dataWords, movableBuffers);
    return key;
}

void ShaderResources::AppendContentKey(std::vector<std::uint32_t>& key, const CompiledShader& shader, bool dataWords, bool movableBuffers) {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    key.reserve(key.size() + 8 + program.bindings.size() * 12);
    key.push_back(dataWords ? 1u : 0u);
    key.push_back(static_cast<std::uint32_t>(shader.stage));
    key.push_back(static_cast<std::uint32_t>(program.variantId));
    key.push_back(static_cast<std::uint32_t>(program.variantId >> 32u));
    key.push_back(static_cast<std::uint32_t>(program.bindings.size()));
    const auto packBits = [&](const std::vector<bool>& bits) {
        key.push_back(static_cast<std::uint32_t>(bits.size()));
        std::uint32_t word = 0;
        for (std::size_t i = 0; i < bits.size(); ++i) {
            if (bits[i]) word |= 1u << (i % 32u);
            if (i % 32u == 31u || i + 1 == bits.size()) {
                key.push_back(word);
                word = 0;
            }
        }
    };
    for (const auto& binding : program.bindings) {
        key.insert(key.end(), {static_cast<std::uint32_t>(binding.kind), static_cast<std::uint32_t>(binding.role), binding.descriptorSet, binding.binding, binding.count, binding.readOnly ? 1u : 0u, binding.imageShape.has_value() ? static_cast<std::uint32_t>(*binding.imageShape) + 1u : 0u, static_cast<std::uint32_t>(binding.guestDescriptor.size())});
        if (movableBuffers && binding.role == ShaderRecompiler::DescriptorRole::GuestBuffers && binding.guestDescriptor.size() == static_cast<std::size_t>(binding.count) * 4u) {
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u;
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                const bool empty = words[2] == 0 || (words[0] == 0 && (words[1] & 0xffffu) == 0);
                if (written || empty) key.insert(key.end(), words, words + 4);
                else key.insert(key.end(), {0u, words[1] & 0xffff0000u, 0u, words[3]});
            }
        } else if ((dataWords && !movableBuffers) || !DataRole(binding.role)) {
            key.insert(key.end(), binding.guestDescriptor.begin(), binding.guestDescriptor.end());
        }
        packBits(binding.imageWritten);
        packBits(binding.samplerDepthCompare);
        packBits(binding.imageDepthCompare);
        // Read-only elements are bound without a write set: an object built for one written set
        // must not serve a build with another (the variant implies it, this makes it explicit).
        packBits(binding.bufferWritten);
    }
}

namespace {

// APS5_PROFILE_DRAW: Revalidate outcomes and time, printed as [rescache] every 10 s next to the
// device's hit/miss line: failed = the object could not be reused (whichever image path it took);
// of the reused ones, fast = every image proved current from stamps, full = the lookups were repeated.
// Why the fast path left an object to the full walk (the "fast-fail by reason" counts).
using FastFail = ShaderResources::FastFail;
constexpr std::array<const char*, static_cast<std::size_t>(FastFail::Count)> FastFailNames{"no record", "collect", "pending image", "evicted image", "memory changed", "keys", "cleared view", "storage keys"};
using OwnRefreshFallback = ShaderResources::OwnRefreshFallback;
constexpr std::array<const char*, static_cast<std::size_t>(OwnRefreshFallback::Count)> OwnRefreshFallbackNames{"disabled", "snapshot texture", "cleared view", "foreign view", "surface key", "not imported", "uncached", "re-run failed"};

struct RevalidateProfile {
    std::atomic<std::uint64_t> calls{0};
    std::atomic<std::uint64_t> nanoseconds{0};
    std::atomic<std::uint64_t> fast{0};
    std::atomic<std::uint64_t> full{0};
    std::atomic<std::uint64_t> failed{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FastFail::Count)> fastFails{};
    // Epoch gate: registry scans skipped because the pending serial was unchanged, and import
    // serial loops skipped because the table's identity was unchanged.
    std::atomic<std::uint64_t> serialSkips{0};
    std::atomic<std::uint64_t> importSkips{0};
    // Elements with DCC keys the fast path accepted through the images' key proofs.
    std::atomic<std::uint64_t> keysProven{0};
    std::atomic<std::uint64_t> storageKeysProven{0};
    // The 'pending image' fast-fails by what overlapped: a surface with an own object (the query
    // excepts it, so the overlapping image is foreign to the surface) or a snapshot texture (no own
    // object); and the fails whose overlapping surfaces all have an own object (the T1 trim of
    // design_cpu_final applies: flush the foreign image, refresh the own object, no full walk).
    std::atomic<std::uint64_t> pendingForeign{0};
    std::atomic<std::uint64_t> pendingSnapshot{0};
    std::atomic<std::uint64_t> pendingT1Eligible{0};
    // The full walks by the reason of the fast proof that preceded them (fastFails also counts the
    // Pending failures the own-object refresh resolved), the Pending failures it resolved (with the
    // objects it refreshed: storage images, view sources) and those it left to the walk by reason;
    // view surfaces a foreign image overlapped that the proof accepted because FindPending names
    // their own source; surfaces a foreign image was still pending over after their own object's
    // refresh (its alias, a straddling image) that the re-run accepted; proofs the verify switch
    // checked against the full walk.
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(FastFail::Count)> fullByReason{};
    std::atomic<std::uint64_t> ownRefreshed{0};
    std::atomic<std::uint64_t> ownStorageRefreshes{0};
    std::atomic<std::uint64_t> ownViewRefreshes{0};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(OwnRefreshFallback::Count)> ownFallbacks{};
    std::atomic<std::uint64_t> ownSourceViews{0};
    std::atomic<std::uint64_t> refreshedOverlaps{0};
    std::atomic<std::uint64_t> proofsVerified{0};
    std::atomic<std::int64_t> lastReport{0};
};

RevalidateProfile& Revalidations() {
    static RevalidateProfile profile;
    return profile;
}

void countFastFail(FastFail reason) {
    if (BuildProfiled()) Revalidations().fastFails[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void countPendingFail(std::span<const StorageTexture::PendingQuery> pending) {
    if (!BuildProfiled()) return;
    auto& profile = Revalidations();
    bool eligible = true;
    for (const auto& query : pending) {
        if (!query.overlaps) continue;
        if (query.except != nullptr) {
            profile.pendingForeign.fetch_add(1, std::memory_order_relaxed);
        } else {
            profile.pendingSnapshot.fetch_add(1, std::memory_order_relaxed);
            eligible = false;
        }
    }
    if (eligible) profile.pendingT1Eligible.fetch_add(1, std::memory_order_relaxed);
}

void countKeysProven(bool storage) {
    if (BuildProfiled()) (storage ? Revalidations().storageKeysProven : Revalidations().keysProven).fetch_add(1, std::memory_order_relaxed);
}

void countFullWalk(FastFail reason) {
    if (BuildProfiled() && reason != FastFail::Count) Revalidations().fullByReason[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void countOwnRefreshFallback(OwnRefreshFallback reason) {
    if (BuildProfiled()) Revalidations().ownFallbacks[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
}

void countOwnRefresh(bool storage) {
    TextureCounts().ownRefreshes.fetch_add(1, std::memory_order_relaxed);
    if (BuildProfiled()) (storage ? Revalidations().ownStorageRefreshes : Revalidations().ownViewRefreshes).fetch_add(1, std::memory_order_relaxed);
}

// APS5_NO_OWN_IMAGE_REFRESH=1: a Pending failure of the fast proof takes the full walk, as before
// the own-object refresh (T1).
bool OwnImageRefreshEnabled() {
    static const bool disabled = std::getenv("APS5_NO_OWN_IMAGE_REFRESH") != nullptr;
    return !disabled;
}

// APS5_VERIFY_PROOFS=1: after every own-object refresh the full walk runs beside the proof and the
// process aborts when it would return another object or upload again (a different decision).
bool VerifyProofs() {
    static const bool enabled = std::getenv("APS5_VERIFY_PROOFS") != nullptr;
    return enabled;
}

// Whether a sampled surface's keys are the very keys of the image the view follows (the same
// metadata, extent, format and alpha placement: the same scan), so the image's proof serves it.
bool SameKeySurface(const StorageTexture* source, const GuestTextureResource& resource, std::uint64_t guestBytes) {
    if (source == nullptr) return false;
    const auto& own = source->Descriptor();
    return own.dccAddress == resource.dccAddress && source->GuestBytes() == guestBytes && own.format == resource.format && own.dccAlphaOnMsb == resource.dccAlphaOnMsb;
}

void countRevalidate(bool fast, bool ok, std::chrono::steady_clock::time_point start) {
    auto& profile = Revalidations();
    const auto now = std::chrono::steady_clock::now();
    profile.calls.fetch_add(1, std::memory_order_relaxed);
    profile.nanoseconds.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count()), std::memory_order_relaxed);
    (!ok ? profile.failed : fast ? profile.fast : profile.full).fetch_add(1, std::memory_order_relaxed);
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    auto last = profile.lastReport.load();
    if (nowMs - last < 10000 || !profile.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto byReason = [](const auto& counts, const auto& names) {
        std::string reasons;
        for (std::size_t i = 0; i < names.size(); ++i) {
            const auto count = counts[i].load(std::memory_order_relaxed);
            if (count == 0) continue;
            reasons += " " + std::string(names[i]) + " " + std::to_string(count);
        }
        return reasons;
    };
    const auto reasons = byReason(profile.fastFails, FastFailNames);
    const auto fullReasons = byReason(profile.fullByReason, FastFailNames);
    const auto fallbacks = byReason(profile.ownFallbacks, OwnRefreshFallbackNames);
    AgcDriver::ReportLine("[rescache] revalidate %llu calls %.1f ms: fast %llu, full %llu, failed %llu; fast-fail by reason:%s; full walks: T1 refreshed %llu (%llu storage images, %llu view sources, %llu views served by their own pending source, %llu overlaps left by the refresh accepted), full by reason:%s, T1 fallback by reason:%s, proofs verified %llu; fast-fail pending image: own 0 (the query excepts the own object), foreign %llu, snapshot %llu, T1 eligible %llu; keys proven: %llu sampled, %llu storage; epoch gate: %llu registry scans skipped, %llu import loops skipped\n", static_cast<unsigned long long>(profile.calls.load()), profile.nanoseconds.load() / 1e6, static_cast<unsigned long long>(profile.fast.load()), static_cast<unsigned long long>(profile.full.load()), static_cast<unsigned long long>(profile.failed.load()), reasons.c_str(), static_cast<unsigned long long>(profile.ownRefreshed.load()), static_cast<unsigned long long>(profile.ownStorageRefreshes.load()), static_cast<unsigned long long>(profile.ownViewRefreshes.load()), static_cast<unsigned long long>(profile.ownSourceViews.load()), static_cast<unsigned long long>(profile.refreshedOverlaps.load()), fullReasons.c_str(), fallbacks.c_str(), static_cast<unsigned long long>(profile.proofsVerified.load()), static_cast<unsigned long long>(profile.pendingForeign.load()), static_cast<unsigned long long>(profile.pendingSnapshot.load()), static_cast<unsigned long long>(profile.pendingT1Eligible.load()), static_cast<unsigned long long>(profile.keysProven.load()), static_cast<unsigned long long>(profile.storageKeysProven.load()), static_cast<unsigned long long>(profile.serialSkips.load()), static_cast<unsigned long long>(profile.importSkips.load()));
}

// Revalidate calls answered by a shared build's proof of the same collect epoch (cumulative).
std::atomic<std::uint64_t> proofsReused{0};

// Window collects (APS5_REUSE_ADDRESS_DRAWS; APS5_COLLECT_WINDOW_KIB, default 2048, 0: none).
// A walk costs about 8 us whatever its size plus 34 ns a page, and the first use of every cached
// snapshot and of every proved surface in a collect epoch is a walk of its own: a few thousand a
// frame. The collect memo answers any range whose pages the epoch already walked, and a walk
// only visits the pages it does not cover, so one walk of the aligned window around an element
// serves every other element of the window for the rest of the epoch. A window walk of 2 MiB
// costs 25 us, three single walks: a window is walked whole only when the last epoch that
// touched it made at least APS5_COLLECT_WINDOW_MIN_USES (3) collects in it (the draws of one
// pass come back with the same buffers), and never for a range that does not fit one window.
// `bounds` clamps the window to the range the element is known to lie in (an imported
// allocation: committed and watched as a whole); a window that cannot be walked whole (it
// leaves the watched arena, or covers uncommitted pages) is left alone for the next 256 epochs
// that touch it. Exact under the epoch contract: it only makes more pages "collected in this
// epoch", which is what every collect of the thread already does for its own range.
struct CollectWindows {
    std::uint64_t bytes;
    std::uint32_t minUses;
};

const CollectWindows& CollectWindowSetting() {
    static const CollectWindows setting = [] {
        const char* kib = std::getenv("APS5_COLLECT_WINDOW_KIB");
        const char* uses = std::getenv("APS5_COLLECT_WINDOW_MIN_USES");
        auto bytes = ShaderResources::ReuseAddressDraws() ? (kib != nullptr ? std::strtoull(kib, nullptr, 10) : 2048ull) << 10u : 0ull;
        // A power of two of at least a tracker block, so windows nest on the tracker's blocks.
        if (bytes != 0) bytes = std::bit_ceil(std::max<std::uint64_t>(bytes, 65536));
        return CollectWindows{bytes, static_cast<std::uint32_t>(std::max(1ull, uses != nullptr ? std::strtoull(uses, nullptr, 10) : 3ull))};
    }();
    return setting;
}

struct CollectWindowCounts {
    std::atomic<std::uint64_t> walks{0};
    std::atomic<std::uint64_t> served{0};
    std::atomic<std::uint64_t> refused{0};
};

CollectWindowCounts& CollectWindowCounters() {
    static CollectWindowCounts counts;
    return counts;
}

std::uint64_t CollectWindowed(std::uint64_t address, std::size_t bytes, const std::pair<std::uint64_t, std::uint64_t>* bounds = nullptr) {
    const auto& setting = CollectWindowSetting();
    const auto epoch = GuestMemory::ThreadCollectEpoch();
    if (setting.bytes == 0 || epoch == 0 || bytes == 0 || bytes >= setting.bytes) return GuestMemory::CollectWrites(address, bytes);
    const auto window = address / setting.bytes;
    if ((address + bytes - 1) / setting.bytes != window) return GuestMemory::CollectWrites(address, bytes);
    // Per thread, like the epochs: the window, the epoch that touched it last with its collects
    // so far, the collects of the epoch before that one, whether this epoch walked it whole.
    struct Slot {
        std::uint64_t window = ~0ull;
        std::uint64_t epoch = 0;
        std::uint32_t uses = 0;
        std::uint32_t previous = 0;
        std::uint32_t skip = 0;
        bool walked = false;
    };
    thread_local std::array<Slot, 4096> slots;
    auto& slot = slots[static_cast<std::size_t>((window * 0x9e3779b97f4a7c15ull) >> 52u)];
    if (slot.window != window) slot = Slot{window, epoch, 0, 0, 0, false};
    if (slot.epoch != epoch) {
        slot.previous = slot.uses;
        slot.uses = 0;
        slot.walked = false;
        slot.epoch = epoch;
        if (slot.skip != 0) --slot.skip;
    }
    ++slot.uses;
    auto& counts = CollectWindowCounters();
    if (slot.walked) {
        counts.served.fetch_add(1, std::memory_order_relaxed);
    } else if (slot.previous >= setting.minUses && slot.skip == 0) {
        auto begin = window * setting.bytes;
        auto end = begin + setting.bytes;
        if (bounds != nullptr) {
            begin = std::max(begin, bounds->first & ~std::uint64_t{4095});
            end = std::min(end, (bounds->second + 4095) & ~std::uint64_t{4095});
        }
        if (begin <= address && address + bytes <= end && GuestMemory::CollectWrites(begin, static_cast<std::size_t>(end - begin)) != 0) {
            slot.walked = true;
            counts.walks.fetch_add(1, std::memory_order_relaxed);
        } else {
            slot.skip = 256;
            counts.refused.fetch_add(1, std::memory_order_relaxed);
        }
    }
    // A memo hit after a window walk: the generation every collect of the epoch returns.
    return GuestMemory::CollectWrites(address, bytes);
}

// APS5_NO_EPOCH_REVALIDATE=1: the per-element registry, cache and stamp checks and the per-region
// flush and import lookups of every Revalidate, as before the epoch gate.
bool EpochRevalidate() {
    static const bool enabled = std::getenv("APS5_NO_EPOCH_REVALIDATE") == nullptr;
    return enabled;
}

}

// Turns the lookups' records into this object's per-texture validation records (see
// ValidatedSurface); the last record of an object is the freshest.
void ShaderResources::captureValidation() {
    if (!keepsTemplateRecords()) {
        validatedTextures.clear();
        lookupLog.clear();
        return;
    }
    const auto record = [](const void* object) -> const LookupRecord* {
        for (auto it = lookupLog.rbegin(); it != lookupLog.rend(); ++it) {
            if (it->object == object) return &*it;
        }
        return nullptr;
    };
    validatedTextures.assign(textures.size(), {});
    depthHoldingSeen = DepthHoldingGeneration();
    for (std::size_t i = 0; i < textures.size(); ++i) {
        if (const auto* found = record(textures[i].get())) validatedTextures[i] = {found->resource, found->bytes, found->keys, found->generation, 0, found->source, true};
    }
    lookupLog.clear();
}

// Proves every texture and storage image still current without repeating its lookup: exactly the
// "unchanged" branches of cachedTexture and StorageTexture::Refresh, evaluated from write stamps,
// the pending-results registry and the DCC keys. Per element: collect first (stamps come only from
// collects, and the generation a record moves to must predate the checks, so a write landing between
// them is stamped newer: the rule of Driver's dispatch cache), then no other image may have results
// pending over the memory (a lookup would flush them into it), a storage image must still be the
// cache's image of its surface, the memory must be unchanged since the object's content matched it,
// and the DCC keys must be what the content was made under whenever the surface has any (a fast
// clear touches only the keys). Anything else, including a failed collect (memory outside the arena,
// uncommitted pages), is left to the full walk.
// The epoch gate: while the pending registry's serial is what this object's last proof saw
// (pendingSerialSeen), the registry is what it was then (every change bumps the serial under the
// registry mutex), so the pending-image checks are skipped; otherwise one scan answers them for
// every surface at once. The stamp checks go through one tracker lock, the cache checks through
// the images' cached flags and one touch of the cache.
bool ShaderResources::fastRevalidate(std::uint64_t serialBefore, std::span<const PendingOverlap> refreshed, FastFail& reason, std::vector<PendingOverlap>& overlapping, bool& accepted) {
    reason = FastFail::Count;
    overlapping.clear();
    accepted = false;
    if (!EpochRevalidate()) return fastRevalidateEach();
    const auto fail = [&reason](FastFail why) {
        reason = why;
        countFastFail(why);
        return false;
    };
    if (validatedTextures.size() != textures.size()) return fail(FastFail::NoRecord);
    if (!textures.empty() && depthHoldingSeen != DepthHoldingGeneration()) return fail(FastFail::Changed);
    const bool unchanged = pendingSerialSeen != 0 && pendingSerialSeen == serialBefore;
    const bool keyProofs = KeyFastPath();
    thread_local std::vector<GuestMemory::UnchangedQuery> queries;
    thread_local std::vector<StorageTexture::PendingQuery> pending;
    // The element each pending query stands for, in query order.
    thread_local std::vector<PendingOverlap> owners;
    thread_local std::vector<const StorageTexture*> images;
    thread_local std::vector<DccKeys> scannedKeys;
    queries.clear();
    pending.clear();
    owners.clear();
    images.clear();
    scannedKeys.assign(textures.size(), DccKeys::Uncompressed);
    const auto query = [&](std::uint64_t begin, std::uint64_t end, const StorageTexture* except, const StorageTexture* identity, PendingOverlap owner) {
        pending.push_back({begin, end, except, identity, false});
        owners.push_back(owner);
    };
    for (std::size_t i = 0; i < textures.size(); ++i) {
        auto& surface = validatedTextures[i];
        if (!surface.valid) return fail(FastFail::NoRecord);
        const auto address = surface.resource.baseAddress;
        const auto bytes = static_cast<std::size_t>(surface.bytes);
        surface.collected = CollectWindowed(address, bytes);
        if (surface.collected == 0) return fail(FastFail::Collect);
        const auto* source = surface.source;
        if (!keyProofs) {
            // Without proofs the keys are compared with the record's (a cleared view's identity
            // below is the record's too): the scan repeats on every call, as before.
            if (surface.resource.dccAddress != 0 && TextureClearKeys(surface.resource, surface.bytes) != surface.keys) return fail(FastFail::Keys);
            scannedKeys[i] = surface.keys;
            if (!unchanged) query(address, address + bytes, source, source != nullptr && surface.keys != DccKeys::Uncompressed ? source : nullptr, {i, false, source != nullptr && surface.keys == DccKeys::Uncompressed});
        } else if (source != nullptr) {
            // The view follows its image, so it stays valid while the image would Refresh as
            // unchanged: the image's own keys must be what its content was uploaded under (its
            // proof answers without a scan while they are unstamped), and its memory unchanged
            // since its generation (the query below). The surface's own keys come from the same
            // proof when they are the image's, else from the texture's.
            const auto& own = source->Descriptor();
            const auto sourceKeys = own.dccAddress != 0 ? ProvedClearKeys(own, source->GuestBytes(), source->KeyProof()) : DccKeys::Uncompressed;
            if (sourceKeys != source->UploadedKeys()) return fail(FastFail::Keys);
            const auto keys = surface.resource.dccAddress == 0 ? DccKeys::Uncompressed : SameKeySurface(source, surface.resource, surface.bytes) ? sourceKeys : ProvedClearKeys(surface.resource, surface.bytes, textures[i]->KeyProof());
            if (surface.resource.dccAddress != 0 && surface.resource.dccAddress != own.dccAddress && (keys != DccKeys::Uncompressed || !StorageImageServesKeys(*source, surface.resource.dccAddress))) return fail(FastFail::Keys);
            scannedKeys[i] = keys;
            // A view of a fast-cleared surface stays one only while its image still has results
            // pending over it (cachedTexture's hit rule), unless the image's own descriptor names
            // the surface's metadata (the lookup views the cleared image then). The identity is
            // asked whatever the registry's serial did: the keys may have moved since the last
            // proof while the registry did not.
            if (keys != DccKeys::Uncompressed && !(own.dccAddress == surface.resource.dccAddress && ClearedViewEnabled() && StorageClearAvailable(context, surface.resource.format, keys))) query(address, address + bytes, source, source, {i, false, false});
            else if (!unchanged) query(address, address + bytes, source, nullptr, {i, false, keys == DccKeys::Uncompressed});
            if (surface.resource.dccAddress != 0 || own.dccAddress != 0) countKeysProven(false);
        } else {
            // A snapshot holds its clear texels or the guest bytes: the keys must be what it was made under.
            const auto keys = surface.resource.dccAddress != 0 ? ProvedClearKeys(surface.resource, surface.bytes, textures[i]->KeyProof()) : DccKeys::Uncompressed;
            if (keys != surface.keys) return fail(FastFail::Keys);
            scannedKeys[i] = keys;
            if (!unchanged) query(address, address + bytes, nullptr, nullptr, {i, false, false});
            if (surface.resource.dccAddress != 0) countKeysProven(false);
        }
        if (source != nullptr) {
            if (!source->Cached()) return fail(FastFail::Evicted);
            queries.push_back({address, bytes, source->Generation()});
            images.push_back(source);
        } else {
            queries.push_back({address, bytes, surface.generation});
        }
    }
    for (std::size_t i = 0; i < storageTextures.size(); ++i) {
        if (storageTextures[i] != nullptr && (i >= storageKeys.size() || !StorageImageServesKeys(*storageTextures[i], storageKeys[i]))) return fail(FastFail::StorageKeys);
        if (i != 0 && storageTextures[i] == storageTextures[i - 1]) continue;
        const auto* image = storageTextures[i].get();
        if (image == nullptr) return fail(FastFail::NoRecord);
        const auto& own = image->Descriptor();
        const auto address = own.baseAddress;
        const auto bytes = static_cast<std::size_t>(image->GuestBytes());
        if (CollectWindowed(address, bytes) == 0) return fail(FastFail::Collect);
        if (own.dccAddress != 0) {
            // Refresh's unchanged branch (its key compare against the keys the content was
            // uploaded under, through the image's proof; the memory query below), minus its
            // byte-compare fallback; the image's generation is left where it is.
            if (!keyProofs || ProvedClearKeys(own, bytes, image->KeyProof()) != image->UploadedKeys()) return fail(FastFail::StorageKeys);
            countKeysProven(true);
        }
        if (!unchanged) query(address, address + bytes, image, nullptr, {i, true, false});
        if (!image->Cached()) return fail(FastFail::Evicted);
        queries.push_back({address, bytes, image->Generation()});
        images.push_back(image);
    }
    if (!pending.empty()) {
        if (!StorageTexture::ScanPending(pending)) return fail(FastFail::ClearedView);
        // A foreign image over a view under uncompressed keys is no failure while FindPending
        // names the view's own source: the lookup views the source (cachedTexture's hit rule)
        // and touches nothing else. A surface whose own object T1 refreshed in this Revalidate
        // (`refreshed`) is no failure either: a storage element's lookup is that Refresh and
        // returns the image whatever stays registered over it; a view's lookup hits on the source
        // when FindPending names it, and views the cached image of its key (the source, proved by
        // refreshOwnObjects) when FindPending names nothing. Every other overlap is the full
        // walk's, or T1's (Revalidate).
        const auto refreshedOwner = [&](const PendingOverlap& owner) {
            return std::find_if(refreshed.begin(), refreshed.end(), [&](const PendingOverlap& entry) { return entry.element == owner.element && entry.storage == owner.storage; });
        };
        for (std::size_t k = 0; k < pending.size(); ++k) {
            if (!pending[k].overlaps) continue;
            if (OwnImageRefreshEnabled() && owners[k].viewUncompressed && pending[k].found == pending[k].except) {
                if (BuildProfiled()) Revalidations().ownSourceViews.fetch_add(1, std::memory_order_relaxed);
                accepted = true;
                continue;
            }
            if (const auto entry = refreshedOwner(owners[k]); entry != refreshed.end() && (entry->storage || pending[k].found == pending[k].except || (pending[k].found == nullptr && entry->sourceEligible))) {
                if (BuildProfiled()) Revalidations().refreshedOverlaps.fetch_add(1, std::memory_order_relaxed);
                accepted = true;
                continue;
            }
            overlapping.push_back(owners[k]);
        }
        if (!overlapping.empty()) {
            countPendingFail(pending);
            return fail(FastFail::Pending);
        }
    } else if (unchanged && BuildProfiled()) {
        Revalidations().serialSkips.fetch_add(1, std::memory_order_relaxed);
    }
    if (!GuestMemory::UnchangedSinceAll(queries)) return fail(FastFail::Changed);
    if (!StorageImagesCached(context, images)) return fail(FastFail::Evicted);
    for (const auto* image : images) image->NoteProved();
    for (std::size_t i = 0; i < validatedTextures.size(); ++i) {
        auto& surface = validatedTextures[i];
        if (surface.source == nullptr) surface.generation = surface.collected;
        surface.keys = scannedKeys[i];
    }
    return true;
}

bool ShaderResources::fastRevalidateEach() {
    if (validatedTextures.size() != textures.size()) return false;
    if (!textures.empty() && depthHoldingSeen != DepthHoldingGeneration()) return false;
    for (std::size_t i = 0; i < textures.size(); ++i) {
        auto& surface = validatedTextures[i];
        if (!surface.valid) return false;
        const auto address = surface.resource.baseAddress;
        const auto bytes = static_cast<std::size_t>(surface.bytes);
        surface.collected = GuestMemory::CollectWrites(address, bytes);
        if (surface.collected == 0) return false;
        if (PendingStorageOverlaps(address, bytes, surface.source)) return false;
        if (surface.source != nullptr && surface.resource.dccAddress != 0 && surface.resource.dccAddress != surface.source->Descriptor().dccAddress) return false;
        if (surface.source != nullptr) {
            // The view follows the image: it needs the image current with guest memory, as the
            // lookup's Refresh would make it.
            if (!StorageImageCached(context, surface.source) || !GuestMemory::UnchangedSince(address, bytes, surface.source->Generation())) return false;
        } else if (!GuestMemory::UnchangedSince(address, bytes, surface.generation)) {
            return false;
        }
        if (surface.resource.dccAddress != 0 && TextureClearKeys(surface.resource, surface.bytes) != surface.keys) return false;
        // A view made under fast-clear keys stays one only while its image still has results pending
        // over the surface (the lookup prefers them to the clear); once they are flushed the clear
        // the image cannot see wins and the lookup makes a snapshot (cachedTexture's hit rule).
        if (surface.source != nullptr && surface.keys != DccKeys::Uncompressed && StorageTexture::FindPending(address, surface.bytes).get() != surface.source) return false;
    }
    for (std::size_t i = 0; i < storageTextures.size(); ++i) {
        if (storageTextures[i] != nullptr && (i >= storageKeys.size() || (storageKeys[i] != 0 && storageKeys[i] != storageTextures[i]->Descriptor().dccAddress))) return false;
        // Consecutive elements of one mip chain share the image (see addStorageImageBinding).
        if (i != 0 && storageTextures[i] == storageTextures[i - 1]) continue;
        const auto* image = storageTextures[i].get();
        if (image == nullptr) return false;
        // StorageTexture::Refresh checks the image's own surface, not the element's descriptor (one
        // image serves every descriptor of its memory, with or without a DCC address), and compares
        // its keys with the keys it was uploaded under, which only the image knows (they move to
        // Uncompressed on every write-back; a fast clear after it puts the key bytes back to what a
        // record saw while the image holds last frame's results), so a surface with keys is left to
        // the full walk. Without a DCC address both sides of that compare are Uncompressed whatever
        // happened, and the memory checks below are Refresh's.
        const auto& own = image->Descriptor();
        if (own.dccAddress != 0) return false;
        const auto address = own.baseAddress;
        const auto bytes = static_cast<std::size_t>(image->GuestBytes());
        if (GuestMemory::CollectWrites(address, bytes) == 0) return false;
        if (PendingStorageOverlaps(address, bytes, image)) return false;
        if (!StorageImageCached(context, image)) return false;
        if (!GuestMemory::UnchangedSince(address, bytes, image->Generation())) return false;
    }
    // Every check passed: snapshot textures are current at the collects issued before them. Storage
    // images keep their own generation: nothing was stamped above it, so the blocks a later write
    // stamps are the same set either way.
    for (auto& surface : validatedTextures) {
        if (surface.source == nullptr) surface.generation = surface.collected;
    }
    return true;
}

ShaderResources::OwnRefreshFallback ShaderResources::refreshOwnObjects(std::span<const CompiledShader> shaders, std::span<PendingOverlap> overlapping) {
    const auto listed = [&](std::size_t element, bool storage) {
        return std::find_if(overlapping.begin(), overlapping.end(), [&](const PendingOverlap& overlap) { return overlap.element == element && overlap.storage == storage; });
    };
    // A sampled view's lookup (cachedTexture): the pending image FindPending serves the surface
    // by, if it fits the view, else the cached storage image of the surface's key, refreshed; the
    // view stays the lookup's answer only when that image is its own source. `sourceEligible` is
    // recorded whichever branch answers: a later refresh of this call may flush the source, and
    // the re-run then judges the surface by the other branch.
    const auto refreshView = [&](std::size_t i, PendingOverlap& overlap) {
        const auto& surface = validatedTextures[i];
        const auto& source = textures[i]->SharedStorageSource();
        if (source == nullptr || source.get() != surface.source) return OwnRefreshFallback::Snapshot;
        if (!source->Cached()) return OwnRefreshFallback::Uncached;
        if (!StorageImageServesKeys(*source, surface.resource.dccAddress)) return OwnRefreshFallback::Keys;
        const auto address = surface.resource.baseAddress;
        const auto bytes = static_cast<std::size_t>(surface.bytes);
        const bool imported = SampledFromStorageEligible(context, surface.resource, surface.bytes);
        overlap.sourceEligible = imported && SurfaceKey(context, source->Descriptor()) == SurfaceKey(context, surface.resource);
        auto found = StorageTexture::FindPending(address, surface.bytes);
        if (found != nullptr && !Texture::CanCopyFrom(*found, surface.resource)) found.reset();
        if (found != nullptr) {
            if (found != source) return OwnRefreshFallback::ForeignView;
            if (!GuestMemory::UnchangedSince(address, bytes, source->Generation())) {
                countOwnRefresh(false);
                source->Refresh();
            }
            return OwnRefreshFallback::Count;
        }
        if (!imported) return OwnRefreshFallback::NotImported;
        if (!overlap.sourceEligible) return OwnRefreshFallback::SurfaceKey;
        countOwnRefresh(false);
        // The cache's image of the key is this source while it is cached (an entry is replaced
        // only after its image left the cache), so the lookup would refresh and return it.
        source->Refresh();
        return OwnRefreshFallback::Count;
    };
    // A storage element's lookup (cachedStorageTexture): the cached image of its surface key,
    // refreshed; the key is the element's words', which mapped to this image at the build.
    const auto refreshStorage = [&](std::size_t i) {
        auto& image = storageTextures[i];
        if (image == nullptr || !image->Cached()) return OwnRefreshFallback::Uncached;
        if (i >= storageKeys.size() || !StorageImageServesKeys(*image, storageKeys[i])) return OwnRefreshFallback::Keys;
        countOwnRefresh(true);
        image->Refresh();
        return OwnRefreshFallback::Count;
    };
    std::size_t textureIndex = 0;
    std::size_t storageIndex = 0;
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) return OwnRefreshFallback::Uncached;
        for (const auto& binding : shader.program->bindings) {
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) continue;
            if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) {
                for (std::uint32_t element = 0; element < binding.count; ++element, ++textureIndex) {
                    if (textureIndex >= textures.size()) return OwnRefreshFallback::Uncached;
                    const auto overlap = listed(textureIndex, false);
                    if (overlap == overlapping.end()) continue;
                    if (const auto fallback = refreshView(textureIndex, *overlap); fallback != OwnRefreshFallback::Count) return fallback;
                }
            } else if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
                for (std::uint32_t element = 0; element < binding.count; ++element, ++storageIndex) {
                    if (storageIndex >= storageTextures.size()) return OwnRefreshFallback::Uncached;
                    if (listed(storageIndex, true) == overlapping.end()) continue;
                    if (const auto fallback = refreshStorage(storageIndex); fallback != OwnRefreshFallback::Count) return fallback;
                }
            }
        }
    }
    return OwnRefreshFallback::Count;
}

bool ShaderResources::Revalidate(std::span<const CompiledShader> shaders, ProofReport* report) {
    const GuestMemory::CollectSiteScope collectSite(GuestMemory::CollectSite::Proof);
    if (report != nullptr) *report = {ProofPath::Full, ProofFailure::Other};
    if (!reusable || shaders.empty()) return false;
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    // APS5_NO_FAST_REVALIDATE=1 always repeats the lookups.
    static const bool noFast = std::getenv("APS5_NO_FAST_REVALIDATE") != nullptr;
    const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto finish = [&](bool fast, bool ok, ProofFailure failure = ProofFailure::Other) {
        if (profile) countRevalidate(fast, ok, start);
        if (report != nullptr) report->failure = ok ? ProofFailure::None : failure;
        return ok;
    };
    // A shared address-based build proved in this collect epoch (APS5_REUSE_ADDRESS_DRAWS; a
    // frame's draws use a template tens of times an epoch): the image proof below is a function
    // of what `proofStamp` names, each read before the checks it stands for, plus the CPU's
    // stores, which a second collect in the epoch would not look at again (the collect memo's
    // contract). The stamp: the driver's own stamps (an image flush, a write-back, a key store),
    // the recorder's write notes, the allocation registry, the pending image registry, the
    // caches' evictions and the depth holdings. While it is the stamp of the last successful
    // proof, only the imports are asked again. What it gives up is what the snapshots' epoch
    // reuse gives up (Recorder::EpochSnapshot): a CPU store another thread's walk stamped within
    // the epoch. APS5_NO_PROOF_EPOCH_REUSE=1 proves every use.
    static const bool proofReuse = std::getenv("APS5_NO_PROOF_EPOCH_REUSE") == nullptr && std::getenv("APS5_NO_COLLECT_MEMO") == nullptr;
    ProofStamp proofStamp;
    if (proofReuse && guestMemory.Shared()) {
        if (const auto* recorder = Recorder::Active(); recorder != nullptr) {
            const auto stamp = recorder->CurrentSnapshotStamp(0);
            proofStamp = {stamp.epoch, stamp.driverStores, stamp.writeNotes, stamp.registry, StorageTexture::PendingSerial(), SampledDepartures().load(std::memory_order_relaxed) + StorageDepartures().load(std::memory_order_relaxed), DepthHoldingGeneration()};
        }
        if (proofStamp.epoch != 0 && proofStamp == provedStamp) {
            if (report != nullptr) report->path = ProofPath::Fast;
            proofsReused.fetch_add(1, std::memory_order_relaxed);
            if (!guestMemory.SharedImportsStand()) {
                provedStamp = {};
                return finish(true, false, ProofFailure::Imports);
            }
            return finish(true, true);
        }
    }
    // Loaded before any check (the memo rule of fastRevalidate): kept as the memo only when the
    // registry did not move through the whole proof, flushes of this call included.
    const auto serialBefore = StorageTexture::PendingSerial();
    // (1) Textures and storage images, from stamps when every element allows it (fastRevalidate),
    // else by the same lookups as the build (which refresh or replace them as guest memory changed):
    // they must hand back the very objects the set's views belong to. The build appended them stage
    // by stage, binding by binding, so the walk repeats that order.
    const auto fullWalk = [&] {
        lookupLog.clear();
        std::size_t textureIndex = 0;
        std::size_t storageIndex = 0;
        for (const auto& shader : shaders) {
            if (shader.program == nullptr) return false;
            for (const auto& binding : shader.program->bindings) {
                if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages) continue;
                if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) {
                    const auto elementWords = binding.count != 0 ? binding.guestDescriptor.size() / binding.count : 0;
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
                        const auto resource = DecodeTextureResource(words);
                        const VkComponentMapping components{ComponentSwizzleFor(resource.dstSelX), ComponentSwizzleFor(resource.dstSelY), ComponentSwizzleFor(resource.dstSelZ), ComponentSwizzleFor(resource.dstSelW)};
                        if (textureIndex >= textures.size() || cachedTexture(context, words, resource, components, 0, !binding.imageDepthCompare.empty() && binding.imageDepthCompare.at(element)) != textures[textureIndex]) return false;
                        ++textureIndex;
                    }
                } else if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
                    for (std::uint32_t element = 0; element < binding.count; ++element) {
                        const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
                        if (storageIndex >= storageTextures.size()) return false;
                        std::shared_ptr<StorageTexture> expected;
                        if (SameAsPreviousStorageElement(binding, element) && StorageDedupeEnabled()) expected = storageTextures[storageIndex - 1];
                        else expected = cachedStorageTexture(context, words, DecodeTextureResource(words), storageMips[storageIndex]);
                        if (expected != storageTextures[storageIndex]) return false;
                        ++storageIndex;
                    }
                }
            }
        }
        if (textureIndex != textures.size() || storageIndex != storageTextures.size()) return false;
        // The walk proved every object current again: the records move to what it proved, or one
        // spurious stamp (a label sharing a 64 KiB block with a surface's edge) would keep this
        // object on the full walk for good.
        captureValidation();
        return true;
    };
    thread_local std::vector<PendingOverlap> overlapping;
    thread_local std::vector<PendingOverlap> refreshed;
    refreshed.clear();
    FastFail reason = FastFail::Count;
    bool accepted = false;
    bool fast = !noFast && fastRevalidate(serialBefore, refreshed, reason, overlapping, accepted);
    // T1 (design_cpu_final M3, rule RT1): a Pending failure whose overlapping images are foreign
    // to the surfaces is resolved by the own objects' refresh (what the walk's lookups would do to
    // them) and the fast proof run again, which is then authoritative (it accepts what stays
    // pending over the refreshed surfaces, see fastRevalidate); a fallback, or a second failure,
    // takes the full walk as before.
    bool ownRefreshed = false;
    if (!fast && !overlapping.empty()) {
        auto fallback = OwnRefreshFallback::Count;
        if (!OwnImageRefreshEnabled()) fallback = OwnRefreshFallback::Disabled;
        else {
            refreshed = overlapping;
            fallback = refreshOwnObjects(shaders, refreshed);
        }
        if (fallback == OwnRefreshFallback::Count) {
            fast = fastRevalidate(serialBefore, refreshed, reason, overlapping, accepted);
            if (fast) ownRefreshed = true;
            else fallback = OwnRefreshFallback::Rerun;
        }
        if (fallback != OwnRefreshFallback::Count) countOwnRefreshFallback(fallback);
    }
    if (report != nullptr) report->path = !fast ? ProofPath::Full : ownRefreshed ? ProofPath::OwnRefreshed : ProofPath::Fast;
    if (fast) {
        const auto now = ResidencyClock::Now();
        for (const auto& texture : textures) {
            if (texture != nullptr) texture->NoteResidencyUse(now);
        }
    }
    if (!fast) {
        countFullWalk(reason);
        if (!fullWalk()) {
            const auto failure = [&] {
                switch (reason) {
                    case FastFail::Pending: return ProofFailure::Pending;
                    case FastFail::Evicted: return ProofFailure::Evicted;
                    case FastFail::Changed: return ProofFailure::Changed;
                    case FastFail::Keys: case FastFail::ClearedView: case FastFail::StorageKeys: return ProofFailure::Keys;
                    default: return ProofFailure::Other;
                }
            }();
            return finish(false, false, failure);
        }
    } else if (ownRefreshed || accepted) {
        if (profile && ownRefreshed) Revalidations().ownRefreshed.fetch_add(1, std::memory_order_relaxed);
        if (VerifyProofs()) {
            // The walk after the refresh, or beside a proof that accepted a foreign overlap, must
            // find every object in place and current: another object, or an upload (a moved
            // content version), is a decision the proof got wrong.
            thread_local std::vector<std::pair<const StorageTexture*, std::uint64_t>> versions;
            versions.clear();
            for (const auto& surface : validatedTextures) {
                if (surface.source != nullptr) versions.emplace_back(surface.source, surface.source->Version());
            }
            for (const auto& image : storageTextures) versions.emplace_back(image.get(), image->Version());
            const bool same = fullWalk();
            const auto moved = std::find_if(versions.begin(), versions.end(), [](const auto& entry) { return entry.first->Version() != entry.second; });
            if (!same || moved != versions.end()) {
                AgcDriver::ReportLine("[rescache] APS5_VERIFY_PROOFS: the T1 proof (%s) disagrees with the full walk (%s)\n", ownRefreshed ? "own-object refresh" : "accepted overlap", !same ? "another object" : "an upload");
                std::fflush(stderr);
                std::abort();
            }
            if (profile) Revalidations().proofsVerified.fetch_add(1, std::memory_order_relaxed);
        }
    }
    // (2) The imported buffers the set reads in place: results of storage images pending in them go
    // to guest memory first (as an upload does), then each import must still be the one the set was
    // written against. This comes last because the lookups and flushes above can reconcile imports
    // themselves; nothing else touches them between here and the dispatch being recorded. Under the
    // epoch gate the flush runs only for regions one registry scan finds pending images over (none
    // while the serial is the memo's), and the serial loop only while the import table's identity
    // moved since the last proof (a retire bumps its epoch, a registry change its generation).
    const auto serialLoop = [&] {
        for (const auto& region : directRegions) {
            auto serial = HostImportSerial(context, region.begin, static_cast<std::size_t>(region.end - region.begin), true);
            if (serial == 0) serial = ImageMirrorSerial(context, region.begin, static_cast<std::size_t>(region.end - region.begin));
            if (serial != region.serial) return false;
        }
        return true;
    };
    if (EpochRevalidate()) {
        if (!directRegions.empty() && !(pendingSerialSeen != 0 && pendingSerialSeen == StorageTexture::PendingSerial())) {
            thread_local std::vector<StorageTexture::PendingQuery> regions;
            regions.clear();
            for (const auto& region : directRegions) regions.push_back({region.begin, region.end, nullptr, nullptr, false});
            StorageTexture::ScanPending(regions);
            // A region over a unit shadow's fresh results (a retile bumped the serial) is
            // published by the flush, as one over a pending image is stored.
            for (const auto& region : regions) {
                if (region.overlaps || AnyShadowedOverlaps(region.begin, static_cast<std::size_t>(region.end - region.begin))) StorageTexture::FlushPending(region.begin, static_cast<std::size_t>(region.end - region.begin), nullptr, "imported buffer region");
            }
        }
        if (HostImportsUnchanged(context, importsProof)) {
            if (profile) Revalidations().importSkips.fetch_add(1, std::memory_order_relaxed);
        } else {
            if (!serialLoop()) return finish(fast, false, ProofFailure::Imports);
            importsProof = HostImportsIdentity(context);
        }
    } else {
        for (const auto& region : directRegions) StorageTexture::FlushPending(region.begin, static_cast<std::size_t>(region.end - region.begin), nullptr, "imported buffer region");
        if (!serialLoop()) return finish(fast, false, ProofFailure::Imports);
    }
    // A shared address-based build has no direct region of its own: its buffers are the address
    // space's, which must have kept every import through the lookups and flushes above (the caller
    // holds the space, see AcquireSharedLease).
    if (guestMemory.Shared() && !guestMemory.SharedImportsStand()) return finish(fast, false, ProofFailure::Imports);
    // (3) Buffers staged in device memory (GuestBufferMemory::AllowDeviceStaging) are copied in
    // from their imports anew for this use, after the flushes above and before the work is
    // recorded; a failure to record leaves the object unusable for this dispatch, not the batch.
    if (auto* recorder = Recorder::Active(); recorder != nullptr) {
        try {
            guestMemory.RecordStagingCopies(*recorder);
        } catch (const std::exception& error) {
            AgcDriver::ReportLine("[resources] staging copies of a reused build failed: %s\n", error.what());
            return finish(fast, false);
        }
    }
    pendingSerialSeen = EpochRevalidate() && StorageTexture::PendingSerial() == serialBefore ? serialBefore : 0;
    // The proof stands for the epoch under the stamp read before it (zero: not a shared build).
    provedStamp = proofStamp;
    return finish(fast, true);
}

std::uint64_t ShaderResources::ProofsReused() {
    return proofsReused.load(std::memory_order_relaxed);
}

namespace {

// APS5_PROFILE_DRAW: what makes content keys miss. On a miss the key is compared with the last key
// seen for the same shader variant(s) and the first differing word is charged to its binding's role
// (and to its index within the V# or T# element), printed as [rescache] every 10 s: it says whether
// V# bases of ring allocations, SRT words or image descriptors churn.
struct ChurnCounts {
    std::array<std::uint64_t, 8> roles{};
    std::array<std::uint64_t, 4> bufferWords{};
    std::array<std::uint64_t, 8> imageWords{};
    std::uint64_t firstSeen = 0;
    std::uint64_t sameKey = 0;
    std::uint64_t layout = 0;
    std::uint64_t drawWords = 0;
};

// Dispatch keys and draw keys are counted apart: what churns a draw's key decides the draw steps.
struct ChurnProfile {
    std::mutex mutex;
    std::unordered_map<std::uint64_t, ResourceCache::Key> lastByVariant;
    ChurnCounts dispatch;
    ChurnCounts draws;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

ChurnProfile& Churn() {
    static ChurnProfile profile;
    return profile;
}

// Charges word `diff` of `key` to the binding of the ContentKey that starts at `at` (the layout of
// ShaderResources::ContentKey), moving `at` past that key; false when `diff` lies beyond it.
bool chargeShaderKey(ChurnCounts& profile, const ResourceCache::Key& key, std::size_t& at, std::size_t diff) {
    if (at + 5 > key.size() || diff < at + 5) {
        ++profile.layout;
        return true;
    }
    const bool dataWords = key[at] != 0;
    const auto bindings = key[at + 4];
    at += 5;
    for (std::uint32_t binding = 0; binding < bindings; ++binding) {
        if (at + 8 > key.size() || diff < at + 8) {
            ++profile.layout;
            return true;
        }
        const auto kind = key[at];
        const auto role = key[at + 1];
        const auto descriptorWords = dataWords || !DataRole(static_cast<ShaderRecompiler::DescriptorRole>(role)) ? key[at + 7] : 0u;
        at += 8;
        if (diff < at + descriptorWords) {
            if (role < profile.roles.size()) ++profile.roles[role];
            const auto index = diff - at;
            if (role == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorRole::GuestBuffers)) ++profile.bufferWords[index % 4];
            else if (role == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorRole::GuestImages) && (kind == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorKind::SampledImage) || kind == static_cast<std::uint32_t>(ShaderRecompiler::DescriptorKind::StorageImage))) ++profile.imageWords[index % 8];
            return true;
        }
        at += descriptorWords;
        for (int flags = 0; flags < 4; ++flags) {
            if (at >= key.size()) {
                ++profile.layout;
                return true;
            }
            const auto words = 1 + (static_cast<std::size_t>(key[at]) + 31) / 32;
            if (diff < at + words) {
                ++profile.layout;
                return true;
            }
            at += words;
        }
    }
    return false;
}

}

std::shared_ptr<ShaderResources> ResourceCache::Find(const Key& key, bool probe) {
    resourceCacheFinds.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(mutex);
    const auto found = index.find(key);
    if (found == index.end()) {
        if (!probe) noteMiss(key);
        return nullptr;
    }
    entries.splice(entries.begin(), entries, found->second);
    return found->second->second;
}

void ResourceCache::noteMiss(const Key& key) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile || key.size() < 5) return;
    // The variant(s) the key belongs to: a compute key starts with the data-words flag, the stage
    // and its variant, a draw key with a marker, the device and its stages' keys (Draw.cpp
    // DrawResourceKey).
    const bool draw = key[0] == 0xffffffffu;
    std::uint64_t variants = 0;
    if (draw) {
        std::size_t at = 4;
        for (std::uint32_t stage = 0; stage < key[3] && at + 4 < key.size(); ++stage) {
            variants = variants * 1000003ull ^ (key[at + 3] | (static_cast<std::uint64_t>(key[at + 4]) << 32u));
            at += 1 + key[at];
        }
    } else {
        variants = key[2] | (static_cast<std::uint64_t>(key[3]) << 32u);
    }
    auto& churn = Churn();
    std::lock_guard lock(churn.mutex);
    auto& counts = draw ? churn.draws : churn.dispatch;
    const auto found = churn.lastByVariant.find(variants);
    if (found == churn.lastByVariant.end()) {
        ++counts.firstSeen;
        churn.lastByVariant.emplace(variants, key);
    } else {
        const auto& previous = found->second;
        const auto common = std::min(key.size(), previous.size());
        std::size_t diff = 0;
        while (diff < common && key[diff] == previous[diff]) ++diff;
        if (diff == common && key.size() == previous.size()) {
            ++counts.sameKey;
        } else if (draw) {
            std::size_t at = 4;
            bool charged = false;
            for (std::uint32_t stage = 0; stage < key[3] && at < key.size() && !charged; ++stage) {
                ++at;
                charged = chargeShaderKey(counts, key, at, diff);
            }
            if (!charged) ++counts.drawWords;
        } else {
            std::size_t at = 0;
            if (!chargeShaderKey(counts, key, at, diff)) ++counts.layout;
        }
        found->second = key;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - churn.lastReport < std::chrono::seconds(10)) return;
    churn.lastReport = now;
    const auto report = [](const char* what, const ChurnCounts& churn) {
        std::string line = std::string("[rescache] miss churn (") + what + "), first differing word by role:";
        char item[256];
        for (std::size_t role = 0; role < churn.roles.size(); ++role) {
            if (churn.roles[role] == 0) continue;
            std::snprintf(item, sizeof(item), " %s %llu", roleName(static_cast<ShaderRecompiler::DescriptorRole>(role)), static_cast<unsigned long long>(churn.roles[role]));
            line += item;
            if (role == static_cast<std::size_t>(ShaderRecompiler::DescriptorRole::GuestBuffers)) {
                std::snprintf(item, sizeof(item), " (V# word %llu/%llu/%llu/%llu)", static_cast<unsigned long long>(churn.bufferWords[0]), static_cast<unsigned long long>(churn.bufferWords[1]), static_cast<unsigned long long>(churn.bufferWords[2]), static_cast<unsigned long long>(churn.bufferWords[3]));
                line += item;
            } else if (role == static_cast<std::size_t>(ShaderRecompiler::DescriptorRole::GuestImages)) {
                std::snprintf(item, sizeof(item), " (T# word %llu/%llu/%llu/%llu/%llu/%llu/%llu/%llu)", static_cast<unsigned long long>(churn.imageWords[0]), static_cast<unsigned long long>(churn.imageWords[1]), static_cast<unsigned long long>(churn.imageWords[2]), static_cast<unsigned long long>(churn.imageWords[3]), static_cast<unsigned long long>(churn.imageWords[4]), static_cast<unsigned long long>(churn.imageWords[5]), static_cast<unsigned long long>(churn.imageWords[6]), static_cast<unsigned long long>(churn.imageWords[7]));
                line += item;
            }
        }
        std::snprintf(item, sizeof(item), "; layout %llu, draw target/index %llu, same key %llu (not inserted or evicted), first seen %llu", static_cast<unsigned long long>(churn.layout), static_cast<unsigned long long>(churn.drawWords), static_cast<unsigned long long>(churn.sameKey), static_cast<unsigned long long>(churn.firstSeen));
        line += item;
        AgcDriver::ReportLine("%s\n", line.c_str());
    };
    report("dispatch", churn.dispatch);
    report("draws", churn.draws);
}

void ResourceCache::Insert(const Key& key, std::shared_ptr<ShaderResources> resources, std::vector<std::shared_ptr<ShaderResources>>* evicted) {
    std::lock_guard lock(mutex);
    if (const auto found = index.find(key); found != index.end()) {
        if (evicted != nullptr) evicted->push_back(std::move(found->second->second));
        entries.erase(found->second);
        index.erase(found);
    }
    entries.emplace_front(key, std::move(resources));
    index.emplace(key, entries.begin());
    // Entries pin their textures and storage images past the texture caches' budgets, so the bound
    // stays modest: APS5_RESOURCE_CACHE_ENTRIES (default 1024) covers several frames of distinct
    // dispatch and draw content.
    static const std::size_t capacity = [] {
        const char* value = std::getenv("APS5_RESOURCE_CACHE_ENTRIES");
        const auto parsed = value ? std::strtoull(value, nullptr, 10) : 1024ull;
        return static_cast<std::size_t>(parsed != 0 ? parsed : 1024ull);
    }();
    while (entries.size() > capacity) {
        if (evicted != nullptr) evicted->push_back(std::move(entries.back().second));
        index.erase(entries.back().first);
        entries.pop_back();
    }
}

void ResourceCache::Remove(const Key& key, const ShaderResources* object) {
    std::lock_guard lock(mutex);
    if (object != nullptr) {
        const auto found = index.find(key);
        if (found == index.end() || found->second->second.get() != object) return;
    }
    erase(key);
}

bool ResourceCache::Touch(const Key& key) {
    resourceCacheTouches.fetch_add(1, std::memory_order_relaxed);
    std::lock_guard lock(mutex);
    const auto found = index.find(key);
    if (found == index.end()) return false;
    entries.splice(entries.begin(), entries, found->second);
    return true;
}

std::uint64_t ResourceCache::Finds() {
    return resourceCacheFinds.load(std::memory_order_relaxed);
}

std::uint64_t ResourceCache::Touches() {
    return resourceCacheTouches.load(std::memory_order_relaxed);
}

void ResourceCache::Clear() {
    std::lock_guard lock(mutex);
    index.clear();
    entries.clear();
}

// APS5_TRACE_VRAM: what the two texture caches count against their budgets and the template
// cache's entries. The images alive are the ledger's; more of them than cache entries means
// images evicted from a cache and still held (a template, a plan, a recipe, a batch in flight).
std::string DescribeTextureCaches() {
    std::size_t sampled = 0, storage = 0;
    std::uint64_t sampledBytes = 0, sampledHost = 0, storageBytes = 0;
    {
        auto& cache = Textures();
        std::lock_guard lock(cache.mutex);
        sampled = cache.entries.size();
        sampledBytes = cache.bytes;
        sampledHost = cache.hostBytes;
    }
    {
        auto& cache = StorageTextures();
        std::lock_guard lock(cache.mutex);
        storage = cache.entries.size();
        storageBytes = cache.bytes;
    }
    char text[256];
    std::snprintf(text, sizeof(text), "sampled texture cache %zu entries/%.0f MiB accounted (%.0f MiB host copies); storage texture cache %zu entries/%.0f MiB accounted; template cache %zu entries", sampled, sampledBytes / 1048576.0, sampledHost / 1048576.0, storage, storageBytes / 1048576.0, SharedResourceCache().Size());
    return text;
}

namespace {
const Vram::SectionRegistration textureSection([](const Context& context) {
    auto text = DescribeTextureCaches();
    if (context.descriptorCache != nullptr) {
        const auto descriptors = context.descriptorCache->Counters();
        char piece[128];
        std::snprintf(piece, sizeof(piece), "; descriptors: %llu sets from %llu pools, %llu layouts", static_cast<unsigned long long>(descriptors.sets), static_cast<unsigned long long>(descriptors.pools), static_cast<unsigned long long>(descriptors.layoutMisses));
        text += piece;
    }
    return text;
});
}

std::size_t ResourceCache::Size() const {
    std::lock_guard lock(mutex);
    return entries.size();
}

void ResourceCache::erase(const Key& key) {
    const auto found = index.find(key);
    if (found == index.end()) return;
    entries.erase(found->second);
    index.erase(found);
}

ResourceCache& SharedResourceCache() {
    static auto* cache = new ResourceCache();
    return *cache;
}

DescriptorCache::DescriptorCache(const Context& context) : context(context), destroyLayout(context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")), destroyPool(context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")), freeSets(context.Function<PFN_vkFreeDescriptorSets>("vkFreeDescriptorSets")) {}

DescriptorCache::~DescriptorCache() {
    for (const auto pool : pools) destroyPool(context.device, pool, nullptr);
    for (const auto& [key, layout] : layouts) destroyLayout(context.device, layout, nullptr);
}

VkDescriptorSetLayout DescriptorCache::Layout(std::span<const std::uint32_t> key, std::span<const VkDescriptorSetLayoutBinding> bindings) {
    std::lock_guard lock(mutex);
    std::vector<std::uint32_t> keyCopy(key.begin(), key.end());
    if (const auto found = layouts.find(keyCopy); found != layouts.end()) {
        ++stats.layoutHits;
        return found->second;
    }
    ++stats.layoutMisses;
    VkDescriptorSetLayoutCreateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    info.pBindings = bindings.data();
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(context.device, &info, nullptr, &layout), "vkCreateDescriptorSetLayout");
    layouts.emplace(std::move(keyCopy), layout);
    return layout;
}

namespace {
// What one chain pool holds; a set needing more of any type gets a dedicated pool.
constexpr std::uint32_t ChainPoolSets = 1024;
constexpr std::array<VkDescriptorPoolSize, 4> ChainPoolSizes{{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_SAMPLER, 512}}};

constexpr std::size_t SpareSetsBound = 4096;

bool RecycleSets() {
    static const bool enabled = std::getenv("APS5_NO_SET_RECYCLE") == nullptr;
    return enabled;
}

}

DescriptorCache::SetAllocation DescriptorCache::Allocate(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes) {
    for (const auto& size : sizes) {
        const auto capacity = std::find_if(ChainPoolSizes.begin(), ChainPoolSizes.end(), [&](const auto& item) { return item.type == size.type; });
        if (capacity == ChainPoolSizes.end() || size.descriptorCount > capacity->descriptorCount) return {};
    }
    std::lock_guard lock(mutex);
    if (const auto found = spare.find(layout); found != spare.end() && !found->second.empty()) {
        const auto allocation = found->second.back();
        found->second.pop_back();
        --spareSets;
        ++stats.recycled;
        return allocation;
    }
    const auto allocate = context.Resolved(&DeviceFunctions::allocateDescriptorSets, "vkAllocateDescriptorSets");
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &layout;
    // Newest pool first: it has the most room; a full or fragmented pool is left for its sets to
    // drain and tried again later.
    for (auto it = pools.rbegin(); it != pools.rend(); ++it) {
        allocation.descriptorPool = *it;
        VkDescriptorSet set = VK_NULL_HANDLE;
        const auto result = allocate(context.device, &allocation, &set);
        if (result == VK_SUCCESS) {
            ++stats.sets;
            return {set, *it, layout};
        }
        if (result != VK_ERROR_OUT_OF_POOL_MEMORY && result != VK_ERROR_FRAGMENTED_POOL) Check(result, "vkAllocateDescriptorSets");
    }
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = ChainPoolSets;
    poolInfo.poolSizeCount = static_cast<std::uint32_t>(ChainPoolSizes.size());
    poolInfo.pPoolSizes = ChainPoolSizes.data();
    VkDescriptorPool pool = VK_NULL_HANDLE;
    Check(context.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(context.device, &poolInfo, nullptr, &pool), "vkCreateDescriptorPool");
    pools.push_back(pool);
    ++stats.pools;
    allocation.descriptorPool = pool;
    VkDescriptorSet set = VK_NULL_HANDLE;
    Check(allocate(context.device, &allocation, &set), "vkAllocateDescriptorSets");
    ++stats.sets;
    return {set, pool, layout};
}

void DescriptorCache::Free(const SetAllocation& allocation) noexcept {
    if (allocation.set == VK_NULL_HANDLE || allocation.pool == VK_NULL_HANDLE) return;
    std::lock_guard lock(mutex);
    if (RecycleSets() && allocation.layout != VK_NULL_HANDLE && spareSets < SpareSetsBound) {
        try {
            spare[allocation.layout].push_back(allocation);
            ++spareSets;
            return;
        } catch (...) {
        }
    }
    static_cast<void>(freeSets(context.device, allocation.pool, 1, &allocation.set));
}

DescriptorCache::Stats DescriptorCache::Counters() const {
    std::lock_guard lock(mutex);
    return stats;
}

std::size_t ShaderResources::addGuestBuffer(std::span<const std::uint32_t> words, const ColorTarget* target, std::uint64_t indexAddress, std::size_t indexBytes, bool written, bool atomic, bool read) {
    Require(words.size() == 4, "buffer descriptor must contain four DWORDs");
    Require((words[1] & 0x40000000u) == 0, "buffer descriptor has reserved bits set");
    const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
    Require(descriptor.Type() == 0u, "buffer descriptor uses an unsupported type");
    const auto address = descriptor.Base48();
    const auto byteSize = descriptor.GetSize();
    if (byteSize == 0 || address == 0) {
        allocations.push_back({0, EmptyBufferBytes, false, nullptr, ShaderRecompiler::DescriptorRole::GuestBuffers, false});
        return allocations.size() - 1;
    }
    Require(byteSize <= context.limits.maxStorageBufferRange, "shader buffer exceeds descriptor range limit");
    Require(byteSize <= std::numeric_limits<std::size_t>::max(), "shader buffer size exceeds host address space");
    const auto size = static_cast<std::size_t>(byteSize);
    Require(target == nullptr || !overlap(address, size, target->address, target->bytes), "shader buffer aliases the render target");
    // APS5_ALL_BUFFERS_WRITTEN=1: every element is noted as written, as before bufferWritten existed.
    static const bool allWritten = std::getenv("APS5_ALL_BUFFERS_WRITTEN") != nullptr;
    Require(!written || !overlap(address, size, indexAddress, indexBytes), "writable shader buffer aliases the index buffer");
    written = written || allWritten;
    const bool swept = read && dispatchThreads != 0 && descriptor.Stride() != 0 && dispatchThreads * descriptor.Stride() * 2u >= byteSize;
    if (written) guestMemory.AddWritable(address, size, atomic, swept);
    else {
        guestMemory.AddReadable(address, size);
        ++readOnlyBuffers;
    }
    if (BuildProfiled()) {
        auto& counters = BufferWrites();
        counters.elements.fetch_add(1, std::memory_order_relaxed);
        if (!written) counters.readOnly.fetch_add(1, std::memory_order_relaxed);
    }
    allocations.push_back({address, size, true, nullptr, ShaderRecompiler::DescriptorRole::ShaderData, written});
    return allocations.size() - 1;
}

std::string ShaderResources::Describe() const {
    const auto sample = [](std::uint64_t address, std::uint64_t bytes) {
        // Heaps bound whole can include uncommitted pages; those ranges are not sampled.
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(address), static_cast<std::size_t>(bytes))) return -1.0;
        std::size_t nonzero = 0;
        std::size_t samples = 0;
        const auto* data = reinterpret_cast<const std::uint32_t*>(address);
        const auto words = bytes / 4;
        const auto step = std::max<std::uint64_t>(1, words / 4096);
        for (std::uint64_t i = 0; i < words; i += step, ++samples) nonzero += data[i] != 0;
        return samples == 0 ? 0.0 : static_cast<double>(nonzero) / samples;
    };
    std::string text;
    char line[160];
    for (const auto& range : describedRanges) {
        std::snprintf(line, sizeof(line), " %s 0x%llx+0x%llx(%ux%u f%u t%d) nz=%.2f", range.kind, static_cast<unsigned long long>(range.address), static_cast<unsigned long long>(range.bytes), range.width, range.height, range.format, range.tileMode, sample(range.address, range.bytes));
        text += line;
        if (range.dccAddress != 0) {
            const auto keys = ReadDccKeys(range.dccAddress, range.bytes);
            std::snprintf(line, sizeof(line), " dcc=%s@0x%llx", DccKeysName(keys), static_cast<unsigned long long>(range.dccAddress));
            text += line;
            if (keys == DccKeys::Mixed) {
                // Where the keys change: first key, how many leading keys match it, and the next key.
                const auto* bytes = reinterpret_cast<const std::uint8_t*>(range.dccAddress);
                const auto count = static_cast<std::size_t>(range.bytes / 256u);
                std::size_t run = 1;
                while (run < count && bytes[run] == bytes[0]) ++run;
                std::snprintf(line, sizeof(line), "(%02x x%zu then %02x of %zu)", bytes[0], run, run < count ? bytes[run] : 0u, count);
                text += line;
            }
        }
    }
    // APS5_TRACE_DISPATCH_IO=2 also lists the words of small buffers (constants).
    static const bool words = [] { const char* value = std::getenv("APS5_TRACE_DISPATCH_IO"); return value != nullptr && value[0] == '2'; }();
    const auto appendWords = [&](const std::uint32_t* data, std::size_t bytes) {
        if (!words || bytes > 0x200) return;
        text += " [";
        for (std::size_t i = 0; i < bytes / 4; ++i) {
            char word[12];
            std::snprintf(word, sizeof(word), "%s%08x", i == 0 ? "" : " ", data[i]);
            text += word;
        }
        text += "]";
    };
    for (const auto& allocation : allocations) {
        if (allocation.guest) {
            std::snprintf(line, sizeof(line), " buffer%s 0x%llx+0x%zx nz=%.2f", allocation.written ? "" : "(ro)", static_cast<unsigned long long>(allocation.address), allocation.size, sample(allocation.address, allocation.size));
            text += line;
            if (GuestMemory::Accessible(reinterpret_cast<const void*>(allocation.address), allocation.size)) appendWords(reinterpret_cast<const std::uint32_t*>(allocation.address), allocation.size);
        } else if (allocation.buffer) {
            const auto bytes = allocation.buffer->Bytes();
            std::snprintf(line, sizeof(line), " data+0x%zx nz=%.2f", allocation.size, sample(reinterpret_cast<std::uint64_t>(bytes.data()), allocation.size));
            text += line;
            appendWords(reinterpret_cast<const std::uint32_t*>(bytes.data()), allocation.size);
        }
    }
    return text;
}

std::size_t ShaderResources::addDataBuffer(std::span<const std::uint32_t> words) {
    const auto size = words.size() * sizeof(std::uint32_t);
    Require(size <= context.limits.maxStorageBufferRange, "shader data buffer exceeds descriptor range limit");
    const bool refreshable = TemplateDataRefresh() && size <= MaxRefreshBytes;
    auto buffer = std::make_unique<Buffer>(context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | (refreshable ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0u), GpuReadProperties(GpuReadKind::StorageCopy));
    std::memcpy(buffer->Bytes().data(), words.data(), size);
    Allocation allocation{0, size, false, std::move(buffer)};
    // A draw's address-based build keeps them whatever the size: a later use compares its words
    // with them and rebuilds the buffer from them (MovedReadOnlyBuffers), it never refreshes.
    if ((refreshable || (addressReuse && usesBda)) && keepsTemplateRecords()) allocation.dataWords.assign(words.begin(), words.end());
    allocations.push_back(std::move(allocation));
    mixDataWords(dataWordsHash, allocations.back().dataWords);
    return allocations.size() - 1;
}

void ShaderResources::rehashDataWords() {
    dataWordsHash = FnvOffset;
    // Data buffers are the non-guest allocations with a buffer (an address-role allocation has
    // none), appended in binding order by buildPrepare: the order DataWordsHash(shader) hashes.
    for (const auto& allocation : allocations) {
        if (!allocation.guest && allocation.buffer != nullptr) mixDataWords(dataWordsHash, allocation.dataWords);
    }
}

std::uint64_t ShaderResources::DataWordsHash(const CompiledShader& shader) {
    Require(shader.program != nullptr, "missing compiled shader");
    std::uint64_t hash = FnvOffset;
    for (const auto& binding : shader.program->bindings) {
        if (DataRole(binding.role)) mixDataWords(hash, binding.guestDescriptor);
    }
    return hash;
}

bool ShaderResources::DataWordsDiffer(const CompiledShader& shader) const {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    if (program.bindings.size() != bindings.size()) return true;
    for (std::size_t index = 0; index < program.bindings.size(); ++index) {
        const auto& binding = program.bindings[index];
        if (!DataRole(binding.role)) continue;
        if (bindings[index].allocations.size() != 1) return true;
        const auto& allocation = allocations[bindings[index].allocations.front()];
        if (allocation.dataWords.size() != binding.guestDescriptor.size() || !std::equal(allocation.dataWords.begin(), allocation.dataWords.end(), binding.guestDescriptor.begin())) return true;
    }
    return false;
}

bool ShaderResources::RefreshData(VkCommandBuffer commands, const CompiledShader& shader, Recorder* recorder) {
    Require(shader.program != nullptr, "missing compiled shader");
    const auto& program = *shader.program;
    Require(program.bindings.size() == bindings.size(), "template bindings disagree with the shader");
    bool recorded = false;
    // One [gputime] range of class TemplateDataRefresh around the updates, begun at the first one
    // (a refresh that finds every word equal records nothing and times nothing).
    auto timing = Recorder::NoTiming;
    std::uint64_t refreshedBytes = 0;
    for (std::size_t index = 0; index < program.bindings.size(); ++index) {
        const auto& binding = program.bindings[index];
        if (!DataRole(binding.role)) continue;
        Require(bindings[index].allocations.size() == 1, "data binding without its buffer");
        auto& allocation = allocations[bindings[index].allocations.front()];
        const auto size = binding.guestDescriptor.size() * sizeof(std::uint32_t);
        Require(allocation.buffer != nullptr && !allocation.guest && allocation.size == size && size <= MaxRefreshBytes, "template data buffer cannot take the dispatch's words");
        if (allocation.dataWords.size() == binding.guestDescriptor.size() && std::equal(allocation.dataWords.begin(), allocation.dataWords.end(), binding.guestDescriptor.begin())) continue;
        if (recorder != nullptr && timing == Recorder::NoTiming) timing = recorder->BeginGpuTiming(Recorder::CommandClass::TemplateDataRefresh);
        writeDataWords(commands, bindings[index].allocations.front(), binding.guestDescriptor);
        allocation.dataWords.assign(binding.guestDescriptor.begin(), binding.guestDescriptor.end());
        refreshedBytes += size;
        recorded = true;
    }
    if (timing != Recorder::NoTiming) recorder->EndGpuTiming(timing, refreshedBytes);
    if (recorded) rehashDataWords();
    return recorded;
}

void ShaderResources::writeDataWords(VkCommandBuffer commands, std::size_t allocation, std::span<const std::uint32_t> words) const {
    const auto& buffer = *allocations[allocation].buffer;
    const auto size = words.size() * sizeof(std::uint32_t);
    if (std::none_of(dataPatches.begin(), dataPatches.end(), [&](const DataPatch& patch) { return patch.allocation == allocation; })) {
        context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer")(commands, buffer.Handle(), 0, size, words.data());
        return;
    }
    std::vector<std::uint32_t> patched(words.begin(), words.end());
    auto* bytes = reinterpret_cast<std::byte*>(patched.data());
    for (const auto& patch : dataPatches) {
        if (patch.allocation == allocation && patch.byte < size) bytes[patch.byte] = static_cast<std::byte>(patch.adjustment);
    }
    context.Resolved(&DeviceFunctions::cmdUpdateBuffer, "vkCmdUpdateBuffer")(commands, buffer.Handle(), 0, size, patched.data());
}

void ShaderResources::PrecollectSurfaces() const {
    for (const auto& range : describedRanges) GuestMemory::CollectWrites(range.address, static_cast<std::size_t>(range.bytes));
}

void ShaderResources::addImageBinding(const ShaderRecompiler::DescriptorBinding& binding, VkShaderStageFlags flags) {
    Require(binding.count != 0, "empty descriptor binding");
    if (binding.kind == ShaderRecompiler::DescriptorKind::StorageImage) {
        // Storage images are guest textures the shader writes (looked up in stage B).
        Require(binding.role == ShaderRecompiler::DescriptorRole::GuestImages, "storage image binding has a non-image role");
        Require(binding.guestDescriptor.size() == static_cast<std::size_t>(binding.count) * 8u, "guest storage image descriptors must contain 8 dwords each");
        Require(context.detiler != nullptr, "device texture detiler is unavailable");
        Require(binding.count <= context.limits.maxPerStageDescriptorStorageImages, "shader storage-image descriptors exceed per-stage limits");
        bindings.push_back({{binding.binding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, binding.count, flags, nullptr}, {}, {}});
        plannedStorageImages += binding.count;
        deferredImages.push_back({&binding, bindings.size() - 1});
        return;
    }
    const bool sampledImage = binding.kind == ShaderRecompiler::DescriptorKind::SampledImage;
    const bool samplerKind = binding.kind == ShaderRecompiler::DescriptorKind::Sampler;
    if (!(sampledImage || samplerKind)) Require(false, std::string("unsupported descriptor kind ") + kindName(binding.kind) + " for role " + roleName(binding.role));
    Require((sampledImage && binding.role == ShaderRecompiler::DescriptorRole::GuestImages) || (samplerKind && binding.role == ShaderRecompiler::DescriptorRole::GuestSamplers), "guest image descriptor role disagrees with its kind");
    Require(binding.guestDescriptor.size() % binding.count == 0, "guest image descriptor size is not a multiple of the binding count");
    const auto elementWords = binding.guestDescriptor.size() / binding.count;

    Binding item{{binding.binding, sampledImage ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLER, binding.count, flags, nullptr}, {}, {}};

    if (sampledImage) {
        Require(elementWords == 8, "guest texture descriptor must contain 8 dwords");
        Require(binding.imageShape.has_value(), "guest image binding is missing an image shape");
        Require(context.detiler != nullptr, "device texture detiler is unavailable");
        Require(context.textureCache != nullptr, "device texture cache is unavailable");
        Require(binding.count <= context.limits.maxPerStageDescriptorSampledImages, "shader sampled-image descriptors exceed per-stage limits");
        plannedSampledImages += binding.count;
        Require(plannedSampledImages <= context.limits.maxDescriptorSetSampledImages, "pipeline sampled-image descriptors exceed device limits");
    } else {
        Require(elementWords == 4, "guest sampler descriptor must contain 4 dwords");
        Require(binding.count <= context.limits.maxPerStageDescriptorSamplers, "shader sampler descriptors exceed per-stage limits");
        Require(binding.samplerDepthCompare.size() == binding.count, "guest sampler binding is missing depth comparison metadata");
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
            const bool compareEnable = binding.samplerDepthCompare.at(element);
            static const bool noSamplerCache = std::getenv("APS5_NO_SAMPLER_CACHE") != nullptr;
            if (context.samplerCache != nullptr && !noSamplerCache) {
                samplers.push_back(context.samplerCache->Get(context, words, compareEnable));
            } else {
                auto resource = DecodeSamplerResource(words);
                resource.compareEnable = compareEnable;
                samplers.push_back(std::make_shared<Sampler>(context, resource));
            }
            item.imageAllocations.push_back(samplers.size() - 1);
        }
        Require(samplers.size() <= context.limits.maxDescriptorSetSamplers, "pipeline sampler descriptors exceed device limits");
    }

    bindings.push_back(std::move(item));
    if (sampledImage) deferredImages.push_back({&binding, bindings.size() - 1});
}

namespace {

// Calls `visit(binding, element, words)` for every sampled and storage image element of a compiled
// shader's bindings, in plan order (the order addImageBinding and resolveImageBinding walk).
template <typename Visit>
void forEachImageElement(const ShaderRecompiler::RecompileResult& program, Visit&& visit) {
    for (const auto& binding : program.bindings) {
        if (binding.role != ShaderRecompiler::DescriptorRole::GuestImages || binding.count == 0) continue;
        if (binding.kind != ShaderRecompiler::DescriptorKind::SampledImage && binding.kind != ShaderRecompiler::DescriptorKind::StorageImage) continue;
        const auto elementWords = binding.guestDescriptor.size() / binding.count;
        for (std::uint32_t element = 0; element < binding.count; ++element) visit(binding, element, std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords));
    }
}

}

bool ShaderResources::precollectImages() {
    // The image lookups of stage B (cachedTexture, StorageTexture::Refresh) each start with a
    // GuestMemory::CollectWrites of their surface, the GetWriteWatch walk that is most of the
    // lookups' time under the lock. The walk is memoized per thread epoch, and the worker thread
    // passes no ordering point between the stages (see GuestMemory::BumpCollectEpoch), so walking
    // every surface here makes stage B's collects memo hits: the walk leaves the lock, the lookups'
    // checks stay where they were. A memo miss (the ring overflowed, an uncommitted page) just walks
    // under the lock as before; a descriptor stage B rejects is left to it. APS5_NO_PRECOLLECT=1
    // disables the pass.
    // The pass also leaves one ImageRecord per element: the decode and the surface description are
    // made once for the build, and a sampled element's cache entry is taken here (a hash lookup
    // under the cache's own mutex) so that stage B, under the device lock, runs only the
    // fastRevalidate predicate on it (see fastTexture). The collect comes before the entry is read,
    // as the predicate's rule demands (the generation the entry moves to must predate the checks).
    static const bool disabled = std::getenv("APS5_NO_PRECOLLECT") != nullptr;
    static const bool noRecords = std::getenv("APS5_NO_STAGE_A_IMAGES") != nullptr;
    if (disabled || deferredImages.empty()) return false;
    imageRecords.clear();
    imageRecords.reserve(plannedSampledImages + plannedStorageImages);
    nextImageRecord = 0;
    auto& counters = TextureCounts();
    for (const auto& deferred : deferredImages) {
        const auto& binding = *deferred.binding;
        const auto elementWords = binding.guestDescriptor.size() / binding.count;
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
            const bool sampled = binding.kind == ShaderRecompiler::DescriptorKind::SampledImage;
            if (sampled && DedupeImages() && !imageRecords.empty()) {
                const auto& previous = imageRecords.back();
                if (previous.sampled && previous.decoded && words.size() == 8 && std::equal(words.begin(), words.end(), previous.words.begin()) && (binding.imageDepthCompare.empty() || !binding.imageDepthCompare.at(element))) {
                    ImageRecord repeated;
                    repeated.sampled = true;
                    repeated.decoded = previous.decoded;
                    repeated.words = previous.words;
                    repeated.resource = previous.resource;
                    repeated.guestBytes = previous.guestBytes;
                    repeated.components = previous.components;
                    imageRecords.push_back(repeated);
                    continue;
                }
            }
            ImageRecord record;
            record.sampled = sampled;
            try {
                const auto* memo = ImageMemoEnabled() && words.size() == 8 ? decodedImage(context.device, words) : nullptr;
                if (memo != nullptr) {
                    record.resource = memo->resource;
                    record.guestBytes = memo->guestBytes;
                } else {
                    record.resource = DecodeTextureResource(words);
                    record.guestBytes = DescribeSurface(record.resource).guestBytes;
                }
                record.generation = GuestMemory::CollectWrites(record.resource.baseAddress, static_cast<std::size_t>(record.guestBytes));
                record.decoded = true;
                if (record.sampled && !noRecords && words.size() == 8 && (binding.imageDepthCompare.empty() || !binding.imageDepthCompare.at(element))) {
                    std::copy(words.begin(), words.end(), record.words.begin());
                    record.components = memo != nullptr ? memo->components : VkComponentMapping{ComponentSwizzleFor(record.resource.dstSelX), ComponentSwizzleFor(record.resource.dstSelY), ComponentSwizzleFor(record.resource.dstSelZ), ComponentSwizzleFor(record.resource.dstSelW)};
                    record.keys = TextureClearKeys(record.resource, record.guestBytes);
                    auto& cache = Textures();
                    std::lock_guard lock(cache.mutex);
                    if (const auto it = findSampledEntry(cache, context.device, record.words, record.components); it != cache.entries.end()) {
                        record.texture = it->texture;
                        record.source = it->source;
                        record.entryKeys = it->keys;
                        record.entryGeneration = it->generation;
                        if (TextureCountersReported()) counters.records.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            } catch (const std::exception&) {
                record.decoded = false;
            }
            imageRecords.push_back(std::move(record));
        }
    }
    return true;
}

std::shared_ptr<Texture> ShaderResources::fastTexture(ImageRecord& record) {
    if (record.texture == nullptr) return nullptr;
    struct Outcome {
        bool profile;
        std::chrono::steady_clock::time_point start;
        bool hit = false;
        ~Outcome() {
            if (profile) LookupOutcomes::Add(hit ? LookupOutcomes::SampledFast : LookupOutcomes::SampledFastMiss, start);
        }
    } outcome{LookupOutcomes::Profiled(), LookupOutcomes::Profiled() ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{}};
    // Exactly the "unchanged" branches of cachedTexture, from write stamps, the pending-results
    // registry and the DCC keys, the way fastRevalidate proves a built object current: the collect
    // first (a memo hit: stage A walked the range), then no other image may have results pending
    // over the memory (a lookup would flush them into it, or view them instead), a storage-sourced
    // view needs its image to be the cache's image of the surface and current with guest memory,
    // a snapshot needs the memory unchanged since its content matched, and the keys must be what
    // the content was made under whenever the surface has any (re-read under the lock: a flush
    // between the stages marks them uncompressed).
    const auto address = record.resource.baseAddress;
    const auto bytes = static_cast<std::size_t>(record.guestBytes);
    if (GuestMemory::CollectWrites(address, bytes) == 0) return nullptr;
    if (PendingStorageOverlaps(address, bytes, record.source.get())) return nullptr;
    auto keys = record.keys;
    if (record.resource.dccAddress != 0) {
        const auto scanStart = outcome.profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        // Through the proof of the image whose keys they are (the view's source when the surface
        // is the image's, else the texture's): a scan only when the key range was stamped.
        keys = ProvedClearKeys(record.resource, record.guestBytes, SameKeySurface(record.source.get(), record.resource, record.guestBytes) ? record.source->KeyProof() : record.texture->KeyProof());
        if (outcome.profile) LookupOutcomes::Add(LookupOutcomes::DccScan, scanStart);
    }
    // The keys the entry was made under (a view of a pending render target was made under its clear
    // keys and stays valid while they are unchanged); a change is the full lookup's to judge.
    if (keys != record.entryKeys) return nullptr;
    if (record.source != nullptr) {
        if (!StorageImageCached(context, record.source.get()) || !GuestMemory::UnchangedSince(address, bytes, record.source->Generation())) return nullptr;
        if (record.resource.dccAddress != 0 && IsDccClear(record.source->FilledKeys())) return nullptr;
        // Under fast-clear keys the view holds only while its image's results are still pending over
        // the surface; flushed, the clear the image cannot see makes the lookup take a snapshot.
        if (keys != DccKeys::Uncompressed && StorageTexture::FindPending(address, record.guestBytes) != record.source) return nullptr;
    } else if (keys == DccKeys::Uncompressed && !GuestMemory::UnchangedSince(address, bytes, record.entryGeneration)) {
        return nullptr;
    }
    // The entry must still be the cache's, holding this object (an evicted object is not reused: the
    // next lookup would make another), and it takes the stage-A generation like a hit would.
    auto& cache = Textures();
    std::lock_guard lock(cache.mutex);
    const auto it = findSampledEntry(cache, context.device, record.words, record.components);
    if (it == cache.entries.end() || it->texture != record.texture) return nullptr;
    if (it->source == nullptr) it->generation = record.generation;
    touchTexture(cache, it);
    logLookup({record.texture.get(), record.resource, record.guestBytes, keys, record.source != nullptr ? 0 : record.generation, record.source.get()});
    outcome.hit = true;
    return std::move(record.texture);
}

void ShaderResources::resolveImageBinding(const ShaderRecompiler::DescriptorBinding& binding, Binding& item) {
    auto& counters = TextureCounts();
    // The element's stage-A record, when the pass ran (records follow the plan order exactly).
    const auto nextRecord = [&]() -> ImageRecord* { return nextImageRecord < imageRecords.size() ? &imageRecords[nextImageRecord++] : nullptr; };
    if (binding.kind == ShaderRecompiler::DescriptorKind::SampledImage) {
        const auto elementWords = binding.guestDescriptor.size() / binding.count;
        for (std::uint32_t element = 0; element < binding.count; ++element) {
            const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * elementWords, elementWords);
            auto* record = nextRecord();
            const auto resource = record != nullptr && record->decoded ? record->resource : DecodeTextureResource(words);
            const bool firstLayer = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DArray;
            if (!firstLayer && !MatchesGuestDimension(*binding.imageShape, resource.dimension)) throw std::runtime_error("AGC graphics: guest texture dimension disagrees with the shader's declared image shape (shape " + std::to_string(static_cast<int>(*binding.imageShape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")");
            const VkComponentMapping components{ComponentSwizzleFor(resource.dstSelX), ComponentSwizzleFor(resource.dstSelY), ComponentSwizzleFor(resource.dstSelZ), ComponentSwizzleFor(resource.dstSelW)};
            const auto guestBytes = record != nullptr && record->decoded ? record->guestBytes : DescribeSurface(resource).guestBytes;
            std::shared_ptr<Texture> texture;
            const bool depthCompare = !binding.imageDepthCompare.empty() && binding.imageDepthCompare.at(element);
            if (DedupeImages() && previousSampled.texture != nullptr && previousSampled.depthCompare == depthCompare && words.size() == 8 && std::equal(words.begin(), words.end(), previousSampled.words.begin())) texture = previousSampled.texture;
            if (texture == nullptr && record != nullptr && record->texture != nullptr) {
                texture = fastTexture(*record);
                if (TextureCountersReported()) (texture != nullptr ? counters.fastHits : counters.fastMisses).fetch_add(1, std::memory_order_relaxed);
            }
            if (texture == nullptr) texture = cachedTexture(context, words, resource, components, guestBytes, depthCompare);
            if (words.size() == 8) {
                std::copy(words.begin(), words.end(), previousSampled.words.begin());
                previousSampled.depthCompare = depthCompare;
                previousSampled.texture = texture;
            }
            textures.push_back(std::move(texture));
            textureFirstLayer.push_back(firstLayer);
            describedRanges.push_back({"texture", resource.baseAddress, guestBytes, resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.dccAddress});
            item.imageAllocations.push_back(textures.size() - 1);
        }
        // The line is due even when every element took the fast path (cachedTexture reports too).
        reportTextureCounters();
        return;
    }
    previousSampled = {};
    // Consecutive identical storage descriptors address successive mips of one texture (dynamic-mip
    // storage writes).
    std::uint32_t mipOffset = 0;
    for (std::uint32_t element = 0; element < binding.count; ++element) {
        const auto words = std::span<const std::uint32_t>(binding.guestDescriptor).subspan(static_cast<std::size_t>(element) * 8u, 8u);
        const bool sameAsPrevious = SameAsPreviousStorageElement(binding, element);
        if (sameAsPrevious) ++mipOffset;
        else mipOffset = 0;
        const auto* record = nextRecord();
        const auto resource = record != nullptr && record->decoded ? record->resource : DecodeTextureResource(words);
        const bool firstLayer = binding.imageShape == ShaderRecompiler::DescriptorImageShape::Image2D && resource.dimension == TextureDimension::k2DArray;
        if (binding.imageShape.has_value() && !firstLayer && !MatchesGuestDimension(*binding.imageShape, resource.dimension)) throw std::runtime_error("AGC graphics: guest storage texture dimension disagrees with the shader's declared image shape (shape " + std::to_string(static_cast<int>(*binding.imageShape)) + ", dimension " + std::to_string(static_cast<int>(resource.dimension)) + ")");
        const auto mip = std::min(resource.baseLevel + mipOffset, resource.mipCount - 1u);
        Require(resource.minLod <= mip * 256u, "guest storage texture descriptor clamps its minimum LOD above the level it addresses, which is not implemented");
        const auto guestBytes = record != nullptr && record->decoded ? record->guestBytes : DescribeSurface(resource).guestBytes;
        // The same surface as the previous element: its image was just looked up and refreshed.
        if (sameAsPrevious && StorageDedupeEnabled()) storageTextures.push_back(storageTextures.back());
        else storageTextures.push_back(cachedStorageTexture(context, words, resource, mip, guestBytes));
        storageMips.push_back(mip);
        storageKeys.push_back(resource.dccAddress);
        storageFirstLayer.push_back(firstLayer);
        // Images the shader only reads have nothing to store back.
        storageWritten.push_back(element >= binding.imageWritten.size() || binding.imageWritten[element]);
        describedRanges.push_back({"storage", resource.baseAddress, guestBytes, resource.width, resource.height, resource.format, static_cast<int>(resource.tileMode), resource.dccAddress});
        item.imageAllocations.push_back(storageTextures.size() - 1);
    }
}

std::vector<std::pair<std::uint64_t, std::uint64_t>> ShaderResources::PresyncSurfaces() const {
    std::vector<std::pair<std::uint64_t, std::uint64_t>> surfaces;
    // A surface's lookup reads guest memory on the CPU unless it is served GPU-direct from a host
    // import: a storage image of imported memory uploads and refreshes from the import, and a
    // sampled texture of one views that image (SampledFromStorageEligible), so neither waits for
    // the producer.
    const auto consider = [&](std::uint32_t format, std::uint64_t address, std::uint64_t guestBytes, bool sampled) {
        const bool gpuDirect = sampled ? SampledFromStorageEligible(context, format, address, guestBytes) : HostImportCovers(context, address, static_cast<std::size_t>(guestBytes));
        if (!gpuDirect) surfaces.emplace_back(address, guestBytes);
    };
    if (completed) {
        // A cached object (the resource cache): Revalidate repeats the lookups of the objects the
        // build made, which never change afterwards (a lookup returning another object fails the
        // revalidation instead), so the surfaces come from the build's own record of them. A
        // sampled texture viewing a storage image refreshes like that image; a snapshot texture
        // over an eligible surface is replaced by a view without reading. Storage images are
        // checked on their own surface (a mip chain can exceed the element's descriptor). Read
        // without the device lock: these members are fixed once the build completed.
        std::size_t textureIndex = 0;
        for (const auto& range : describedRanges) {
            if (std::strcmp(range.kind, "texture") == 0 && textureIndex < textures.size()) {
                const auto& texture = textures[textureIndex++];
                if (texture->ViewsStorageImage()) consider(range.format, range.address, range.bytes, false);
                else consider(range.format, range.address, range.bytes, true);
            }
        }
        for (std::size_t index = 0; index < storageTextures.size(); ++index) {
            if (index != 0 && storageTextures[index] == storageTextures[index - 1]) continue;
            const auto& image = *storageTextures[index];
            consider(image.Descriptor().format, image.Descriptor().baseAddress, image.GuestBytes(), false);
        }
        return surfaces;
    }
    if (!imageRecords.empty()) {
        for (const auto& record : imageRecords) {
            if (record.decoded) consider(record.resource.format, record.resource.baseAddress, record.guestBytes, record.sampled);
        }
        return surfaces;
    }
    // Stage A has not run (an address-based build, or the pass is off): decoded from the shader.
    if (deferredCompute.program == nullptr) return surfaces;
    forEachImageElement(*deferredCompute.program, [&](const ShaderRecompiler::DescriptorBinding& binding, std::uint32_t, std::span<const std::uint32_t> words) {
        try {
            const auto resource = DecodeTextureResource(words);
            consider(resource.format, resource.baseAddress, DescribeSurface(resource).guestBytes, binding.kind == ShaderRecompiler::DescriptorKind::SampledImage);
        } catch (const std::exception&) {
            // Stage B reports the bad descriptor.
        }
    });
    return surfaces;
}

ShaderResources::~ShaderResources() {
    release();
}

void ShaderResources::release() noexcept {
    // A set from the cache's pool chain goes back to it; a dedicated pool dies with its set.
    if (cachePool && context.descriptorCache != nullptr) context.descriptorCache->Free({_set, cachePool, _layout});
    if (pool) context.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(context.device, pool, nullptr);
    if (_layout && ownsLayout) context.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(context.device, _layout, nullptr);
    cachePool = VK_NULL_HANDLE;
    pool = VK_NULL_HANDLE;
    _set = VK_NULL_HANDLE;
    _layout = VK_NULL_HANDLE;
}

VkDescriptorSetLayout ShaderResources::Layout() const {
    return _layout;
}

ShaderResources::OwnedSetPool::~OwnedSetPool() {
    if (cache == nullptr) return;
    for (const auto& set : free) cache->Free(set.allocation);
}

ShaderResources::DrawBindings::~DrawBindings() {
    if (cache != nullptr && allocation.set != VK_NULL_HANDLE) cache->Free(allocation);
}

namespace {
// APS5_PROFILE_DRAW: draw input snapshots copied and reused (Recorder::ReusableDrawSnapshot),
// printed every 10 s as [drawsnap]. Called under GuestMemory::GpuMutex (plain counters).
void CountDrawSnapshot(bool reused, std::size_t bytes) {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if (!profile) return;
    static std::uint64_t copies = 0, copiedBytes = 0, reuses = 0, reusedBytes = 0;
    static auto last = std::chrono::steady_clock::now();
    if (reused) {
        ++reuses;
        reusedBytes += bytes;
    } else {
        ++copies;
        copiedBytes += bytes;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(10)) return;
    last = now;
    const auto cache = Recorder::DrawSnapshotCounts();
    AgcDriver::ReportLine("[drawsnap] draw input snapshots (10 s): copied %llu (%.1f MiB), reused %llu (%.1f MiB); cache (cumulative): %llu lookups found no entry, %llu found a stale one (%llu of them reused with unchanged bytes, %llu with only their changed blocks copied), %llu evicted\n", static_cast<unsigned long long>(copies), copiedBytes / 1048576.0, static_cast<unsigned long long>(reuses), reusedBytes / 1048576.0, static_cast<unsigned long long>(cache.absent), static_cast<unsigned long long>(cache.stale), static_cast<unsigned long long>(cache.revalidated), static_cast<unsigned long long>(cache.patched), static_cast<unsigned long long>(cache.evicted));
    copies = copiedBytes = reuses = reusedBytes = 0;
}
}

namespace {

// APS5_SNAPSHOT_ADDRESS_DRAWS=1 (local experiment): an address-based draw also binds snapshots of
// its read-only buffer elements instead of the imports (a shader's scattered reads of imported
// system memory cross PCIe, a snapshot can sit in video memory), from
// APS5_SNAPSHOT_ADDRESS_DRAWS_MIN_KIB (0: no lower bound; the GPU caches small windows of imported
// memory, which a copy does not beat) up to APS5_SNAPSHOT_ADDRESS_DRAWS_MAX_KIB (8192) each.
// APS5_SNAPSHOT_ADDRESS_DRAWS_SMALL_BYTES (4096; 0: none): an element up to that size is copied
// into the batch arena on its first use in a collect epoch instead of being kept in the snapshot
// cache (per-draw constants at fresh addresses are copied once and never bound again: a
// write-watch walk for a generation nobody uses costs more than their bytes).
struct AddressSnapshotWindow {
    bool enabled;
    std::size_t floor;
    std::size_t limit;
    std::size_t small;
};

const AddressSnapshotWindow& AddressSnapshots() {
    static const AddressSnapshotWindow window = [] {
        const auto kib = [](const char* name, unsigned long long fallback) {
            const char* value = std::getenv(name);
            return static_cast<std::size_t>(value != nullptr ? std::strtoull(value, nullptr, 10) : fallback) << 10u;
        };
        const char* small = std::getenv("APS5_SNAPSHOT_ADDRESS_DRAWS_SMALL_BYTES");
        return AddressSnapshotWindow{std::getenv("APS5_SNAPSHOT_ADDRESS_DRAWS") != nullptr, kib("APS5_SNAPSHOT_ADDRESS_DRAWS_MIN_KIB", 0), kib("APS5_SNAPSHOT_ADDRESS_DRAWS_MAX_KIB", 8192), std::min<std::size_t>(small != nullptr ? static_cast<std::size_t>(std::strtoull(small, nullptr, 10)) : 4096, Recorder::ArenaMaxBytes)};
    }();
    return window;
}

ShaderResources::AddressSnapshotStats addressSnapshotStats;
// The recorder's snapshot stamp for the addressDrawBindings call in progress (addressSnapshot).
thread_local Recorder::SnapshotStamp addressSnapshotStamp;

// The descriptor sets of a batch's address-based draws, freed back to the descriptor cache when
// the batch's kept objects go (Recorder::BatchScratch).
struct DrawBatchSets {
    DescriptorCache* cache = nullptr;
    std::vector<DescriptorCache::SetAllocation> sets;
    // APS5_OWNED_DRAW_SETS: the sets that go back to their template's pool instead.
    std::vector<std::pair<std::shared_ptr<ShaderResources::OwnedSetPool>, ShaderResources::OwnedDrawSet>> owned;
    ~DrawBatchSets() {
        for (auto& [pool, set] : owned) {
            std::lock_guard lock(pool->mutex);
            if (pool->free.size() < 16) {
                try {
                    pool->free.push_back(std::move(set));
                    continue;
                } catch (...) {
                }
            }
            if (pool->cache != nullptr) pool->cache->Free(set.allocation);
        }
        if (cache == nullptr) return;
        for (const auto& allocation : sets) cache->Free(allocation);
    }
};

bool OwnedDrawSets() {
    static const bool enabled = std::getenv("APS5_OWNED_DRAW_SETS") != nullptr;
    return enabled;
}

// APS5_OWNED_DRAW_SETS counters for [bindsplit]: sets taken back from a template's pool, and the
// one-element restores they needed.
std::uint64_t ownedReused = 0;
std::uint64_t ownedRestores = 0;

}

const ShaderResources::AddressSnapshotStats& ShaderResources::AddressSnapshotCounters() {
    const auto& windows = CollectWindowCounters();
    addressSnapshotStats.windowWalks = windows.walks.load(std::memory_order_relaxed);
    addressSnapshotStats.windowServed = windows.served.load(std::memory_order_relaxed);
    addressSnapshotStats.windowRefused = windows.refused.load(std::memory_order_relaxed);
    return addressSnapshotStats;
}

namespace {
// APS5_TRACE_BIND_SPLIT: addressSnapshot's time by step, in nanoseconds, and how its calls ended
// (see [bindsplit]). Plain counters: the callers hold GuestMemory::GpuMutex.
struct SnapSplit {
    std::uint64_t calls = 0, checksNs = 0, epochNs = 0, pendingNs = 0, smallNs = 0, collectNs = 0;
    std::uint64_t refused = 0, epochHits = 0, small = 0, collected = 0, dataNs = 0, dataBuffers = 0;
};
SnapSplit snapSplit;
bool SnapSplitTraced() {
    static const bool traced = std::getenv("APS5_TRACE_BIND_SPLIT") != nullptr;
    return traced;
}
struct SnapLaps {
    bool on = SnapSplitTraced();
    std::chrono::steady_clock::time_point mark = on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    void lap(std::uint64_t SnapSplit::*field) {
        if (!on) return;
        const auto at = std::chrono::steady_clock::now();
        snapSplit.*field += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(at - mark).count());
        mark = at;
    }
    void count(std::uint64_t SnapSplit::*field) {
        if (on) ++(snapSplit.*field);
    }
};
}

bool ShaderResources::addressSnapshot(Recorder& recorder, std::uint64_t begin, std::size_t bytes, VkDescriptorBufferInfo& info, bool& transient) const {
    transient = false;
    auto& stats = addressSnapshotStats;
    const auto& window = AddressSnapshots();
    SnapLaps laps;
    laps.count(&SnapSplit::calls);
    const auto leave = [&](SnapshotRefusal reason) {
        ++stats.inPlace[static_cast<std::size_t>(reason)];
        laps.count(&SnapSplit::refused);
        laps.lap(&SnapSplit::checksNs);
        return false;
    };
    const auto bind = [&](VkBuffer buffer, VkDeviceSize offset, bool video) {
        ++stats.bound;
        if (video) ++stats.video;
        info = {buffer, offset, bytes};
        return true;
    };
    // What depends on this build and on the range alone, whatever the epoch.
    if (bytes < window.floor) return leave(SnapshotRefusal::UnderWindow);
    if (bytes > window.limit) return leave(SnapshotRefusal::OverWindow);
    if (guestMemory.WritesOverlap(begin, bytes)) return leave(SnapshotRefusal::Written);
    std::pair<std::uint64_t, std::uint64_t> region;
    if (!guestMemory.BoundInPlace(begin, bytes, &region)) return leave(SnapshotRefusal::OutsideImport);
    // A range proved in this collect epoch under the stamp still in force: no check is repeated.
    const auto& now = addressSnapshotStamp;
    laps.lap(&SnapSplit::checksNs);
    const auto proved = recorder.EpochSnapshot(begin, bytes, now);
    laps.lap(&SnapSplit::epochNs);
    if (proved.buffer != VK_NULL_HANDLE) {
        laps.count(&SnapSplit::epochHits);
        ++stats.epochSkips;
        if (proved.rechecked) ++stats.epochRechecks;
        transient = proved.transient;
        return bind(proved.buffer, proved.offset, proved.video);
    }
    // What the recorded work still has to write is read in place, behind it. (Pending image
    // results and unit shadows are not asked about: an address-based build binds its elements in
    // place without flushing them either, so a copy of the import's bytes reads what the import
    // would; the snapshot path of the builds never asked.)
    if (recorder.PendingWriteHits(begin, bytes)) return leave(SnapshotRefusal::Pending);
    laps.lap(&SnapSplit::pendingNs);
    // A small element: its bytes into the batch arena, with no collect and no cache entry. Its
    // proof is the copy itself, good for the collect epoch like a memo hit's (a later use in the
    // epoch binds the same bytes, the next epoch copies again); the range lies in an import the
    // lease pins, so its pages are there to read.
    if (bytes <= window.small) {
        if (const auto arena = recorder.ArenaAllocate(bytes); arena.bytes != nullptr) {
            std::memcpy(arena.bytes, reinterpret_cast<const void*>(begin), bytes);
            recorder.NoteEpochSnapshot(begin, bytes, now, 0, arena.block, arena.offset);
            ++stats.small;
            stats.smallBytes += bytes;
            laps.count(&SnapSplit::small);
            laps.lap(&SnapSplit::smallNs);
            transient = true;
            return bind(arena.block->Handle(), arena.offset, arena.block->InVideoMemory());
        }
    }
    // Collected before the lookup and before a copy: the snapshot is then current as of a collect
    // of this epoch, which is what lets the epoch stand for it afterwards (a store made during
    // the copy is stamped newer by the next walk and compared then). A range that is not
    // write-watched has no generation and would be copied for every draw: left in place.
    const GuestMemory::CollectSiteScope collectSite(GuestMemory::CollectSite::DrawSnapshot);
    // With its window (CollectWindowed), clamped to the import the range lies in.
    const auto generation = CollectWindowed(begin, bytes, &region);
    if (generation == 0) return leave(SnapshotRefusal::Unwatched);
    auto buffer = recorder.ReusableDrawSnapshot(begin, bytes, Recorder::SnapshotUse::Storage, nullptr, generation);
    if (buffer != nullptr) {
        ++stats.reused;
    } else {
        if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) return leave(SnapshotRefusal::OutsideImport);
        buffer = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, GpuReadProperties(GpuReadKind::StorageCopy));
        // As PrepareDrawBindings copies: a snapshot in video memory keeps its bytes in system
        // memory too, for the compares.
        std::vector<std::byte> shadow;
        if ((GpuReadProperties(GpuReadKind::StorageCopy) & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
            shadow.assign(reinterpret_cast<const std::byte*>(begin), reinterpret_cast<const std::byte*>(begin) + bytes);
            std::memcpy(buffer->Bytes().data(), shadow.data(), bytes);
        } else {
            std::memcpy(buffer->Bytes().data(), reinterpret_cast<const void*>(begin), bytes);
        }
        recorder.KeepDrawSnapshot(begin, bytes, generation, now.registry, buffer, Recorder::SnapshotUse::Storage, 0, std::move(shadow));
        ++stats.copied;
        stats.copiedBytes += bytes;
    }
    recorder.NoteEpochSnapshot(begin, bytes, now, generation, buffer);
    CaptureTrace::Log("draw-snapshot batch=%llu address=%llx bytes=%zu", static_cast<unsigned long long>(recorder.Submissions() + 1), static_cast<unsigned long long>(begin), bytes);
    laps.count(&SnapSplit::collected);
    laps.lap(&SnapSplit::collectNs);
    return bind(buffer->Handle(), 0, buffer->InVideoMemory());
}

namespace {
// APS5_TRACE_BIND_SPLIT=1 (local, not for upstream): where addressDrawBindings spends its time,
// printed as [bindsplit] every 10 s: the gather (snapshots, data buffers), the memo, the set's
// allocation, the copy of the template's set and the writes of the selected elements, with what
// a set holds (bindings, descriptors by type, writes). Plain counters: the callers hold
// GuestMemory::GpuMutex.
struct BindSplit {
    std::uint64_t calls = 0, built = 0, gatherNs = 0, memoNs = 0, allocNs = 0, copyNs = 0, writeNs = 0;
    std::uint64_t bindings = 0, descriptors = 0, writes = 0, selected = 0;
    std::array<std::uint64_t, 5> byType{};
    std::uint64_t maxDescriptors = 0, overPush = 0;
    std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};
bool BindSplitTraced() {
    static const bool traced = std::getenv("APS5_TRACE_BIND_SPLIT") != nullptr;
    return traced;
}
BindSplit& TheBindSplit() {
    static BindSplit split;
    return split;
}
std::uint64_t SplitNs(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(to - from).count());
}
void ReportBindSplit() {
    auto& split = TheBindSplit();
    const auto now = std::chrono::steady_clock::now();
    if (now - split.last < std::chrono::seconds(10)) return;
    const auto per = [&](std::uint64_t ns, std::uint64_t n) { return n != 0 ? static_cast<double>(ns) / 1000.0 / static_cast<double>(n) : 0.0; };
    const auto avg = [&](std::uint64_t value) { return split.built != 0 ? static_cast<double>(value) / static_cast<double>(split.built) : 0.0; };
    AgcDriver::ReportLine("[bindsplit] address draw sets (10 s): %llu calls, %llu sets built; avg us per call: gather %.2f, memo %.2f; per set built: allocate %.2f, copy of the template set %.2f, writes %.2f; per set built: %.1f bindings, %.1f descriptors (storage buffers %.1f, sampled images %.1f, storage images %.1f, samplers %.1f, other %.1f), %.1f selected, %.1f writes; largest set %llu descriptors, %llu sets over 32; owned sets: %llu taken back, %llu one-element restores\n", static_cast<unsigned long long>(split.calls), static_cast<unsigned long long>(split.built), per(split.gatherNs, split.calls), per(split.memoNs, split.calls), per(split.allocNs, split.built), per(split.copyNs, split.built), per(split.writeNs, split.built), avg(split.bindings), avg(split.descriptors), avg(split.byType[0]), avg(split.byType[1]), avg(split.byType[2]), avg(split.byType[3]), avg(split.byType[4]), avg(split.selected), avg(split.writes), static_cast<unsigned long long>(split.maxDescriptors), static_cast<unsigned long long>(split.overPush), static_cast<unsigned long long>(ownedReused), static_cast<unsigned long long>(ownedRestores));
    const auto& snap = snapSplit;
    const auto perSnap = [&](std::uint64_t ns) { return snap.calls != 0 ? static_cast<double>(ns) / 1000.0 / static_cast<double>(snap.calls) : 0.0; };
    AgcDriver::ReportLine("[bindsplit] snapshots of address draws (10 s): %llu elements (%.1f per call), avg us per element: checks %.3f, epoch lookup %.3f, pending writes %.3f, small copy %.3f, collect and cache %.3f; ended: %llu left in place, %llu epoch hits, %llu small copies, %llu collected; data buffers %llu, %.3f us each\n", static_cast<unsigned long long>(snap.calls), split.calls != 0 ? static_cast<double>(snap.calls) / static_cast<double>(split.calls) : 0.0, perSnap(snap.checksNs), perSnap(snap.epochNs), perSnap(snap.pendingNs), perSnap(snap.smallNs), perSnap(snap.collectNs), static_cast<unsigned long long>(snap.refused), static_cast<unsigned long long>(snap.epochHits), static_cast<unsigned long long>(snap.small), static_cast<unsigned long long>(snap.collected), static_cast<unsigned long long>(snap.dataBuffers), snap.dataBuffers != 0 ? static_cast<double>(snap.dataNs) / 1000.0 / static_cast<double>(snap.dataBuffers) : 0.0);
    snapSplit = {};
    ownedReused = ownedRestores = 0;
    split = {};
    split.last = now;
}
}

std::shared_ptr<ShaderResources::DrawBindings> ShaderResources::addressDrawBindings(Recorder& recorder, std::span<const MovedBuffer> moved) const {
    const auto& window = AddressSnapshots();
    if (!window.enabled && moved.empty()) return {};
    const bool splitTraced = BindSplitTraced();
    const auto splitStart = splitTraced ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    auto splitMark = splitStart;
    const auto splitLap = [&](std::uint64_t BindSplit::*field) {
        if (!splitTraced) return;
        const auto at = std::chrono::steady_clock::now();
        TheBindSplit().*field += SplitNs(splitMark, at);
        splitMark = at;
    };
    if (splitTraced) {
        ++TheBindSplit().calls;
        ReportBindSplit();
    }
    // The call's result and scratch, kept per thread: nearly every address-based draw gets here.
    thread_local DrawBindings result;
    thread_local std::vector<std::size_t> selected;
    // What each selected element binds, in `selected`'s order.
    thread_local std::vector<VkDescriptorBufferInfo> infos;
    thread_local std::vector<VkCopyDescriptorSet> copies;
    thread_local std::vector<VkWriteDescriptorSet> writes;
    result.pushPatches.clear();
    result.allocation = {};
    selected.clear();
    infos.clear();
    // One stamp for the call: nothing below waits for recorded work or notes a write.
    addressSnapshotStamp = recorder.CurrentSnapshotStamp(guestMemory.ReadSetToken());
    // What the set will bind, hashed as it is gathered: everything (`full`), and everything but
    // the batch arena's copies, which a later draw never binds again at the same place (`stable`).
    std::uint64_t full = 14695981039346656037ull;
    std::uint64_t stable = 14695981039346656037ull;
    const auto mixInto = [](std::uint64_t& hash, std::uint64_t value) { hash = (hash ^ value) * 1099511628211ull; };
    const auto mixBinding = [&](std::size_t index, const VkDescriptorBufferInfo& info, bool arena) {
        mixInto(full, index);
        mixInto(full, reinterpret_cast<std::uint64_t>(info.buffer));
        mixInto(full, info.offset);
        mixInto(full, info.range);
        mixInto(stable, index);
        if (arena) {
            mixInto(stable, info.range);
            return;
        }
        mixInto(stable, reinterpret_cast<std::uint64_t>(info.buffer));
        mixInto(stable, info.offset);
        mixInto(stable, info.range);
    };
    // APS5_EPOCH_MEMO (see ElementMemo): the element's last binding under this very stamp and batch.
    static const bool epochMemo = std::getenv("APS5_EPOCH_MEMO") != nullptr;
    const std::array<std::uint64_t, 5> memoStamp{addressSnapshotStamp.epoch, addressSnapshotStamp.driverStores, addressSnapshotStamp.writeNotes, addressSnapshotStamp.registry, addressSnapshotStamp.space};
    const std::uint64_t memoBatch = epochMemo ? recorder.OpenBatchId() : 0;
    if (epochMemo && elementMemos.size() != allocations.size()) elementMemos.assign(allocations.size(), {});
    const auto snapshotOf = [&](std::size_t index, std::uint64_t begin, std::size_t bytes, VkDescriptorBufferInfo& info, bool& transient) {
        if (!epochMemo || memoStamp[0] == 0 || memoStamp[4] == 0) return addressSnapshot(recorder, begin, bytes, info, transient);
        auto& memo = elementMemos[index];
        if (memo.buffer != VK_NULL_HANDLE && memo.address == begin && memo.bytes == bytes && memo.batch == memoBatch && memo.stamp == memoStamp) {
            ++addressSnapshotStats.bound;
            ++addressSnapshotStats.epochSkips;
            info = {memo.buffer, memo.offset, bytes};
            transient = memo.transient;
            return true;
        }
        if (!addressSnapshot(recorder, begin, bytes, info, transient)) {
            memo.buffer = VK_NULL_HANDLE;
            return false;
        }
        memo = {begin, bytes, memoStamp, memoBatch, info.buffer, info.offset, transient};
        return true;
    };
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        const auto& item = allocations[index];
        const auto override = std::find_if(moved.begin(), moved.end(), [&](const MovedBuffer& entry) { return entry.allocation == index; });
        if (override != moved.end() && !override->words.empty()) {
            // The draw's own data buffer, from the arena (a Buffer of its own when it is too large).
            SnapLaps dataLaps;
            dataLaps.count(&SnapSplit::dataBuffers);
            VkDescriptorBufferInfo info{};
            std::byte* bytes = nullptr;
            if (const auto arena = recorder.ArenaAllocate(override->size); arena.bytes != nullptr) {
                info = {arena.block->Handle(), arena.offset, override->size};
                bytes = arena.bytes;
            } else {
                auto buffer = std::make_shared<Buffer>(context, override->size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, GpuReadProperties(GpuReadKind::StorageCopy));
                info = {buffer->Handle(), 0, buffer->Bytes().size()};
                bytes = buffer->Bytes().data();
                recorder.Keep(std::move(buffer));
            }
            std::memcpy(bytes, override->words.data(), override->size);
            for (const auto& patch : dataPatches) {
                if (patch.allocation == index && patch.byte < override->size) bytes[patch.byte] = static_cast<std::byte>(patch.adjustment);
            }
            // The adjustments of the moved elements that this buffer carries, over the build's:
            // zero leaves the byte the words hold, as the build leaves it.
            for (const auto& entry : moved) {
                if (!entry.inPlace) continue;
                const auto& element = allocations[entry.allocation];
                if (element.pushByte >= 0 || element.dataAllocation != static_cast<std::int64_t>(index) || element.dataByte >= override->size) continue;
                bytes[element.dataByte] = entry.adjustment != 0 ? static_cast<std::byte>(entry.adjustment) : reinterpret_cast<const std::byte*>(override->words.data())[element.dataByte];
            }
            selected.push_back(index);
            infos.push_back(info);
            mixBinding(index, info, true);
            dataLaps.lap(&SnapSplit::dataNs);
            continue;
        }
        if (override != moved.end() && override->inPlace) {
            if (override->adjustment != item.adjustment && item.pushByte >= 0) result.pushPatches.emplace_back(static_cast<std::uint32_t>(item.pushByte), override->adjustment);
            // A snapshot when the range qualifies, from the adjustment the import gives the
            // range (the one the shader is told), so every template snapshots a buffer alike;
            // else in place, as MovedReadOnlyBuffers found it.
            auto info = override->info;
            bool transient = false;
            if (window.enabled) snapshotOf(index, override->address - override->adjustment, override->size + override->adjustment, info, transient);
            selected.push_back(index);
            infos.push_back(info);
            mixBinding(index, info, transient);
            // The adjustment the shader is told goes with the binding (two ranges can share an
            // aligned offset and a length).
            mixInto(full, override->adjustment);
            mixInto(stable, override->adjustment);
            continue;
        }
        if (!window.enabled || !item.guest) continue;
        if (item.written) {
            ++addressSnapshotStats.inPlace[static_cast<std::size_t>(SnapshotRefusal::Written)];
            continue;
        }
        VkDescriptorBufferInfo info{};
        bool transient = false;
        if (!snapshotOf(index, item.address - item.adjustment, item.size + item.adjustment, info, transient)) continue;
        selected.push_back(index);
        infos.push_back(info);
        mixBinding(index, info, transient);
    }
    splitLap(&BindSplit::gatherNs);
    if (selected.empty()) return {};
    // An identical draw of this batch (the same bindings, hence the same bytes: the batch keeps
    // every buffer, and a set made for it is not written again) binds that draw's set. One of
    // another batch would need a set that outlives its batch; counted, with the draws that
    // differ from a recent one by their arena copies alone (see AddressSnapshotStats).
    const auto batch = recorder.OpenBatchId();
    bool repeatedAcross = false;
    bool repeatedButArena = false;
    for (const auto& memo : drawSetMemos) {
        if (memo.set == VK_NULL_HANDLE) continue;
        if (memo.full == full) {
            if (memo.batch == batch) {
                ++addressSnapshotStats.setsReused;
                result.allocation.set = memo.set;
                return std::shared_ptr<DrawBindings>(std::shared_ptr<void>(), &result);
            }
            repeatedAcross = true;
        } else if (memo.stable == stable) {
            repeatedButArena = true;
        }
    }
    splitLap(&BindSplit::memoNs);
    ++addressSnapshotStats.setsBuilt;
    if (repeatedAcross) ++addressSnapshotStats.setsRepeatedAcrossBatches;
    else if (repeatedButArena) ++addressSnapshotStats.setsRepeatedButArena;
    Require(context.descriptorCache != nullptr, "draw snapshots require a descriptor cache");
    if (drawBindingSizes.empty()) {
        std::map<VkDescriptorType, std::uint32_t> counts;
        for (const auto& binding : bindings) counts[binding.layout.descriptorType] += binding.layout.descriptorCount;
        for (const auto& [type, count] : counts) drawBindingSizes.push_back({type, count});
    }
    const bool owning = OwnedDrawSets();
    OwnedDrawSet owned;
    bool reusedOwned = false;
    if (owning) {
        if (ownedSets == nullptr) {
            ownedSets = std::make_shared<OwnedSetPool>();
            ownedSets->cache = context.descriptorCache;
        }
        std::lock_guard lock(ownedSets->mutex);
        if (!ownedSets->free.empty()) {
            owned = std::move(ownedSets->free.back());
            ownedSets->free.pop_back();
            reusedOwned = true;
        }
    }
    if (!reusedOwned) owned.allocation = context.descriptorCache->Allocate(_layout, drawBindingSizes);
    const auto allocation = owned.allocation;
    Require(allocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
    // The batch frees the set with its kept objects (or gives it back to the template's pool).
    auto& scratch = recorder.BatchScratch();
    if (scratch == nullptr) {
        auto sets = std::make_shared<DrawBatchSets>();
        sets->cache = context.descriptorCache;
        scratch = sets;
        recorder.Keep(std::move(sets));
    }
    auto& batchSets = *static_cast<DrawBatchSets*>(scratch.get());
    if (!owning) batchSets.sets.push_back(allocation);
    splitLap(&BindSplit::allocNs);
    copies.clear();
    if (!reusedOwned) {
        for (const auto& binding : bindings) {
            VkCopyDescriptorSet copy{VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET};
            copy.srcSet = _set;
            copy.srcBinding = binding.layout.binding;
            copy.dstSet = allocation.set;
            copy.dstBinding = binding.layout.binding;
            copy.descriptorCount = binding.layout.descriptorCount;
            copies.push_back(copy);
        }
    }
    writes.clear();
    for (const auto& binding : bindings) {
        for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
            const auto found = std::find(selected.begin(), selected.end(), binding.allocations[element]);
            if (found == selected.end()) continue;
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = allocation.set;
            write.dstBinding = binding.layout.binding;
            write.dstArrayElement = static_cast<std::uint32_t>(element);
            write.descriptorCount = 1;
            write.descriptorType = binding.layout.descriptorType;
            write.pBufferInfo = &infos[static_cast<std::size_t>(found - selected.begin())];
            writes.push_back(write);
        }
    }
    if (owning) {
        // A reused set: the elements its last draw wrote that this one leaves go back to the
        // template's descriptors. Then the set's deviations are this draw's writes.
        for (const auto& [binding, element] : owned.deviations) {
            const bool rewritten = std::any_of(writes.begin(), writes.end(), [&](const VkWriteDescriptorSet& write) { return write.dstBinding == binding && write.dstArrayElement == element; });
            if (rewritten) continue;
            VkCopyDescriptorSet copy{VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET};
            copy.srcSet = _set;
            copy.srcBinding = binding;
            copy.srcArrayElement = element;
            copy.dstSet = allocation.set;
            copy.dstBinding = binding;
            copy.dstArrayElement = element;
            copy.descriptorCount = 1;
            copies.push_back(copy);
            ++ownedRestores;
        }
        if (reusedOwned) ++ownedReused;
        owned.deviations.clear();
        for (const auto& write : writes) owned.deviations.emplace_back(write.dstBinding, write.dstArrayElement);
        batchSets.owned.emplace_back(ownedSets, std::move(owned));
    }
    const auto update = context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets");
    if (splitTraced) splitMark = std::chrono::steady_clock::now();
    if (!copies.empty()) update(context.device, 0, nullptr, static_cast<std::uint32_t>(copies.size()), copies.data());
    splitLap(&BindSplit::copyNs);
    update(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    splitLap(&BindSplit::writeNs);
    if (splitTraced) {
        auto& split = TheBindSplit();
        ++split.built;
        split.bindings += bindings.size();
        split.writes += writes.size();
        split.selected += selected.size();
        std::uint64_t total = 0;
        for (const auto& binding : bindings) {
            const auto type = binding.layout.descriptorType;
            const std::size_t slot = type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ? 0 : type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ? 1 : type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? 2 : type == VK_DESCRIPTOR_TYPE_SAMPLER ? 3 : 4;
            split.byType[slot] += binding.layout.descriptorCount;
            total += binding.layout.descriptorCount;
        }
        split.descriptors += total;
        split.maxDescriptors = std::max(split.maxDescriptors, total);
        if (total > 32) ++split.overPush;
    }
    drawSetMemos[drawSetNext++ % drawSetMemos.size()] = {full, stable, allocation.set, batch};
    // The set only: `cache` stays null, so the scratch frees nothing.
    result.allocation.set = allocation.set;
    return std::shared_ptr<DrawBindings>(std::shared_ptr<void>(), &result);
}

std::optional<std::vector<ShaderResources::MovedBuffer>> ShaderResources::MovedReadOnlyBuffers(std::span<const CompiledShader> shaders, Recorder& recorder) const {
    std::vector<MovedBuffer> moved;
    if (!MovedReadOnlyBuffers(shaders, recorder, moved)) return std::nullopt;
    return moved;
}

bool ShaderResources::MovedReadOnlyBuffers(std::span<const CompiledShader> shaders, Recorder& recorder, std::vector<MovedBuffer>& moved) const {
    moved.clear();
    // An address-based build that is not shared serves no second use.
    if (_set == VK_NULL_HANDLE || (usesBda && !guestMemory.Shared())) return true;
    // The build made one binding per stage binding, in this order: the template's is found by
    // position, and searched for only if the stages are not the build's layout after all.
    std::size_t position = 0;
    const auto keptBinding = [&](const ShaderRecompiler::DescriptorBinding& binding) {
        const auto at = position - 1;
        if (at < bindings.size() && bindings[at].layout.binding == binding.binding) return bindings.begin() + static_cast<std::ptrdiff_t>(at);
        return std::find_if(bindings.begin(), bindings.end(), [&](const Binding& item) { return item.layout.binding == binding.binding; });
    };
    for (const auto& shader : shaders) {
        if (shader.program == nullptr) return false;
        for (const auto& binding : shader.program->bindings) {
            ++position;
            if (DataRole(binding.role)) {
                const auto kept = keptBinding(binding);
                if (kept == bindings.end() || kept->allocations.size() != 1) return false;
                const auto index = kept->allocations.front();
                const auto& item = allocations[index];
                if (item.guest || item.buffer == nullptr || item.size != binding.guestDescriptor.size() * sizeof(std::uint32_t)) return false;
                const bool patched = std::any_of(dataPatches.begin(), dataPatches.end(), [&](const DataPatch& patch) { return patch.allocation == index; });
                const bool same = !item.dataWords.empty() ? item.dataWords == binding.guestDescriptor : !patched && std::memcmp(item.buffer->Bytes().data(), binding.guestDescriptor.data(), item.size) == 0;
                if (same) continue;
                if (item.dataWords.empty() && patched) return false;
                moved.push_back({index, 0, item.size, binding.guestDescriptor});
                continue;
            }
            if (binding.role != ShaderRecompiler::DescriptorRole::GuestBuffers) continue;
            const auto kept = keptBinding(binding);
            if (kept == bindings.end() || kept->allocations.size() != binding.count || binding.guestDescriptor.size() != static_cast<std::size_t>(binding.count) * 4u) return false;
            for (std::uint32_t element = 0; element < binding.count; ++element) {
                const auto* words = binding.guestDescriptor.data() + static_cast<std::size_t>(element) * 4u;
                const ShaderRecompiler::ShaderBufferResource descriptor{{words[0], words[1], words[2], words[3]}};
                const auto address = descriptor.Base48();
                const auto size = descriptor.GetSize();
                const auto index = kept->allocations[element];
                const auto& item = allocations[index];
                const bool empty = size == 0 || address == 0;
                if (!item.guest) {
                    if (empty) continue;
                    return false;
                }
                if (item.address == address && item.size == size) continue;
                const bool written = element >= binding.bufferWritten.size() || binding.bufferWritten[element];
                if (written || empty || item.written || size > context.limits.maxStorageBufferRange) return false;
                if (guestMemory.Shared()) {
                    // Bound in place through the address space, as the build binds its own
                    // elements (no copy: the lease pins the range and the pass tracking treats
                    // an address-based draw's reads as unknown). An adjustment other than the
                    // build's needs somewhere to go, as a build's nonzero one does.
                    std::uint32_t adjustment = 0;
                    const auto info = guestMemory.SharedDescriptor(address, static_cast<std::size_t>(size), adjustment);
                    if (!info.has_value()) return false;
                    if (adjustment != item.adjustment && item.pushByte < 0 && item.dataAllocation < 0) return false;
                    moved.push_back({index, address, static_cast<std::size_t>(size), {}, true, adjustment, *info});
                    continue;
                }
                const auto begin = address - item.adjustment;
                const auto bytes = static_cast<std::size_t>(size) + item.adjustment;
                if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) return false;
                if (guestMemory.WritesOverlap(begin, bytes) || recorder.PendingWriteOverlaps(begin, bytes) || PendingStorageOverlaps(begin, bytes, nullptr) || AnyShadowedOverlaps(begin, bytes)) return false;
                moved.push_back({index, address, static_cast<std::size_t>(size)});
            }
        }
    }
    // An element bound in place at another adjustment whose shader reads it from its data buffer:
    // the use needs a data buffer of its own to carry it, built from the template's words when
    // this use's are the same.
    for (std::size_t at = 0, count = moved.size(); at < count; ++at) {
        if (!moved[at].inPlace) continue;
        const auto& item = allocations[moved[at].allocation];
        if (moved[at].adjustment == item.adjustment || item.pushByte >= 0) continue;
        const auto data = static_cast<std::size_t>(item.dataAllocation);
        if (std::any_of(moved.begin(), moved.end(), [&](const MovedBuffer& entry) { return entry.allocation == data && !entry.words.empty(); })) continue;
        const auto& source = allocations[data];
        if (source.dataWords.empty() || item.dataByte >= source.size) return false;
        moved.push_back({data, 0, source.size, source.dataWords});
    }
    return true;
}

std::shared_ptr<ShaderResources::DrawBindings> ShaderResources::PrepareDrawBindings(Recorder& recorder, std::span<const MovedBuffer> moved) const {
    // APS5_SNAPSHOT_ADDRESS_DRAWS (see AddressSnapshots): an address-based draw also binds
    // snapshots of its read-only buffer elements instead of the imports.
    const bool addressDraws = AddressSnapshots().enabled;
    const auto addressDrawLimit = AddressSnapshots().limit;
    const auto addressDrawFloor = AddressSnapshots().floor;
    if (_set == VK_NULL_HANDLE) return {};
    // APS5_REUSE_ADDRESS_DRAWS: an address-based build's draws (a shared template's uses with
    // what they moved, and every build's read-only elements under the snapshot switch) are bound
    // by addressDrawBindings.
    if (usesBda && ReuseAddressDraws()) return addressDrawBindings(recorder, moved);
    if (usesBda && !addressDraws) return {};
    const auto reads = guestMemory.InPlaceReads();
    auto result = std::make_shared<DrawBindings>();
    // Scratch of the call, kept per thread.
    thread_local std::vector<std::size_t> selected;
    // What each selected element binds, in `selected`'s order.
    thread_local std::vector<VkDescriptorBufferInfo> infos;
    selected.clear();
    infos.clear();
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        const auto& item = allocations[index];
        const auto override = std::find_if(moved.begin(), moved.end(), [&](const MovedBuffer& entry) { return entry.allocation == index; });
        if (override != moved.end() && !override->words.empty()) {
            auto buffer = std::make_shared<Buffer>(context, override->size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, GpuReadProperties(GpuReadKind::StorageCopy));
            std::memcpy(buffer->Bytes().data(), override->words.data(), override->size);
            for (const auto& patch : dataPatches) {
                if (patch.allocation == index && patch.byte < override->size) buffer->Bytes()[patch.byte] = static_cast<std::byte>(patch.adjustment);
            }
            selected.push_back(index);
            infos.push_back({buffer->Handle(), 0, buffer->Bytes().size()});
            result->snapshots.push_back({0, std::move(buffer)});
            continue;
        }
        std::uint64_t address = item.address;
        std::size_t size = item.size;
        if (override != moved.end()) {
            address = override->address;
            size = override->size;
        } else {
            if (!item.guest || item.written || guestMemory.WritesOverlap(item.address, item.size)) continue;
            const bool direct = std::any_of(reads.begin(), reads.end(), [&](const auto& range) { return item.address >= range.first && item.address < range.second && item.size <= range.second - item.address; });
            if (!direct || recorder.PendingWriteOverlaps(item.address, item.size)) continue;
        }
        const auto begin = address - item.adjustment;
        const auto bytes = size + item.adjustment;
        if (usesBda && (bytes > addressDrawLimit || bytes < addressDrawFloor)) continue;
        // The range's CPU stores so far are stamped first, so the reuse check sees them; a store
        // made after the collect (during the copy) is stamped newer by the next one and drops the
        // snapshot then.
        const auto registryGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
        std::uint64_t generation = 0;
        auto buffer = recorder.CollectedDrawSnapshot(begin, bytes, generation);
        if (buffer != nullptr) {
            CountDrawSnapshot(true, bytes);
        } else {
            buffer = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, GpuReadProperties(GpuReadKind::StorageCopy));
            // A snapshot in video memory is compared with the guest's bytes through a copy in
            // system memory, made first so that both hold the same bytes.
            std::vector<std::byte> shadow;
            if ((GpuReadProperties(GpuReadKind::StorageCopy) & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
                shadow.assign(reinterpret_cast<const std::byte*>(begin), reinterpret_cast<const std::byte*>(begin) + bytes);
                std::memcpy(buffer->Bytes().data(), shadow.data(), bytes);
            } else {
                std::memcpy(buffer->Bytes().data(), reinterpret_cast<const void*>(begin), bytes);
            }
            recorder.KeepDrawSnapshot(begin, bytes, generation, registryGeneration, buffer, Recorder::SnapshotUse::Storage, 0, std::move(shadow));
            CountDrawSnapshot(false, bytes);
        }
        selected.push_back(index);
        infos.push_back({buffer->Handle(), 0, buffer->Bytes().size()});
        result->snapshots.push_back({begin, std::move(buffer)});
        CaptureTrace::Log("draw-snapshot batch=%llu address=%llx bytes=%zu", static_cast<unsigned long long>(recorder.Submissions() + 1), static_cast<unsigned long long>(begin), bytes);
    }
    if (selected.empty()) return {};
    Require(context.descriptorCache != nullptr, "draw snapshots require a descriptor cache");
    if (drawBindingSizes.empty()) {
        std::map<VkDescriptorType, std::uint32_t> counts;
        for (const auto& binding : bindings) counts[binding.layout.descriptorType] += binding.layout.descriptorCount;
        for (const auto& [type, count] : counts) drawBindingSizes.push_back({type, count});
    }
    result->cache = context.descriptorCache;
    result->allocation = result->cache->Allocate(_layout, drawBindingSizes);
    Require(result->allocation.set != VK_NULL_HANDLE, "draw snapshot descriptor allocation failed");
    thread_local std::vector<VkCopyDescriptorSet> copies;
    copies.clear();
    for (const auto& binding : bindings) {
        VkCopyDescriptorSet copy{VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET};
        copy.srcSet = _set;
        copy.srcBinding = binding.layout.binding;
        copy.dstSet = result->allocation.set;
        copy.dstBinding = binding.layout.binding;
        copy.descriptorCount = binding.layout.descriptorCount;
        copies.push_back(copy);
    }
    const auto update = context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets");
    update(context.device, 0, nullptr, static_cast<std::uint32_t>(copies.size()), copies.data());
    thread_local std::vector<VkWriteDescriptorSet> writes;
    writes.clear();
    for (const auto& binding : bindings) {
        for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
            const auto found = std::find(selected.begin(), selected.end(), binding.allocations[element]);
            if (found == selected.end()) continue;
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = result->allocation.set;
            write.dstBinding = binding.layout.binding;
            write.dstArrayElement = static_cast<std::uint32_t>(element);
            write.descriptorCount = 1;
            write.descriptorType = binding.layout.descriptorType;
            write.pBufferInfo = &infos[static_cast<std::size_t>(found - selected.begin())];
            writes.push_back(write);
        }
    }
    update(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    recorder.Keep(result);
    return result;
}

namespace {

// APS5_DISPATCH_SNAPSHOTS (local experiment, not for upstream): a recorded dispatch binds copies of
// its read-only guest buffer elements instead of the imports they are bound in place in (a
// scattered shader read of imported system memory costs ~5.7 ns on the RTX 5080 against ~0.1 ns
// from video memory). =1: the snapshot cache draws use (Recorder::ReusableDrawSnapshot): a range
// with no GPU write or image result pending is copied by the CPU into mappable video memory once
// and reused while the write watch proves it unchanged, changed 64 KiB blocks being patched in.
// =2: also the ranges recorded work still writes (the previous dispatch's output), which no CPU
// copy can serve: a vkCmdCopyBuffer out of the import into a device-local buffer, recorded right
// before the dispatch, per use. From APS5_DISPATCH_SNAPSHOTS_MIN_KIB (0) up to
// APS5_DISPATCH_SNAPSHOTS_MAX_KIB (16384) per element. A build that stores by address
// (GPU-selected V#s) takes none: its stores could land in a range it also reads.
struct DispatchSnapshotWindow {
    int mode;
    std::size_t floor;
    std::size_t limit;
};

const DispatchSnapshotWindow& DispatchSnapshots() {
    static const DispatchSnapshotWindow window = [] {
        const auto kib = [](const char* name, unsigned long long fallback) {
            const char* value = std::getenv(name);
            return static_cast<std::size_t>(value != nullptr ? std::strtoull(value, nullptr, 10) : fallback) << 10u;
        };
        const char* mode = std::getenv("APS5_DISPATCH_SNAPSHOTS");
        return DispatchSnapshotWindow{mode == nullptr ? 0 : mode[0] == '2' ? 2 : 1, kib("APS5_DISPATCH_SNAPSHOTS_MIN_KIB", 0), kib("APS5_DISPATCH_SNAPSHOTS_MAX_KIB", 16384)};
    }();
    return window;
}

// The [dispatch-snap] line, every 10 s. Under GuestMemory::GpuMutex (plain counters).
enum class DispatchRefusal : std::size_t { UnderWindow, OverWindow, PendingWrite, PendingImage, OutsideImport, Unwatched, AddressStores, NoBuffer, Count };

struct DispatchSnapshotStats {
    std::uint64_t dispatches = 0;
    std::uint64_t served = 0;
    std::uint64_t bound = 0;
    std::uint64_t video = 0;
    std::uint64_t reused = 0;
    std::uint64_t cpuCopies = 0;
    std::uint64_t cpuBytes = 0;
    std::uint64_t gpuCopies = 0;
    std::uint64_t gpuBytes = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DispatchRefusal::Count)> inPlace{};
};

void reportDispatchSnapshots(DispatchSnapshotStats& stats) {
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(10)) return;
    const auto seconds = std::chrono::duration<double>(now - last).count();
    last = now;
    const auto count = [](std::uint64_t value) { return static_cast<unsigned long long>(value); };
    const auto left = [&](DispatchRefusal reason) { return count(stats.inPlace[static_cast<std::size_t>(reason)]); };
    AgcDriver::ReportLine("[dispatch-snap] %.1f s: %llu dispatches, %llu bound a set with copies; read-only elements bound to a copy: %llu (%llu in video memory): %llu reused, %llu copied by the CPU (%.1f MiB), %llu copied by the GPU (%.1f MiB); left in place: %llu under the size window, %llu over it, %llu GPU write pending, %llu image result pending, %llu not in an import bound in place, %llu not write-watched, %llu in a build that stores by address, %llu without a buffer\n", seconds, count(stats.dispatches), count(stats.served), count(stats.bound), count(stats.video), count(stats.reused), count(stats.cpuCopies), stats.cpuBytes / 1048576.0, count(stats.gpuCopies), stats.gpuBytes / 1048576.0, left(DispatchRefusal::UnderWindow), left(DispatchRefusal::OverWindow), left(DispatchRefusal::PendingWrite), left(DispatchRefusal::PendingImage), left(DispatchRefusal::OutsideImport), left(DispatchRefusal::Unwatched), left(DispatchRefusal::AddressStores), left(DispatchRefusal::NoBuffer));
    stats = {};
}

}

bool ShaderResources::DispatchSnapshotsEnabled() {
    return DispatchSnapshots().mode != 0;
}

std::shared_ptr<ShaderResources::DrawBindings> ShaderResources::PrepareDispatchBindings(Recorder& recorder, VkCommandBuffer commands, bool& copiedOnGpu) const {
    copiedOnGpu = false;
    const auto& window = DispatchSnapshots();
    if (window.mode == 0 || _set == VK_NULL_HANDLE) return {};
    static DispatchSnapshotStats stats;
    ++stats.dispatches;
    const auto leave = [&](DispatchRefusal reason) { ++stats.inPlace[static_cast<std::size_t>(reason)]; };
    auto result = std::make_shared<DrawBindings>();
    thread_local std::vector<std::size_t> selected;
    thread_local std::vector<VkDescriptorBufferInfo> infos;
    // The import ranges the GPU copies: source, then the snapshot's index in `result->snapshots`.
    thread_local std::vector<std::pair<VkDescriptorBufferInfo, std::size_t>> gpuCopies;
    selected.clear();
    infos.clear();
    gpuCopies.clear();
    const auto registryGeneration = GuestAllocations::GuestAllocationsGeneration_nid_postfix();
    for (std::size_t index = 0; index < allocations.size(); ++index) {
        const auto& item = allocations[index];
        if (!item.guest || item.written) continue;
        const auto begin = item.address - item.adjustment;
        const auto bytes = item.size + item.adjustment;
        if (guestMemory.WritesOverlap(begin, bytes)) continue;
        if (bdaWrites) {
            leave(DispatchRefusal::AddressStores);
            continue;
        }
        if (bytes < window.floor) {
            leave(DispatchRefusal::UnderWindow);
            continue;
        }
        if (bytes > window.limit) {
            leave(DispatchRefusal::OverWindow);
            continue;
        }
        if (!guestMemory.BoundInPlace(begin, bytes)) {
            leave(DispatchRefusal::OutsideImport);
            continue;
        }
        // An image result still to be stored into the range is not in the import yet: in place,
        // as before (the build's flush left it pending only when it could not be stored).
        if (PendingStorageOverlaps(begin, bytes, nullptr) || AnyShadowedOverlaps(begin, bytes)) {
            leave(DispatchRefusal::PendingImage);
            continue;
        }
        std::shared_ptr<Buffer> buffer;
        if (recorder.PendingWriteOverlaps(begin, bytes)) {
            // Recorded work still writes the range, in queue order before this dispatch: only a
            // copy recorded here, after it, holds what the dispatch would read in place.
            if (window.mode < 2) {
                leave(DispatchRefusal::PendingWrite);
                continue;
            }
            std::uint32_t adjustment = 0;
            const auto source = guestMemory.SharedDescriptor(item.address, item.size, adjustment);
            if (!source.has_value() || adjustment != item.adjustment) {
                leave(DispatchRefusal::OutsideImport);
                continue;
            }
            try {
                buffer = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            } catch (const std::exception&) {
                leave(DispatchRefusal::NoBuffer);
                continue;
            }
            gpuCopies.emplace_back(*source, result->snapshots.size());
            ++stats.gpuCopies;
            stats.gpuBytes += bytes;
            ++stats.video;
        } else {
            if (!GuestMemory::Accessible(reinterpret_cast<const void*>(begin), bytes)) {
                leave(DispatchRefusal::OutsideImport);
                continue;
            }
            // Collected before the lookup and before a copy, as addressSnapshot does: the reuse
            // check then sees every CPU store so far, and a store made during the copy is stamped
            // newer by the next walk. A range outside the write watch could never be reused.
            const GuestMemory::CollectSiteScope collectSite(GuestMemory::CollectSite::DrawSnapshot);
            const auto generation = GuestMemory::CollectWrites(begin, bytes);
            if (generation == 0) {
                leave(DispatchRefusal::Unwatched);
                continue;
            }
            buffer = recorder.ReusableDrawSnapshot(begin, bytes, Recorder::SnapshotUse::Storage, nullptr, generation);
            if (buffer != nullptr) {
                ++stats.reused;
            } else {
                // Mappable video memory whatever APS5_VRAM_BUFFERS says (system memory when the
                // device has none); a snapshot there keeps its bytes in system memory too, for
                // the cache's compares.
                try {
                    buffer = std::make_shared<Buffer>(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                } catch (const std::exception&) {
                    leave(DispatchRefusal::NoBuffer);
                    continue;
                }
                std::vector<std::byte> shadow;
                if (buffer->InVideoMemory()) {
                    shadow.assign(reinterpret_cast<const std::byte*>(begin), reinterpret_cast<const std::byte*>(begin) + bytes);
                    std::memcpy(buffer->Bytes().data(), shadow.data(), bytes);
                } else {
                    std::memcpy(buffer->Bytes().data(), reinterpret_cast<const void*>(begin), bytes);
                }
                recorder.KeepDrawSnapshot(begin, bytes, generation, registryGeneration, buffer, Recorder::SnapshotUse::Storage, 0, std::move(shadow));
                ++stats.cpuCopies;
                stats.cpuBytes += bytes;
            }
            if (buffer->InVideoMemory()) ++stats.video;
        }
        ++stats.bound;
        selected.push_back(index);
        infos.push_back({buffer->Handle(), 0, bytes});
        result->snapshots.push_back({begin, std::move(buffer)});
    }
    if (!selected.empty()) ++stats.served;
    reportDispatchSnapshots(stats);
    if (selected.empty()) return {};
    Require(context.descriptorCache != nullptr, "dispatch snapshots require a descriptor cache");
    if (drawBindingSizes.empty()) {
        std::map<VkDescriptorType, std::uint32_t> counts;
        for (const auto& binding : bindings) counts[binding.layout.descriptorType] += binding.layout.descriptorCount;
        for (const auto& [type, count] : counts) drawBindingSizes.push_back({type, count});
    }
    result->cache = context.descriptorCache;
    result->allocation = result->cache->Allocate(_layout, drawBindingSizes);
    Require(result->allocation.set != VK_NULL_HANDLE, "dispatch snapshot descriptor allocation failed");
    // The build's set, copied, with the selected elements rebound (as PrepareDrawBindings does).
    thread_local std::vector<VkCopyDescriptorSet> copies;
    copies.clear();
    for (const auto& binding : bindings) {
        VkCopyDescriptorSet copy{VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET};
        copy.srcSet = _set;
        copy.srcBinding = binding.layout.binding;
        copy.dstSet = result->allocation.set;
        copy.dstBinding = binding.layout.binding;
        copy.descriptorCount = binding.layout.descriptorCount;
        copies.push_back(copy);
    }
    const auto update = context.Resolved(&DeviceFunctions::updateDescriptorSets, "vkUpdateDescriptorSets");
    update(context.device, 0, nullptr, static_cast<std::uint32_t>(copies.size()), copies.data());
    thread_local std::vector<VkWriteDescriptorSet> writes;
    writes.clear();
    for (const auto& binding : bindings) {
        for (std::size_t element = 0; element < binding.allocations.size(); ++element) {
            const auto found = std::find(selected.begin(), selected.end(), binding.allocations[element]);
            if (found == selected.end()) continue;
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = result->allocation.set;
            write.dstBinding = binding.layout.binding;
            write.dstArrayElement = static_cast<std::uint32_t>(element);
            write.descriptorCount = 1;
            write.descriptorType = binding.layout.descriptorType;
            write.pBufferInfo = &infos[static_cast<std::size_t>(found - selected.begin())];
            writes.push_back(write);
        }
    }
    update(context.device, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
    if (!gpuCopies.empty()) {
        // After every write recorded so far (the leading barrier), before the dispatch, whose own
        // leading barrier (transfer writes to compute reads, forced by `copiedOnGpu`) makes the
        // copies visible to it. The imports are live, or retired ones this batch already keeps.
        const auto timing = recorder.BeginGpuTiming(Recorder::CommandClass::DispatchSnapshot);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        Recorder::CountBarriers(Recorder::CommandClass::DispatchSnapshot);
        std::uint64_t copiedBytes = 0;
        for (const auto& [source, snapshot] : gpuCopies) {
            CopyBuffer(context, commands, source.buffer, source.offset, result->snapshots[snapshot].buffer->Handle(), 0, source.range);
            copiedBytes += source.range;
        }
        recorder.EndGpuTiming(timing, copiedBytes);
        copiedOnGpu = true;
    }
    recorder.Keep(result);
    return result;
}

std::string ShaderResources::DescribePlacements() const {
    using Placement = GuestBufferMemory::Placement;
    static constexpr std::array<const char*, static_cast<std::size_t>(Placement::Count)> names{"staged device-local", "in place (read-only)", "in place (address-based)", "in place (outside the staging window)", "in place (not stageable)", "copied by the GPU (misaligned)", "mirror", "copied by the CPU (outside an import)", "unbound"};
    std::array<std::size_t, static_cast<std::size_t>(Placement::Count)> counts{};
    std::string elements;
    std::size_t listed = 0;
    for (const auto& item : allocations) {
        if (!item.guest) continue;
        const auto placement = static_cast<std::size_t>(guestMemory.PlacementOf(item.address, item.size, usesBda));
        ++counts[placement];
        if (listed++ >= 16) continue;
        char text[128];
        std::snprintf(text, sizeof(text), " 0x%llx+0x%zx %s: %s;", static_cast<unsigned long long>(item.address), item.size, item.written ? "written" : "read-only", names[placement]);
        elements += text;
    }
    std::string text = " elements:";
    for (std::size_t placement = 0; placement < counts.size(); ++placement) {
        if (counts[placement] == 0) continue;
        char entry[96];
        std::snprintf(entry, sizeof(entry), " %zu %s,", counts[placement], names[placement]);
        text += entry;
    }
    if (listed == 0) text += " none";
    return text + elements;
}

void ShaderResources::Bind(VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout layout) const {
    if (_set == VK_NULL_HANDLE) return;
    context.Resolved(&DeviceFunctions::cmdBindDescriptorSets, "vkCmdBindDescriptorSets")(commands, bindPoint, layout, 0, 1, &_set, 0, nullptr);
}

namespace {
// Debug aid: APS5_GPU_NO_WRITEBACK=1 keeps GPU results out of guest memory.
bool SkipWriteBack() {
    static const bool skip = std::getenv("APS5_GPU_NO_WRITEBACK") != nullptr;
    return skip;
}
}

void ShaderResources::WriteBack() {
    WriteBackBuffers();
    if (SkipWriteBack()) return;
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (storageWritten[index]) storageTextures[index]->MarkDirty();
    }
}

void ShaderResources::MarkGpuWrites(Recorder& recorder, bool repeated) {
    // The CP's next read of the GDS is ordered after this work (Pm4::InstallGdsBacking).
    if (usesGds) Pm4::NoteGdsShaderUse();
    // APS5_RESIDENT_STAGING: stores through addresses carry no stamp until the batch completes.
    if (bdaWrites) GuestBufferMemory::NoteAddressStores();
    // The ranges this use reads in place through their host imports (read-only and written elements
    // alike, and an address-based build's whole leased heaps), before the writes: a CPU store into
    // one of them (the copy HLE) must not land before the recorded work read it.
    // APS5_REUSE_ADDRESS_DRAWS: an address-based use served by the cached address space alone
    // notes that space's heaps once per batch (the set is the same for every such use, and the
    // batch keeps it), unless the capture trace wants every note.
    const auto readSet = ReuseAddressDraws() && Recorder::ReadTracking() && !CaptureTrace::Enabled() ? guestMemory.ReadSetToken() : 0;
    if (readSet == 0 || !recorder.ReadSetNoted(readSet)) {
        recorder.NotePendingReads(guestMemory.InPlaceReads(), guestMemory.HoldsLease() ? Recorder::ReadKind::AddressBased : Recorder::ReadKind::DispatchElement);
        if (readSet != 0) recorder.NoteReadSet(readSet);
    }
    if (SkipWriteBack()) return;
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (storageWritten[index]) storageTextures[index]->MarkDirty();
    }
    // Written sub-ranges of buffers the GPU copied out of a host import go back into it by the GPU,
    // recorded here after the work: those regions then need no CPU write-back (HasCopiedWrites),
    // and the note and mark below cover them like direct writes.
    // A shared build's later use in the same batch: the batch holds its notes and marks already.
    if (repeated) return;
    guestMemory.RecordCopyBacks(recorder);
    // Only the written elements' ranges (AddWritable): a read-only element is neither noted here
    // nor marked as a direct write, so CPU reads of its memory never wait for this work.
    recorder.NotePendingWrites(guestMemory.Writes());
    guestMemory.MarkDirectWrites();
    if (!BuildProfiled()) return;
    auto& counters = BufferWrites();
    // Recorded uses only (dispatches and recorded draws): a synchronous draw writes back in
    // WriteBack() and publishes no pending writes, so it has no notes to skip.
    counters.notesSkipped.fetch_add(readOnlyBuffers, std::memory_order_relaxed);
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    auto last = counters.lastReport.load();
    if (nowMs - last < 10000 || !counters.lastReport.compare_exchange_strong(last, nowMs)) return;
    const auto elements = counters.elements.load();
    const auto readOnly = counters.readOnly.load();
    AgcDriver::ReportLine("[buffers] descriptor elements bound: %llu total, %llu written, %llu read-only; %llu pending-write notes skipped\n", static_cast<unsigned long long>(elements), static_cast<unsigned long long>(elements - readOnly), static_cast<unsigned long long>(readOnly), static_cast<unsigned long long>(counters.notesSkipped.load()));
}

void ShaderResources::WriteBackBuffers() {
    // A shared build is never committed: each of its uses completes like this one.
    if (guestMemory.Shared()) {
        CompleteSharedUse();
        return;
    }
    if (bda) bda->CheckFault();
    if (SkipWriteBack()) return;
    guestMemory.WriteBack();
}

void ShaderResources::CompleteSharedUse() {
    if (bda) {
        try {
            bda->CheckFault();
        } catch (...) {
            // The record stays in the buffer the uses share: this object serves no further use
            // (AcquireSharedLease), and the ones already recorded report it as they complete.
            faulted.store(true, std::memory_order_relaxed);
            throw;
        }
    }
    if (SkipWriteBack()) return;
    guestMemory.CompleteShared();
}

bool ShaderResources::WritesMemory() const {
    return usesGds || HoldsLease() || NeedsCompletion() || !guestMemory.Writes().empty() || std::any_of(storageWritten.begin(), storageWritten.end(), [](bool written) { return written; });
}

PassBlock ShaderResources::LegacyPassBlock() const {
    if (HoldsLease()) return PassBlock::Lease;
    if (bda != nullptr && !guestMemory.HasCopiedWrites()) return PassBlock::Completion;
    if (!guestMemory.Writes().empty() || guestMemory.HasCopiedWrites()) return PassBlock::Buffers;
    if (std::any_of(storageWritten.begin(), storageWritten.end(), [](bool written) { return written; })) return PassBlock::Images;
    if (usesGds) return PassBlock::Gds;
    return PassBlock::None;
}

bool ShaderResources::ReadsOverlap(std::uint64_t address, std::size_t bytes) const {
    const auto reads = guestMemory.InPlaceReads();
    return std::any_of(reads.begin(), reads.end(), [&](const auto& range) { return address < range.second && range.first < address + bytes; });
}

std::vector<std::pair<VkImage, bool>> ShaderResources::StorageImages() const {
    std::vector<std::pair<VkImage, bool>> images;
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (storageTextures[index] != nullptr) images.emplace_back(storageTextures[index]->Image(), storageWritten[index]);
    }
    for (const auto& texture : textures) {
        if (texture != nullptr && texture->StorageSource() != nullptr) images.emplace_back(texture->StorageSource()->Image(), false);
    }
    return images;
}

const std::vector<std::pair<VkImage, bool>>& ShaderResources::StorageImageList() const {
    if (!storageImagesListed) {
        storageImageList = StorageImages();
        storageImagesListed = true;
    }
    return storageImageList;
}

bool ShaderResources::PinsDeparted() const {
    for (const auto& texture : textures) {
        if (texture == nullptr) continue;
        if (texture->Departed()) return true;
        if (const auto* source = texture->StorageSource(); source != nullptr && !source->Cached()) return true;
    }
    for (const auto& image : storageTextures) {
        if (image != nullptr && !image->Cached()) return true;
    }
    return false;
}

std::uint64_t ShaderResources::DepartedBytes() const {
    std::uint64_t bytes = 0;
    for (const auto& texture : textures) {
        if (texture != nullptr && texture->Departed() && !texture->ViewsStorageImage()) bytes += texture->AllocationBytes();
    }
    for (std::size_t index = 0; index < storageTextures.size(); ++index) {
        if (index != 0 && storageTextures[index] == storageTextures[index - 1]) continue;
        if (storageTextures[index] != nullptr && !storageTextures[index]->Cached()) bytes += storageTextures[index]->GuestBytes();
    }
    return bytes;
}

std::size_t ResourceCache::SweepDeparted(std::uint64_t* bytes) {
    static const bool enabled = std::getenv("APS5_NO_RESOURCE_CACHE_SWEEP") == nullptr;
    if (!enabled) return 0;
    // Plain: every caller holds GuestMemory::GpuMutex (the draw path).
    static std::uint64_t sweptFrame = ~0ull;
    static std::uint64_t sweptDepartures = ~0ull;
    const auto frame = ResidencyClock::Frame();
    if (frame == sweptFrame) return 0;
    sweptFrame = frame;
    const auto departures = SampledDepartures().load(std::memory_order_relaxed) + StorageDepartures().load(std::memory_order_relaxed);
    if (departures == sweptDepartures) return 0;
    sweptDepartures = departures;
    std::vector<std::shared_ptr<ShaderResources>> dropped;
    {
        std::lock_guard lock(mutex);
        for (auto it = entries.begin(); it != entries.end();) {
            // Only a completed object's lists are fixed; one still building is its builder's.
            if (it->second == nullptr || !it->second->Completed() || !it->second->PinsDeparted()) {
                ++it;
                continue;
            }
            if (bytes != nullptr) *bytes += it->second->DepartedBytes();
            dropped.push_back(std::move(it->second));
            index.erase(it->first);
            it = entries.erase(it);
        }
    }
    return dropped.size();
}

bool ShaderResources::ReadsImage(const StorageTexture* image) const {
    if (image == nullptr) return false;
    for (const auto& texture : textures) {
        if (texture != nullptr && texture->StorageSource() == image) return true;
    }
    return std::any_of(storageTextures.begin(), storageTextures.end(), [&](const auto& storage) { return storage.get() == image; });
}

}
