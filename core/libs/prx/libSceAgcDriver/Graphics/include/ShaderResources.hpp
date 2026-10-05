#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SHADERRESOURCES_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_SHADERRESOURCES_HPP

#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/BdaResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/PassHazards.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Sampler.hpp"
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Shaders.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace AgcDriver {
struct DrawRecipe;
}

namespace AgcDriver::Graphics {

class Recorder;

void FlushCachedTextures(VkDevice device);
void ClearCachedTextures(VkDevice device);

// The cached storage image of a surface (render targets use it as their resident image); brought up
// to date with guest memory before it is returned.
std::shared_ptr<StorageTexture> CachedStorageSurface(const Context& context, const GuestTextureResource& resource);
// Whether `image` is still the storage cache's image of its surface (what CachedStorageSurface
// would return); a true answer counts as a use for the cache's eviction order, as a lookup would.
bool StorageImageCached(const Context& context, const StorageTexture* image);
bool StorageImageServesKeys(const StorageTexture& image, std::uint64_t dccAddress);

// Defined in Texture.cpp beside the pending-results registry, for the fast Revalidate below: whether
// a storage image other than `except` has results pending in [address, address + bytes).
bool PendingStorageOverlaps(std::uint64_t address, std::size_t bytes, const StorageTexture* except);

// Per-device descriptor objects shared by ShaderResources builds: set layouts by their binding list
// (immutable; kept until the device is torn down, which Vulkan allows even for pipeline layouts made
// from them) and a chain of descriptor pools that sets are freed back to, so a build creates and
// destroys no layout or pool of its own. A set lives exactly as long as its ShaderResources, which
// its batch keeps until the GPU completed. The cache locks itself: stage A of a dispatch build
// (ShaderResources::buildPrepare, from VulkanDevice::PrepareDispatch) takes layouts and sets from it
// without GuestMemory::GpuMutex, while other threads free and update sets of the same pools under
// that lock (Vulkan synchronizes the pool for allocate/free and the set for update, so that is
// legal). APS5_NO_LAYOUT_CACHE=1 / APS5_NO_POOL_CACHE=1 restore the per-build objects.
class DescriptorCache {
public:
    explicit DescriptorCache(const Context& context);
    ~DescriptorCache();
    DescriptorCache(const DescriptorCache&) = delete;
    DescriptorCache& operator=(const DescriptorCache&) = delete;

    // The layout for `key` (binding, type, count, stage flags per binding, as ShaderResources builds
    // it from `bindings`), created on first use.
    VkDescriptorSetLayout Layout(std::span<const std::uint32_t> key, std::span<const VkDescriptorSetLayoutBinding> bindings);
    struct SetAllocation {
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkDescriptorPool pool = VK_NULL_HANDLE;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    };
    // A set of `layout` needing `sizes` descriptors from the pool chain, opening a pool when no pool
    // has room; a null set when the needs exceed what one chain pool holds (the caller then makes a
    // dedicated pool, as before).
    SetAllocation Allocate(VkDescriptorSetLayout layout, std::span<const VkDescriptorPoolSize> sizes);
    void Free(const SetAllocation& allocation) noexcept;
    // APS5_PROFILE_DRAW counters: layouts served from the map / created, sets allocated, pools opened.
    struct Stats {
        std::uint64_t layoutHits = 0;
        std::uint64_t layoutMisses = 0;
        std::uint64_t sets = 0;
        std::uint64_t pools = 0;
        std::uint64_t recycled = 0;
    };
    Stats Counters() const;

private:
    Context context;
    PFN_vkDestroyDescriptorSetLayout destroyLayout;
    PFN_vkDestroyDescriptorPool destroyPool;
    PFN_vkFreeDescriptorSets freeSets;
    mutable std::mutex mutex;
    std::map<std::vector<std::uint32_t>, VkDescriptorSetLayout> layouts;
    std::vector<VkDescriptorPool> pools;
    std::unordered_map<VkDescriptorSetLayout, std::vector<SetAllocation>> spare;
    std::size_t spareSets = 0;
    Stats stats;
};

class ShaderResources {
public:
    ShaderResources(const Context& context, const ShaderRecompiler::RecompileResult& vertex, const ShaderRecompiler::RecompileResult& fragment, const ColorTarget& target, std::uint64_t indexAddress, std::size_t indexBytes);
    // `stageWrites` (APS5_DRAW_STAGING): the draw will be recorded, so its written buffer elements
    // inside host imports may be staged in device memory (GuestBufferMemory::AllowDrawStaging); a
    // build made with it must not serve a synchronous draw (see StagesBuffers).
    ShaderResources(const Context& context, std::span<const CompiledShader> shaders, const ColorTarget& target, std::uint64_t indexAddress, std::size_t indexBytes, std::span<const GuestMemorySnapshot> snapshots = {}, bool stageWrites = false);
    ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots = {});
    // Two-stage build for dispatches (see build): with `deferred` the constructor runs stage A only,
    // which needs no device lock, and Complete() runs stage B under GuestMemory::GpuMutex; until
    // then no other member may be used. `compute` and `snapshots` must outlive Complete(). An
    // address-based shader (BDA tables) builds entirely in Complete(): its lease acquisition
    // reconciles imports and refreshes mirrors, which needs the lock.
    ShaderResources(const Context& context, const CompiledShader& compute, std::span<const GuestMemorySnapshot> snapshots, bool deferred, std::uint64_t dispatchThreads = 0);
    void Complete();
    bool Completed() const { return completed; }
    ~ShaderResources();
    ShaderResources(const ShaderResources&) = delete;
    ShaderResources& operator=(const ShaderResources&) = delete;
    VkDescriptorSetLayout Layout() const;
    void Bind(VkCommandBuffer commands, VkPipelineBindPoint bindPoint, VkPipelineLayout layout) const;
    struct DrawBindings {
        DrawBindings() = default;
        DrawBindings(const DrawBindings&) = delete;
        DrawBindings& operator=(const DrawBindings&) = delete;
        struct Snapshot {
            std::uint64_t address;
            std::shared_ptr<Buffer> buffer;
        };
        DescriptorCache* cache = nullptr;
        DescriptorCache::SetAllocation allocation;
        std::vector<Snapshot> snapshots;
        // Push constant bytes of the elements bound in place at another alignment adjustment than
        // the build's (MovedBuffer::inPlace): the position in the block and the adjustment, zero
        // meaning the byte the stages assembled (the build records no patch for it either).
        std::vector<std::pair<std::uint32_t, std::uint32_t>> pushPatches;
        ~DrawBindings();
    };
    struct MovedBuffer {
        std::size_t allocation;
        std::uint64_t address;
        std::size_t size;
        // A data buffer's words: the draw's own stage's (or this object's), alive for the draw.
        std::span<const std::uint32_t> words;
        // A read-only V# of a shared address-based build's later use: bound in place through the
        // address space, like the build's own elements, at `adjustment` and by `info` (see
        // PrepareDrawBindings), unless it takes a snapshot.
        bool inPlace = false;
        std::uint32_t adjustment = 0;
        VkDescriptorBufferInfo info{};
    };
    // For an address-based build under APS5_REUSE_ADDRESS_DRAWS the result is the calling
    // thread's scratch, valid until its next call (the set and everything it binds are kept by
    // the recorder's open batch; see addressDrawBindings).
    std::shared_ptr<DrawBindings> PrepareDrawBindings(Recorder& recorder, std::span<const MovedBuffer> moved = {}) const;
    // APS5_DISPATCH_SNAPSHOTS (local experiment, see ShaderResources.cpp): the set one use of a
    // compute build binds instead of its own, with its read-only guest buffer elements that are
    // bound in place inside an import served by copies in video memory, or null when no element
    // qualifies (the build's own set is then bound). `commands` is the open batch's command
    // buffer: with APS5_DISPATCH_SNAPSHOTS=2 the copies of ranges recorded work still writes are
    // recorded into it, and `copiedOnGpu` tells the caller its leading barrier must be recorded
    // (it makes those copies visible to the dispatch). Kept by the recorder; per use, so a
    // cached template or a recipe hit proves its snapshots anew like any other use.
    std::shared_ptr<DrawBindings> PrepareDispatchBindings(Recorder& recorder, VkCommandBuffer commands, bool& copiedOnGpu) const;
    static bool DispatchSnapshotsEnabled();
    // The [dispatch-res] trace's account of the guest buffer elements: how many are staged in
    // device memory and how many are not, by reason, then the first elements one by one.
    std::string DescribePlacements() const;
    std::optional<std::vector<MovedBuffer>> MovedReadOnlyBuffers(std::span<const CompiledShader> shaders, Recorder& recorder) const;
    // The same into the caller's vector (cleared first; a draw path keeps one per thread): false
    // when a buffer cannot be moved.
    bool MovedReadOnlyBuffers(std::span<const CompiledShader> shaders, Recorder& recorder, std::vector<MovedBuffer>& moved) const;
    void WriteBack();
    // Deferred completion: MarkGpuWrites registers the results the recorded work leaves on the GPU
    // (storage images stay there; buffer ranges are noted so CPU reads wait); WriteBackBuffers runs
    // once the work completed.
    // `repeated`: a shared address-based build's later use in a batch that already holds its
    // write notes and marks (UsedInBatch): its written ranges are the same for every use, they
    // stay pending until the batch completed and are marked again by its completion.
    void MarkGpuWrites(Recorder& recorder, bool repeated = false);
    // Whether NoteBatchUse was called with this batch (Recorder::OpenBatchId) last: a shared
    // build's uses in one batch register one completion (the fault check, the write marks and
    // the hold on the address space are the batch's, not each draw's). Under GuestMemory::GpuMutex.
    bool UsedInBatch(std::uint64_t batch) const { return batch != 0 && lastUseBatch == batch; }
    void NoteBatchUse(std::uint64_t batch) { lastUseBatch = batch; }
    // Revalidate calls a shared build answered from its proof of the same collect epoch
    // (cumulative, the [addrdraw] line).
    static std::uint64_t ProofsReused();
    // StorageImages, computed once (the images are fixed with the build).
    const std::vector<std::pair<VkImage, bool>>& StorageImageList() const;
    // Whether a texture or storage image this object holds has left its cache (Texture::Departed,
    // StorageTexture::Cached): the object then pins video memory the caches no longer count, and
    // could not be proved current again anyway.
    bool PinsDeparted() const;
    // The device bytes of those images (a departed snapshot texture's allocation, an uncached
    // storage image's surface), for the sweep's count.
    std::uint64_t DepartedBytes() const;
    void WriteBackBuffers();
    // Whether WriteBackBuffers has anything the CPU must see (copied written buffers, BDA faults).
    bool NeedsCompletion() const { return bda != nullptr || guestMemory.HasCopiedWrites(); }
    // Whether a written buffer was copied (its results reach guest memory by the CPU write-back).
    bool HasCopiedWrites() const { return guestMemory.HasCopiedWrites(); }
    bool HoldsLease() const { return guestMemory.HoldsLease(); }
    // Whether a buffer region is staged in device memory: every use must then be recorded and call
    // MarkGpuWrites (the copy-back), which a synchronous draw does not.
    bool StagesBuffers() const { return guestMemory.HasStagedRegions(); }
    GuestBufferMemory::DrawStagingTally DrawStaging() const { return guestMemory.DrawStaging(); }
    bool WritesOverlap(std::uint64_t address, std::size_t bytes) const { return guestMemory.WritesOverlap(address, bytes); }
    // Whether a region the recorded work reads in place through a host import overlaps the range.
    bool ReadsOverlap(std::uint64_t address, std::size_t bytes) const;
    // The recorder's hazard tracker inputs (Recorder::NoteAccess): the written elements' guest
    // ranges, the regions read in place, and the storage images with whether each is written.
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& GpuWrites() const { return guestMemory.Writes(); }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> InPlaceReads() const { return guestMemory.InPlaceReads(); }
    std::vector<std::pair<VkImage, bool>> StorageImages() const;
    const std::vector<std::shared_ptr<Texture>>& SampledTextures() const { return textures; }
    // Whether a use writes guest memory beyond a draw's attachments (storage images, written or
    // copied buffers, an address-based build's unknown writes), or reads `image` (a view of it
    // sampled, or the image itself bound): a recorded draw's render pass may only be continued by
    // a draw for which neither holds (Draw.cpp).
    bool WritesMemory() const;
    PassBlock LegacyPassBlock() const;
    bool UsesGds() const { return usesGds; }
    bool BdaWrites() const { return bdaWrites; }
    bool UsesBda() const { return usesBda; }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> DeviceReads() const { return guestMemory.DeviceReads(); }
    bool ReadsImage(const StorageTexture* image) const;
    const std::vector<std::uint32_t>& LayoutKey() const { return layoutKey; }
    // Debug aid: each bound guest resource with the fraction of sampled bytes that are nonzero.
    std::string Describe() const;

    // Reuse across dispatches of identical content (see the resource cache in VulkanDevice): the
    // key is everything a compute build consumes from its compiled shader (variant, and per binding
    // kind, role, slot, count, guest descriptor words, written images, depth-compare samplers); push
    // constants are not part of it because every dispatch pushes its own. Reusable says whether this
    // object can serve a later dispatch with the same key: nothing needs completion work and every
    // guest buffer is read in place through a host import. Revalidate brings a reusable object up to
    // date for another use, or says it cannot be reused any more (a buffer's import went away, or a
    // texture or storage image the set references is no longer the one its memory maps to).
    // Without `dataWords` the ShaderData and FlattenedSrt descriptor words stay out of the key
    // (their count and size remain): a compute template then serves dispatches whose constants
    // differ, and the hit refreshes its data buffers with the dispatch's words (RefreshData).
    static std::vector<std::uint32_t> ContentKey(const CompiledShader& shader, bool dataWords = true, bool movableBuffers = false);
    // The same words appended to `key` (a draw's key is built in one vector the thread keeps).
    static void AppendContentKey(std::vector<std::uint32_t>& key, const CompiledShader& shader, bool dataWords = true, bool movableBuffers = false);
    // Records the shader's ShaderData and FlattenedSrt words into this object's data buffers
    // (vkCmdUpdateBuffer, a transfer write the caller's pre-dispatch barrier makes visible; a
    // buffer already holding the words is left alone). Returns whether anything was recorded. With
    // `recorder` the updates are one [gputime] range of class TemplateDataRefresh.
    bool RefreshData(VkCommandBuffer commands, const CompiledShader& shader, Recorder* recorder = nullptr);
    // Walks the write watch over every image surface of a completed object (no device lock: the
    // ranges are fixed by the build), so a Revalidate's collects on the same worker are memo hits.
    void PrecollectSurfaces() const;
    bool Reusable() const { return reusable; }
    static bool NeverReusable(std::span<const CompiledShader> shaders);
    // Address-based draw templates (APS5_REUSE_ADDRESS_DRAWS=1, default off; ignored under
    // APS5_SYNC_LEASE_DISPATCH). A build that maps registered memory through BDA tables was never
    // reusable, for five reasons, each handled as follows when the switch is on:
    //  - the lease: the build pins every registered allocation until its write-back, and an object
    //    kept in the resource cache would pin them for good. A shared build holds no lease of its
    //    own: every use holds the cached address space (ShareLease for the first, AcquireSharedLease
    //    for a later one) and its completion releases it, as a single-use build's write-back does;
    //  - the space and the table: the descriptor set names the imports and the BDA table of the
    //    address space the build mapped. A later use must find that very space still cached and
    //    current (AcquireSharedLease) and its imports unretired once the lookups ran (Revalidate);
    //    any change of the registry or the imports fails it and the draw builds anew;
    //  - per-build memory state: mirrors are refreshed per build, copied ranges and captured
    //    regions outside the space are copied per build. A use refreshes the mirrors as a build
    //    does; a build with a region of its own is not shared, and a use whose captured regions do
    //    not all lie in the space takes a build;
    //  - the write-back: a single-use build commits once. A shared build has only written ranges
    //    bound in place (anything else is not shared), so a use's completion only marks them
    //    written (CompleteSharedUse);
    //  - the fault buffer: one per build, read at completion. Uses in flight share it, so a fault
    //    is reported by every use completing after it and retires the object (the next use
    //    builds); a build that scans the buffer's written-page slots (stores by address) clears
    //    per-use state there and is not shared.
    // Only a draw's build can be shared; SharingRefusal says why one was not (Count: not asked).
    static bool ReuseAddressDraws();
    enum class AddressRefusal : std::size_t { None, StoresByAddress, NoSpace, OwnRegions, MirrorWrites, Count };
    AddressRefusal SharingRefusal() const { return addressRefusal; }
    using SharedLease = GuestBufferMemory::SpaceLease;
    bool SharesLease() const { return guestMemory.Shared(); }
    // Turns a reusable address-based build into a shared one before it goes into the resource
    // cache; the result is the building draw's hold, to be released by its completion.
    SharedLease ShareLease();
    // Another use of a shared build, before its Revalidate, under GuestMemory::GpuMutex: null with
    // the reason (a fault reported since, the address space replaced or its imports moved, a
    // captured region outside the space: only then does the object stay valid for other draws),
    // else the use's hold, without which no other member of a shared build may be used.
    enum class SharedMiss : std::size_t { None, Faulted, Space, Snapshot, Count };
    SharedLease AcquireSharedLease(std::span<const GuestMemorySnapshot> snapshots, SharedMiss& miss);
    // One use's completion work (WriteBackBuffers runs it for a shared build): the fault check and
    // the write marks. The caller releases the use's hold afterwards.
    void CompleteSharedUse();
    // The read-only buffer elements of address-based draws under APS5_REUSE_ADDRESS_DRAWS with
    // APS5_SNAPSHOT_ADDRESS_DRAWS (PrepareDrawBindings: a build's own elements and the ones a
    // template or plan hit moved), cumulative, for the [addrdraw] line: elements bound to a
    // snapshot (`video`: one whose buffer is in video memory; `small`: an element of at most
    // APS5_SNAPSHOT_ADDRESS_DRAWS_SMALL_BYTES copied into the batch arena with no collect and no
    // cache entry), of them the binds that skipped
    // the checks by the collect epoch (Recorder::EpochSnapshot; `epochRechecks`: after asking
    // the range again because a driver stamp or a write note had moved), reused a cached
    // snapshot after its checks, or copied one; and the elements left bound in place through the import, by
    // reason: under or over the size window (APS5_SNAPSHOT_ADDRESS_DRAWS_MIN_KIB / _MAX_KIB),
    // written by the build, a GPU write or image result pending over the range, not in an import
    // bound in place (a mirror, a region of the build's own, an inaccessible page), not
    // write-watched (a snapshot of it could never be reused). Read and written under
    // GuestMemory::GpuMutex.
    enum class SnapshotRefusal : std::size_t { UnderWindow, OverWindow, Written, Pending, OutsideImport, Unwatched, Count };
    struct AddressSnapshotStats {
        std::uint64_t bound = 0;
        std::uint64_t video = 0;
        std::uint64_t epochSkips = 0;
        std::uint64_t epochRechecks = 0;
        std::uint64_t reused = 0;
        std::uint64_t copied = 0;
        std::uint64_t copiedBytes = 0;
        std::uint64_t small = 0;
        std::uint64_t smallBytes = 0;
        // Window collects (see ShaderResources.cpp CollectWindowed): windows walked whole, the
        // element collects those walks answered (each would have been a walk of its own, less
        // the one per window that the window walk replaced) and the windows that could not be
        // walked whole.
        std::uint64_t windowWalks = 0;
        std::uint64_t windowServed = 0;
        std::uint64_t windowRefused = 0;
        // A draw's own set (addressDrawBindings): bound again from an identical draw of the same
        // batch, or built; of the built ones, those whose bindings repeated a recent draw of the
        // template in another batch (a set kept across batches would have served), and those
        // that repeated one but for the batch arena's copies (the draw's data buffers, its small
        // elements): what a set whose arena-backed bindings were not part of its identity
        // would serve.
        std::uint64_t setsReused = 0;
        std::uint64_t setsBuilt = 0;
        std::uint64_t setsRepeatedAcrossBatches = 0;
        std::uint64_t setsRepeatedButArena = 0;
        std::array<std::uint64_t, static_cast<std::size_t>(SnapshotRefusal::Count)> inPlace{};
    };
    static const AddressSnapshotStats& AddressSnapshotCounters();
    // Draw plans (APS5_DRAW_PLANS, see Draw.hpp): the recipes of the draws this cached object
    // served, by the draws' plan key (their state key and push constant placement), most recently
    // attached first and bounded; an attach under a key replaces that key's plan. Under
    // GuestMemory::GpuMutex.
    std::shared_ptr<const DrawRecipe> FindPlan(std::uint64_t planKey) const;
    void AttachPlan(std::uint64_t planKey, std::shared_ptr<const DrawRecipe> plan);
    // `shaders` are the stages the object was built from, in build order (a recorded draw's vertex
    // and fragment stages, or one compute stage): their bindings are walked like the build did.
    // How a Revalidate proved (or refused) the object, for the [recipe] line: the proof path taken
    // and, on a refusal, what failed.
    enum class ProofPath : std::size_t { Fast, OwnRefreshed, Full, Count };
    enum class ProofFailure : std::size_t { None, Imports, Evicted, Pending, Changed, Keys, Other, Count };
    struct ProofReport {
        ProofPath path = ProofPath::Fast;
        ProofFailure failure = ProofFailure::None;
    };
    bool Revalidate(std::span<const CompiledShader> shaders, ProofReport* report = nullptr);
    bool Revalidate(const CompiledShader& shader, ProofReport* report = nullptr) { return Revalidate(std::span<const CompiledShader>(&shader, 1), report); }
    // The proof a recipe hit runs on its template (design_cpu_final M4): Revalidate by that name.
    bool ProveCurrent(std::span<const CompiledShader> shaders, ProofReport* report = nullptr) { return Revalidate(shaders, report); }
    bool ProveCurrent(const CompiledShader& shader, ProofReport* report = nullptr) { return Revalidate(shader, report); }
    // FNV over the ShaderData/FlattenedSrt words, in binding order: of this object's data buffers
    // as the recorded work will find them (maintained by addDataBuffer and RefreshData), and of a
    // compiled shader's descriptors. A recipe hit refreshes the template iff the two differ.
    std::uint64_t DataWordsHash() const { return dataWordsHash; }
    static std::uint64_t DataWordsHash(const CompiledShader& shader);
    void PatchPushConstants(std::span<std::byte, PipelinePushConstantBytes> bytes) const {
        for (const auto& [position, adjustment] : pushPatches) bytes[position] = static_cast<std::byte>(adjustment);
    }
    // Whether RefreshData would record anything for `shader` (the per-word compare; verification).
    bool DataWordsDiffer(const CompiledShader& shader) const;
    // Why the fast proof of a Revalidate left the object to the full walk (the [rescache]
    // revalidate line's reasons); Count: it did not.
    enum class FastFail : std::size_t { NoRecord, Collect, Pending, Evicted, Changed, Keys, ClearedView, StorageKeys, Count };
    // Why a Pending failure was left to the full walk instead of the own-object refresh (T1, see
    // refreshOwnObjects); Count: it was not.
    enum class OwnRefreshFallback : std::size_t { Disabled, Snapshot, Keys, ForeignView, SurfaceKey, NotImported, Uncached, Rerun, Count };
    // APS5_PROFILE_DRAW: the build's sub-phases ([resources] totals) and their times in
    // milliseconds (zero when not profiling), so the draw path can total them separately from
    // dispatches. bindingsMs covers the binding plan (stage A) and the image lookups (stage B);
    // prepareMs/completeMs are the two stages' totals.
    enum class BuildPhase : std::size_t { Bindings, Precollect, Upload, Descriptors, StageA, Images, Bda, StageB, Count };
    struct BuildTiming {
        double bindingsMs = 0;
        double uploadMs = 0;
        double descriptorsMs = 0;
        double prepareMs = 0;
        double completeMs = 0;
    };
    const BuildTiming& Timing() const { return timing; }
    // The image surfaces (address, bytes) whose stage-B lookup reads guest memory on the CPU and so
    // would wait, under the device lock, for recorded work writing them: a surface without a host
    // import, or a sampled one that stays a CPU snapshot (block compressed, or the storage-view path
    // is off). VulkanDevice::PrepareDispatch waits for that work before the lock (the pre-sync).
    // Usable right after the deferred constructor, for address-based builds too (it decodes the
    // compiled shader's bindings itself when stage A has not run), and on a completed object from
    // the resource cache (its Revalidate repeats the same lookups), without the device lock.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> PresyncSurfaces() const;

private:
    struct DescribedRange {
        const char* kind;
        std::uint64_t address;
        std::uint64_t bytes;
        std::uint32_t width = 0;
        std::uint32_t height = 0;
        std::uint32_t format = 0;
        int tileMode = 0;
        std::uint64_t dccAddress = 0;
    };
    std::vector<DescribedRange> describedRanges;
    struct Allocation {
        std::uint64_t address;
        std::size_t size;
        bool guest;
        std::unique_ptr<Buffer> buffer;
        ShaderRecompiler::DescriptorRole role = ShaderRecompiler::DescriptorRole::ShaderData;
        // Guest buffers: whether the shader may store to the range (see addGuestBuffer).
        bool written = true;
        // Data buffers: the words the buffer holds once the recorded work ran (see RefreshData).
        std::vector<std::uint32_t> dataWords;
        std::uint32_t adjustment = 0;
        std::int32_t pushByte = -1;
        std::int64_t dataAllocation = -1;
        std::uint32_t dataByte = 0;
    };
    struct DataPatch {
        std::size_t allocation;
        std::uint32_t byte;
        std::uint32_t adjustment;
    };
    void writeDataWords(VkCommandBuffer commands, std::size_t allocation, std::span<const std::uint32_t> words) const;

    struct Binding {
        VkDescriptorSetLayoutBinding layout;
        std::vector<std::size_t> allocations;
        std::vector<std::size_t> imageAllocations;
    };

    // A host-imported buffer region the set reads in place, with its import's identity at build time.
    struct DirectRegion {
        std::uint64_t begin;
        std::uint64_t end;
        std::uint64_t serial;
    };

    // The build is two stages. A (buildPrepare, no device lock): the binding plan with its layout
    // entries, samplers, data buffers, the upload's prepare stage (imports already serving the
    // regions, read-only copies), the set layout and the descriptor set. B (buildComplete, under
    // GuestMemory::GpuMutex): the texture and storage image lookups (they refresh, upload and flush
    // through the recorder), the rest of the upload, the BDA objects, the descriptor writes and the
    // reusability record. build runs both.
    void build(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint64_t indexAddress, std::size_t indexBytes);
    void buildPrepare(std::span<const CompiledShader> shaders, const ColorTarget* target, std::uint64_t indexAddress, std::size_t indexBytes);
    void buildComplete();
    // APS5_PROFILE_DRAW: closes the current sub-phase of the build into the [resources] totals.
    double phase(BuildPhase which);
    // `written` is the element's DescriptorBinding::bufferWritten: a read-only element binds the
    // same way but is left out of the write set (no write-back, no pending-write note). `atomic`
    // is its bufferAtomic (see GuestBufferMemory::AddWritable).
    std::size_t addGuestBuffer(std::span<const std::uint32_t> words, const ColorTarget* target, std::uint64_t indexAddress, std::size_t indexBytes, bool written, bool atomic, bool read);
    std::size_t addDataBuffer(std::span<const std::uint32_t> words);
    // Stage A: the layout entry of an image binding (samplers are taken at once, the sampler cache
    // locks itself); stage B looks the sampled textures and storage images up (resolveImageBinding).
    void addImageBinding(const ShaderRecompiler::DescriptorBinding& binding, VkShaderStageFlags flags);
    // Stage A: one record per planned image element (ImageRecord), in plan order: the decoded
    // descriptor and its surface size (computed once for the build), the write watch walked so stage
    // B's collects are memo hits (APS5_NO_PRECOLLECT=1 skips the pass entirely), and for a sampled
    // element the cache entry it maps to, taken now so stage B needs only the fastRevalidate
    // predicate under the lock (APS5_NO_STAGE_A_IMAGES=1 leaves every element to the full lookup).
    // Whether it ran.
    bool precollectImages();
    struct ImageRecord {
        bool sampled = false;
        // The descriptor decoded and its surface described; false leaves the element to stage B's
        // own decode (which reports the error).
        bool decoded = false;
        std::array<std::uint32_t, 8> words{};
        GuestTextureResource resource{};
        std::uint64_t guestBytes = 0;
        VkComponentMapping components{};
        // Sampled elements: the keys read in stage A and the collect made BEFORE stage B's checks
        // (the generation the entry moves to when they pass).
        DccKeys keys = DccKeys::Uncompressed;
        std::uint64_t generation = 0;
        // The cache entry's object, storage source, keys and generation as of stage A (texture null:
        // no entry, the full lookup makes one).
        std::shared_ptr<Texture> texture;
        std::shared_ptr<StorageTexture> source;
        DccKeys entryKeys = DccKeys::Uncompressed;
        std::uint64_t entryGeneration = 0;
    };
    std::vector<ImageRecord> imageRecords;
    struct PreviousSampled {
        std::array<std::uint32_t, 8> words{};
        bool depthCompare = false;
        std::shared_ptr<Texture> texture;
    };
    PreviousSampled previousSampled;
    // The next record resolveImageBinding consumes (records follow the deferredImages order).
    std::size_t nextImageRecord = 0;
    // Stage B: the record's texture when the fastRevalidate predicate proves it current under the
    // lock and the cache still holds it; null sends the element to cachedTexture.
    std::shared_ptr<Texture> fastTexture(ImageRecord& record);
    void resolveImageBinding(const ShaderRecompiler::DescriptorBinding& binding, Binding& item);
    void forgetDeferredInputs();
    void release() noexcept;
    void prepareAddressBindings(std::span<const CompiledShader> shaders, std::span<const GuestMemorySnapshot> snapshots);
    VkDescriptorBufferInfo descriptor(Allocation& allocation);
    void noteReusable();
    AddressRefusal sharingRefusal() const;
    // The snapshot an address-based draw binds for the read-only range [begin, begin + bytes) of
    // one of its elements instead of the import (see AddressSnapshotStats): `info` receives it,
    // kept by the recorder's open batch; false leaves the element in place. Under the stamp
    // addressDrawBindings took for the call.
    // `transient`: the snapshot is a copy in the batch arena (good for the collect epoch only).
    bool addressSnapshot(Recorder& recorder, std::uint64_t begin, std::size_t bytes, VkDescriptorBufferInfo& info, bool& transient) const;
    // PrepareDrawBindings for an address-based build under APS5_REUSE_ADDRESS_DRAWS: the draw's
    // own data buffers (from the batch arena), its moved elements and, under
    // APS5_SNAPSHOT_ADDRESS_DRAWS, the snapshots of its read-only elements, in a set the batch
    // frees; nothing is allocated per draw but the set.
    std::shared_ptr<DrawBindings> addressDrawBindings(Recorder& recorder, std::span<const MovedBuffer> moved) const;
    // The descriptor counts of a draw's own set (PrepareDrawBindings), computed once.
    mutable std::vector<VkDescriptorPoolSize> drawBindingSizes;
    void reportDescriptorCaches() const;
    // What a sampled texture was proved current against when the build (or the last full Revalidate)
    // looked it up, so the next Revalidate can repeat the proof from write stamps and the DCC keys
    // alone instead of repeating the lookup (see fastRevalidate). One record per textures[] entry; an
    // element without a record always takes the full lookup. Storage images need no record: their
    // proof comes from the image itself (its own surface and generation).
    struct ValidatedSurface {
        GuestTextureResource resource{};
        std::uint64_t bytes = 0;
        // Keys read at the lookup (a snapshot texture holds its clear texels while they are unchanged).
        DccKeys keys = DccKeys::Uncompressed;
        // Snapshot textures: write generation the content was known current at, always from a
        // collect issued before the checks that proved it.
        std::uint64_t generation = 0;
        // The collect of the check in progress, moved into `generation` once every element passed.
        std::uint64_t collected = 0;
        // Storage-sourced textures: the image the view follows (kept alive by the texture).
        const StorageTexture* source = nullptr;
        bool valid = false;
    };
    void captureValidation();
    bool keepsTemplateRecords() const;
    // A surface of this object the fast proof found a foreign image pending over: the element
    // (index into textures, or into storageTextures when `storage`) whose own object stands in for
    // the full walk's lookup (refreshOwnObjects). `viewUncompressed`: a storage-sourced view under
    // uncompressed keys, which the lookup serves by its own source whenever FindPending names it.
    // `sourceEligible` (set by refreshOwnObjects on a resolved view): the lookup views the cached
    // image of the surface's key, i.e. the view's source, when FindPending names nothing.
    struct PendingOverlap {
        std::size_t element;
        bool storage;
        bool viewUncompressed;
        bool sourceEligible = false;
    };
    // `serialBefore` is StorageTexture::PendingSerial() loaded before any check of this Revalidate;
    // `refreshed` the surfaces refreshOwnObjects resolved earlier in the same Revalidate (empty on
    // the first run): a foreign image still pending over one of them (the own image's alias, an
    // image whose pending units lie outside the surface) is no failure, since the lookup's answer
    // for the surface is the own object its Refresh just brought current, whatever the registry
    // holds; `reason` receives why the proof failed (Count: it passed), `overlapping` the surfaces
    // of a Pending failure in element order (empty otherwise), `accepted` whether a foreign overlap
    // was accepted by either rule (the verify switch then runs the full walk beside the proof).
    bool fastRevalidate(std::uint64_t serialBefore, std::span<const PendingOverlap> refreshed, FastFail& reason, std::vector<PendingOverlap>& overlapping, bool& accepted);
    // T1 (design_cpu_final M3): brings the overlapping surfaces' own objects up to date exactly as
    // the full walk's lookups would, in the walk's binding order, without the lookups: a storage
    // image's Refresh (which stores the foreign images over its surface first), a view's source's
    // Refresh when the lookup would refresh the cached image of the view's surface key, nothing
    // when FindPending names the source itself (the lookup hits on it). Count, or the fallback
    // reason: a surface whose lookup could return another object (a snapshot texture, a view under
    // fast-clear keys, a foreign pending image the lookup would view, a surface key mapping to
    // another cached image, a surface outside the host imports, an object no longer cached), which
    // leaves the whole object to the full walk. Under GuestMemory::GpuMutex, as Revalidate is.
    OwnRefreshFallback refreshOwnObjects(std::span<const CompiledShader> shaders, std::span<PendingOverlap> overlapping);
    // Today's per-element walk (APS5_NO_EPOCH_REVALIDATE=1).
    bool fastRevalidateEach();
    Context context;
    std::vector<std::uint32_t> layoutKey;
    GuestBufferMemory guestMemory;
    std::unique_ptr<BdaResources> bda;
    bool usesBda = false;
    bool usesFaultBuffer = false;
    // A GDS binding (the device's GDS buffer, Context::gdsBuffer): every use is noted for the CP
    // (Pm4::NoteGdsShaderUse) and counts as a memory write.
    bool usesGds = false;
    bool bdaWrites = false;
    VkDescriptorSetLayout _layout = VK_NULL_HANDLE;
    // Whether the layout is this object's own (no cache) and destroyed with it.
    bool ownsLayout = false;
    VkDescriptorSet _set = VK_NULL_HANDLE;
    // Dedicated pool of this object's set (no cache, or a set too large for a cache pool).
    VkDescriptorPool pool = VK_NULL_HANDLE;
    // The cache pool the set was allocated from, freed back to it on release.
    VkDescriptorPool cachePool = VK_NULL_HANDLE;
    std::vector<Allocation> allocations;
    // Guest buffer elements bound read-only: each use of this object skips that many pending-write
    // notes (counted in MarkGpuWrites for the [buffers] line).
    std::size_t readOnlyBuffers = 0;
    std::uint64_t dispatchThreads = 0;
    std::vector<std::shared_ptr<Texture>> textures;
    std::vector<bool> textureFirstLayer;
    std::vector<std::shared_ptr<StorageTexture>> storageTextures;
    std::vector<std::uint32_t> storageMips;
    std::vector<std::uint64_t> storageKeys;
    std::vector<bool> storageFirstLayer;
    std::vector<bool> storageWritten;
    std::vector<std::shared_ptr<Sampler>> samplers;
    bool reusable = false;
    // A draw's build under APS5_REUSE_ADDRESS_DRAWS: it keeps the template records and may be
    // shared when it holds a lease (see ReuseAddressDraws).
    bool addressReuse = false;
    AddressRefusal addressRefusal = AddressRefusal::Count;
    // A use of this shared build reported a fault (set under GuestMemory::GpuMutex by a completion,
    // atomic for the reader's sake): it serves no further use.
    std::atomic<bool> faulted{false};
    std::uint64_t lastUseBatch = 0;
    // The last few sets addressDrawBindings made for this object: the hash of everything they
    // bound (and of the push constant patches that go with it), the same with the arena-backed
    // bindings left out, the set and the batch that owns it. A draw of the same batch with the
    // same hash binds the set again; the rest is counted (AddressSnapshotStats).
    struct DrawSetMemo {
        std::uint64_t full = 0;
        std::uint64_t stable = 0;
        VkDescriptorSet set = VK_NULL_HANDLE;
        std::uint64_t batch = 0;
    };
    mutable std::array<DrawSetMemo, 4> drawSetMemos{};
    mutable std::uint8_t drawSetNext = 0;
    mutable std::vector<std::pair<VkImage, bool>> storageImageList;
    mutable bool storageImagesListed = false;
    // A shared build's image proof within a collect epoch (APS5_REUSE_ADDRESS_DRAWS; see
    // Revalidate): what the last successful proof stood on.
    struct ProofStamp {
        std::uint64_t epoch = 0;
        std::uint64_t driverStores = 0;
        std::uint64_t writeNotes = 0;
        std::uint64_t registry = 0;
        std::uint64_t pendingSerial = 0;
        std::uint64_t departures = 0;
        std::uint64_t depthHolding = 0;
        bool operator==(const ProofStamp&) const = default;
    };
    ProofStamp provedStamp;
    std::vector<std::pair<std::uint64_t, std::shared_ptr<const DrawRecipe>>> plans;
    std::vector<DirectRegion> directRegions;
    std::vector<ValidatedSurface> validatedTextures;
    // The pending registry's serial at the last Revalidate that proved this object, taken before
    // its checks and kept only when unchanged after them (0: none): while it is still the serial,
    // no image other than the object's own sources has results pending over its surfaces and
    // regions, so those checks are skipped (see fastRevalidate).
    std::uint64_t pendingSerialSeen = 0;
    std::uint64_t depthHoldingSeen = 0;
    // The import table's identity when the direct regions' serials were last proved.
    HostImportsProof importsProof;
    // FNV-1a offset basis: the hash of no data buffers (DataWordsHash).
    std::uint64_t dataWordsHash = 14695981039346656037ull;
    void rehashDataWords();
    std::vector<std::pair<std::uint32_t, std::uint32_t>> pushPatches;
    std::vector<DataPatch> dataPatches;
    BuildTiming timing;
    // Build state carried from stage A to stage B: the bindings in plan order, the image bindings
    // still to look up (index into `bindings`; the DescriptorBinding lives in the compiled shader),
    // the descriptor counts the set was sized for, and the compute stage of a deferred build.
    std::vector<Binding> bindings;
    struct DeferredImages {
        const ShaderRecompiler::DescriptorBinding* binding;
        std::size_t index;
    };
    std::vector<DeferredImages> deferredImages;
    std::uint64_t storageBuffers = 0;
    std::uint32_t plannedSampledImages = 0;
    std::uint32_t plannedStorageImages = 0;
    // The compute constructor's shader and captured regions: the caller's objects, valid only until
    // the build (Complete() for a deferred one) is done, and reset then (forgetDeferredInputs).
    CompiledShader deferredCompute{};
    std::span<const GuestMemorySnapshot> deferredSnapshots;
    // The deferred build is address-based: everything runs in Complete().
    bool lockedBuild = false;
    // Stage A runs without the device lock (a deferred, non-address-based build).
    bool unlockedPrepare = false;
    bool completed = false;
    std::chrono::steady_clock::time_point phaseStart;
};

// Built ShaderResources by the content the build consumed (see ShaderResources::ContentKey), so a
// dispatch or recorded draw repeating an earlier one binds the earlier descriptor set instead of
// building one (import lookups, sampler and descriptor objects, guest buffer uploads). Only reusable
// objects (nothing to complete, every buffer read in place) are kept, and an entry serves a later
// use only after ShaderResources::Revalidate brought it up to date. Entries hold their object alive;
// the recorder keeps it as well while its batch runs. One LRU bound covers dispatch and draw entries
// together (APS5_RESOURCE_CACHE_ENTRIES, default 1024); the map has its own mutex and locks itself:
// VulkanDevice::PrepareDispatch probes it without GuestMemory::GpuMutex (a hit there is revalidated
// under the lock by the dispatch; inserts and Revalidate stay under it). Compute keys start with the
// stage, draw keys with a marker word, so the two never meet, and both name the device handle: the driver replaces
// the headless device with the windowed one while workers may still hold the old one, so a replaced
// device's entries (whose descriptor sets and pooled buffers belong to it) are unreachable by key
// and dropped by that device's Clear(). APS5_NO_RESOURCE_CACHE=1 (dispatches) and
// APS5_NO_DRAW_RESOURCE_CACHE=1 (draws) build every object as before; the device clears the cache at
// teardown before its descriptor caches go.
class ResourceCache {
public:
    using Key = std::vector<std::uint32_t>;
    // `probe`: a lookup the caller repeats on a miss (a draw plan's, before the draw's own), left
    // out of the miss churn accounting so one draw's miss is charged once.
    std::shared_ptr<ShaderResources> Find(const Key& key, bool probe = false);
    // Drops the entries whose object holds a texture or storage image that left its cache
    // (ShaderResources::PinsDeparted): the texture caches evict under a video memory budget and
    // count an evicted image as freed, while an entry here kept it alive, so a full cache of
    // idle templates held gigabytes past the budget (4096 entries exhausted a 16 GiB card).
    // Called by the draw path; walks the entries once per residency frame, and only when a cache
    // dropped something since the last walk. Returns how many it dropped (they are destroyed
    // after the cache's lock is released) and adds the bytes of the departed images they held
    // to `bytes` (an image several entries held is counted for each).
    // APS5_NO_RESOURCE_CACHE_SWEEP=1 never walks.
    std::size_t SweepDeparted(std::uint64_t* bytes = nullptr);
    // `evicted`, when given, receives the objects the insert displaces (the entry replaced under the
    // key, the ones over the bound) instead of their being destroyed here: a caller under the GPU
    // mutex hands them to the recorder so the destruction runs off the lock (VulkanDevice::dispatch).
    void Insert(const Key& key, std::shared_ptr<ShaderResources> resources, std::vector<std::shared_ptr<ShaderResources>>* evicted = nullptr);
    // With `object`, erases the entry only while it still holds that object (a replacement made
    // meanwhile by another worker stays).
    void Remove(const Key& key, const ShaderResources* object = nullptr);
    // Moves the entry to the front of the LRU (a recipe hit uses its template without Find, and a
    // hot template must not age out under the recipes that depend on it); false when no entry.
    bool Touch(const Key& key);
    // Find and Touch calls so far (the [rescache] line: a recipe hit touches instead of finding).
    static std::uint64_t Finds();
    static std::uint64_t Touches();
    void Clear();
    std::size_t Size() const;

private:
    void erase(const Key& key);
    void noteMiss(const Key& key);
    struct KeyHash {
        std::size_t operator()(const Key& key) const noexcept {
            std::uint64_t hash = 14695981039346656037ull;
            for (const auto word : key) hash = (hash ^ word) * 1099511628211ull;
            return static_cast<std::size_t>(hash);
        }
    };
    mutable std::mutex mutex;
    std::list<std::pair<Key, std::shared_ptr<ShaderResources>>> entries;
    std::unordered_map<Key, decltype(entries)::iterator, KeyHash> index;
};

// The process-wide cache the device and the draw path share (keys name the device; a device clears
// it when it goes). Never destroyed: entries belong to the device, not to static teardown.
ResourceCache& SharedResourceCache();
std::string DescribeTextureCaches();

}

#endif
