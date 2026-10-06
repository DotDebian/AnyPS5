#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Report.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Shaders/ShaderRegistry.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "Optimization/ResourceProgram.hpp"
#include "prx/libc/include/HostThreadLocal.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace AgcDriver::DriverDetail {

namespace {

// The capture memo (APS5_CAPTURE_MEMO=1, the draw front end's stages only).
//
// Every stage of every draw walks its SRT (ShaderMemory::Capture) and then looks its compiled
// result up by the snapshot the walk produced: the result memo of the recompiler is keyed by the
// walk's output, so nothing ever spares the walk. The walk is a function of the stage's plan (its
// source entry), of its user data and of the words it reads, in the order it reads them
// (ShaderMemory::ReadEvent), and the compiled result is a function of the walk's output, of the
// stage's push constant place and of its vertex stage info. So a stage whose inputs are an
// entry's and whose recorded reads replay with the same answers takes the entry's compiled
// result: the replay marks exactly the words the walk would have read (the regions the draw's
// recheck and cache go by are the same), and neither the walk nor the lookup is made.
//
// It cannot serve a draw whose user words changed, which is every draw whose SRT lies in memory
// the title allocates anew each frame: the walk's output then holds other addresses, and only a
// walk can tell what they are. The outcomes on the [drawahead] line say how many draws that is.
//
// An entry is made the second time the same inputs are seen (a table of the last keys seen
// decides, without allocating), so stages whose user words never repeat cost a hash and two
// probes. Entries are immutable and replaced whole; the table is direct mapped.
struct CaptureMemoEntry {
    const void* source = nullptr;
    std::uint64_t codeAddress = 0;
    std::uint32_t pushOffset = 0;
    std::uint32_t pushBytes = 0;
    std::vector<std::uint32_t> userData;
    std::optional<ShaderRecompiler::ShaderVertexStageInfo> vertex;
    std::vector<ShaderMemory::ReadEvent> reads;
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
};

struct CaptureMemo {
    static constexpr std::size_t Entries = 1u << 14u;
    static constexpr std::size_t SeenInputs = 1u << 16u;
    static constexpr std::size_t SeenPrograms = 1u << 12u;
    // A walk longer than this (a bindless table's scan) is not recorded.
    static constexpr std::size_t MaxReads = 4096;
    std::array<std::atomic<std::shared_ptr<const CaptureMemoEntry>>, Entries> entries;
    std::array<std::atomic<std::uint64_t>, SeenInputs> seenInputs{};
    std::array<std::atomic<std::uint64_t>, SeenPrograms> seenPrograms{};
    std::array<std::atomic<std::uint64_t>, static_cast<std::size_t>(CaptureMemoOutcome::Count)> outcomes{};
};

CaptureMemo& TheCaptureMemo() {
    static CaptureMemo memo;
    return memo;
}

// The thread's capture and read log (see AheadStage::scratch and the memo above).
struct StageScratch {
    ShaderRecompiler::ResourceCapture capture;
    std::vector<ShaderMemory::ReadEvent> reads;
};

StageScratch& ThreadStageScratch() {
    struct StageScratchTag {};
    return HostThreadLocal<StageScratch, StageScratchTag>();
}

}

std::array<std::uint64_t, static_cast<std::size_t>(CaptureMemoOutcome::Count)> CaptureMemoTotals() {
    std::array<std::uint64_t, static_cast<std::size_t>(CaptureMemoOutcome::Count)> totals{};
    auto& memo = TheCaptureMemo();
    for (std::size_t i = 0; i < totals.size(); ++i) totals[i] = memo.outcomes[i].load(std::memory_order_relaxed);
    return totals;
}

ShaderRecompiler::RecompileResult Driver::compileDrawStage(std::size_t i, std::uint32_t pushOffset, const QueueState& queue, const Submission& submission, const std::vector<DrawProgram>& programs, const Graphics::State& graphics, const ShaderRecompiler::ShaderPixelStageInfo& pixel, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, std::vector<ShaderRecompiler::MemoryRegion>& memory, const std::vector<ShaderRecompiler::LinkedProgram>& linked, const Pm4::DrawParameters& drawParameters, const std::shared_ptr<VulkanDevice>& localDevice, ShaderMemory& shaderMemory, std::vector<StageCapture>& stageCaptures, std::vector<bool>& recompiled, bool drawHit, const std::vector<std::shared_ptr<DispatchVariant>>& matched, const std::vector<std::vector<ShaderRecompiler::MemoryRegion>>& matchedRegions, bool profile, std::uint64_t dumpTarget, std::uint64_t dumpSlot1, std::uint64_t& captures, DrawPhaseTiming& phaseTiming, std::array<double, DrawDriverPhaseCount>& phaseMs, std::string& rejected, AheadStage* ahead) {
    using Stage = ShaderRecompiler::ShaderStage;
    phaseTiming.Phase(DrawRowVectors);
    // The front end's times (AheadStage): each lap goes to the part that just ended.
    const bool timed = ahead != nullptr && ahead->timed;
    auto lapStarted = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto lap = [&](std::uint64_t AheadStage::* part) {
        if (!timed) return;
        const auto now = std::chrono::steady_clock::now();
        ahead->*part += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now - lapStarted).count());
        lapStarted = now;
    };
    if (ahead != nullptr) ++ahead->stages;
    const auto& program = programs[i];
    const auto waveSize = program.binary.stage == Stage::Fragment ? graphics.stages.fragmentWaveSize : graphics.stages.vertexWaveSize;
    ShaderRecompiler::RecompileRequest request{
        program.binary,
        {waveSize, program.firstUserSgpr, program.userData, std::nullopt, program.binary.stage == Stage::Fragment ? std::optional(pixel) : std::nullopt, vertexInfos[i], memory},
        localDevice->Target(),
        {0, 0, pushOffset, (graphics.stages.mesh ? ShaderRecompiler::MeshDrawPushOffsetBytes : Graphics::PipelinePushConstantBytes) - pushOffset},
        ShaderRecompiler::GraphicsCompileContext{program.firstUserSgpr, linked, graphics.stages.mesh, graphics.stages.tessellation, {drawParameters.indexAddress, drawParameters.indexCount, drawParameters.indexSize, drawParameters.instanceCount}}
    };
    const auto waitedBefore = traceCapSync() || profile ? Graphics::Recorder::ThreadWaitedMs() : 0.0;
    const std::string* poisoned = nullptr;
    const auto handle = SourceHandleFor(*program.snapshot, program.codeOffset, localDevice->Serial(), request, false, FailureMemo() ? &poisoned : nullptr);
    if (handle == nullptr && poisoned != nullptr) {
        rejected = *poisoned;
        return {};
    }
    auto& stageCapture = stageCaptures[i];
    stageCapture.forgetSerial = GuestMemory::ForgetSerial();
    stageCapture.pushOffset = pushOffset;
    lap(&AheadStage::handleNs);
    // The capture memo: `memoized` is the entry's result on a hit; `record` on a miss that is to
    // make an entry (the walk's steps then go to the thread's log).
    std::shared_ptr<const ShaderRecompiler::RecompileResult> memoized;
    bool record = false;
    std::size_t memoSlot = 0;
    const void* memoSource = nullptr;
    const bool memo = ahead != nullptr && ahead->memo && handle != nullptr;
    const auto sameInputs = [&](const CaptureMemoEntry& entry) {
        if (entry.source != memoSource || entry.codeAddress != program.binary.codeAddress || entry.pushOffset != request.layout.pushConstantOffsetBytes || entry.pushBytes != request.layout.pushConstantSizeBytes) return false;
        if (entry.userData.size() != program.userData.size() || !std::equal(entry.userData.begin(), entry.userData.end(), program.userData.begin())) return false;
        if (entry.vertex.has_value() != vertexInfos[i].has_value()) return false;
        return !entry.vertex.has_value() || sameVertexInfo(*entry.vertex, *vertexInfos[i]);
    };
    std::shared_ptr<const ShaderRecompiler::ResourceCapture> ownedCapture;
    const ShaderRecompiler::ResourceCapture* capture = nullptr;
    {
        const SampledReadScope sampling(evidenceReads);
        if (memo) {
            auto& table = TheCaptureMemo();
            memoSource = handle->source.get();
            std::uint64_t programKey = 0xcbf29ce484222325ull;
            const auto mix = [](std::uint64_t& key, std::uint64_t value) {
                key ^= value;
                key *= 0x100000001b3ull;
            };
            mix(programKey, reinterpret_cast<std::uintptr_t>(memoSource));
            mix(programKey, program.binary.codeAddress);
            mix(programKey, (static_cast<std::uint64_t>(request.layout.pushConstantOffsetBytes) << 32u) | request.layout.pushConstantSizeBytes);
            auto inputKey = programKey;
            mix(inputKey, program.userData.size());
            for (const auto word : program.userData) mix(inputKey, word);
            // Zero is an empty slot of the tables of keys seen.
            if (programKey == 0) programKey = 1;
            if (inputKey == 0) inputKey = 1;
            memoSlot = static_cast<std::size_t>(inputKey & (CaptureMemo::Entries - 1));
            auto outcome = CaptureMemoOutcome::Hit;
            const auto entry = table.entries[memoSlot].load(std::memory_order_acquire);
            if (entry != nullptr && sameInputs(*entry)) {
                switch (shaderMemory.ReplayReads(entry->reads)) {
                    case ShaderMemory::Replay::Same:
                        memoized = entry->compiled;
                        break;
                    case ShaderMemory::Replay::Differs:
                        outcome = CaptureMemoOutcome::WordsDiffer;
                        record = true;
                        break;
                    case ShaderMemory::Replay::Unreadable:
                        outcome = CaptureMemoOutcome::Unreadable;
                        record = true;
                        break;
                }
            } else if (table.seenInputs[static_cast<std::size_t>((inputKey >> 14u) & (CaptureMemo::SeenInputs - 1))].exchange(inputKey, std::memory_order_relaxed) == inputKey) {
                outcome = entry != nullptr ? CaptureMemoOutcome::Displaced : CaptureMemoOutcome::Admitted;
                record = true;
            } else {
                outcome = table.seenPrograms[static_cast<std::size_t>(programKey & (CaptureMemo::SeenPrograms - 1))].exchange(programKey, std::memory_order_relaxed) == programKey ? CaptureMemoOutcome::NewUserWords : CaptureMemoOutcome::NewProgram;
            }
            table.outcomes[static_cast<std::size_t>(outcome)].fetch_add(1, std::memory_order_relaxed);
        }
        if (memoized == nullptr) {
            // The thread's storage, for the front end's stages that ask for it only.
            const bool keptCapture = ahead != nullptr && ahead->scratch && handle != nullptr;
            auto* scratch = record || keptCapture ? &ThreadStageScratch() : nullptr;
            struct LogScope {
                ShaderMemory& memory;
                ~LogScope() { memory.LogReads(nullptr); }
            } logScope{shaderMemory};
            if (record) {
                scratch->reads.clear();
                shaderMemory.LogReads(&scratch->reads);
            }
            if (keptCapture) {
                shaderMemory.CaptureInto(request, *handle, scratch->capture);
                capture = &scratch->capture;
            } else {
                ownedCapture = shaderMemory.Capture(request, handle.get());
                capture = ownedCapture.get();
            }
        }
    }
    lap(&AheadStage::captureNs);

    stageCapture.regions = shaderMemory.TakeRecentRegions();
    recompiled[i] = true;
    shaderMemory.Regions(memory);

    if (drawHit) {
        for (std::size_t j = 0; j < programs.size(); ++j) {
            if (matched[j] != nullptr && !recompiled[j]) memory.insert(memory.end(), matchedRegions[j].begin(), matchedRegions[j].end());
        }
    }
    request.context.memory = memory;
    if (traceCapSync()) traceCapture("draw-capture", program.binary.codeAddress, submission.queue, memory, Graphics::Recorder::ThreadWaitedMs() - waitedBefore);
    if (profile) {
        ++captures;
        phaseTiming.Phase(DrawRowCapture);

        const auto waited = std::min(Graphics::Recorder::ThreadWaitedMs() - waitedBefore, phaseMs[DrawRowCapture]);
        phaseMs[DrawRowCapture] -= waited;
        phaseMs[DrawRowCaptureHookWaits] += waited;
    }
    if (dumpTarget != 0) {

        const auto slot0 = (static_cast<std::uint64_t>(readRegister(queue.context, 0x390)) << 40u) | (static_cast<std::uint64_t>(readRegister(queue.context, 0x318)) << 8u);
        if ((graphics.hasColorTarget && graphics.color.address == dumpTarget) || slot0 == dumpTarget) static_cast<void>(dumpRequest(program.binary.codeAddress, request));
    }
    if (dumpSlot1 != 0) {
        const auto value = [&](std::uint32_t offset) -> std::uint64_t { const auto it = queue.context.find(offset); return it == queue.context.end() ? 0u : it->second; };
        const auto slot1 = (value(0x391) << 40u) | (value(0x327) << 8u);
        if (slot1 == dumpSlot1) {
            static_cast<void>(dumpRequest(program.binary.codeAddress, request));
            if (std::FILE* file = std::fopen("draw_slot1.regs", "w")) {
                for (const auto& [offset, value] : queue.context) std::fprintf(file, "context %x %08x\n", offset, value);
                for (const auto& [offset, value] : queue.userConfig) std::fprintf(file, "uconfig %x %08x\n", offset, value);
                for (const auto& [offset, value] : queue.shader) std::fprintf(file, "shader %x %08x\n", offset, value);
                std::fclose(file);
            }
        }
    }

    static const bool reuseCapture = std::getenv("APS5_NO_CAPTURE_REUSE") == nullptr;
    phaseTiming.Phase(DrawRowCapture);
    lap(&AheadStage::regionsNs);

    if (memoized != nullptr) {
        stageCapture.compiled = std::move(memoized);
    } else {
        bool resultMemoHit = false;
        stageCapture.compiled = reuseCapture ? ShaderRecompiler::Recompile(request, *capture, &resultMemoHit) : std::make_shared<const ShaderRecompiler::RecompileResult>(ShaderRecompiler::Recompile(request));
        if (ahead != nullptr) ++(resultMemoHit ? ahead->resultMemoHits : ahead->resultMemoMisses);
        if (record && stageCapture.compiled != nullptr) {
            // The walk's steps and what they led to, for the next stage with these inputs.
            auto& scratch = ThreadStageScratch();
            if (scratch.reads.size() <= CaptureMemo::MaxReads) {
                auto entry = std::make_shared<CaptureMemoEntry>();
                entry->source = memoSource;
                entry->codeAddress = program.binary.codeAddress;
                entry->pushOffset = request.layout.pushConstantOffsetBytes;
                entry->pushBytes = request.layout.pushConstantSizeBytes;
                entry->userData = program.userData;
                entry->vertex = vertexInfos[i];
                entry->reads.assign(scratch.reads.begin(), scratch.reads.end());
                entry->compiled = stageCapture.compiled;
                TheCaptureMemo().entries[memoSlot].store(std::move(entry), std::memory_order_release);
            }
        }
    }
    if (ahead != nullptr && ahead->sharedResult) {
        phaseTiming.Phase(DrawRowRecompile);
        lap(&AheadStage::compileNs);
        return {};
    }
    ShaderRecompiler::RecompileResult result = *stageCapture.compiled;
    phaseTiming.Phase(DrawRowRecompile);
    lap(&AheadStage::compileNs);
    return result;
}

void Driver::cacheDrawStages(bool useDrawEntries, bool drawHit, const Pm4::DrawParameters& drawParameters, const std::optional<Graphics::IndirectDrawPath>& indirectCpu, const std::vector<DrawProgram>& programs, const std::vector<StageCapture>& stageCaptures, const std::vector<std::optional<ShaderRecompiler::ShaderVertexStageInfo>>& vertexInfos, const std::vector<std::vector<Graphics::DecodeRead>>& decodeReads, bool verifyHit, const std::vector<std::shared_ptr<DispatchVariant>>& matched, std::vector<std::shared_ptr<DispatchVariant>>& fresh, std::uint64_t drawKey, bool registerKey, const std::shared_ptr<const DrawDecode>& decode, DrawPhaseTiming& phaseTiming) {
    if (useDrawEntries && !drawHit && !(drawParameters.indirect && indirectCpu) && (verifyHit || admitDrawKey(drawKey))) {
        phaseTiming.Phase(DrawRowVectors);
        std::uint64_t unstable = 0, mismatches = 0;
        for (std::size_t i = 0; i < programs.size(); ++i) {
            const auto& stageCapture = stageCaptures[i];
            if (stageCapture.compiled == nullptr) continue;
            auto variant = std::make_shared<DispatchVariant>();
            variant->compiled = stageCapture.compiled;
            variant->shader = programs[i].snapshot;
            variant->forgetSerial = stageCapture.forgetSerial;
            variant->pushOffset = stageCapture.pushOffset;
            if (vertexInfos[i]) variant->vertexInfo = std::make_shared<const ShaderRecompiler::ShaderVertexStageInfo>(*vertexInfos[i]);

            std::vector<ShaderRecompiler::MemoryRegion> regions(stageCapture.regions.begin(), stageCapture.regions.end());
            for (const auto& read : decodeReads[i]) regions.push_back({read.address, std::as_bytes(std::span(read.bytes))});
            std::stable_sort(regions.begin(), regions.end(), [](const ShaderRecompiler::MemoryRegion& a, const ShaderRecompiler::MemoryRegion& b) { return a.guestAddress < b.guestAddress; });
            for (const auto& region : regions) {
                variant->runs.emplace_back(region.guestAddress, region.guestAddress + region.bytes.size());
                const auto count = region.bytes.size() / sizeof(std::uint32_t);
                const auto offset = variant->words.size();
                variant->words.resize(offset + count);
                std::memcpy(variant->words.data() + offset, region.bytes.data(), count * sizeof(std::uint32_t));
            }
            if (verifyHit && matched[i] != nullptr && (matched[i]->runs != variant->runs || matched[i]->words != variant->words)) {
                ++mismatches;
                static std::atomic<std::uint64_t> reports{0};
                if (reports.fetch_add(1) < 20) AgcDriver::ReportLine("[draw-cache] verify: stage %zu (program 0x%llx) of a hit captured differently: %zu runs / %zu words matched, %zu / %zu fresh\n", i, static_cast<unsigned long long>(programs[i].binary.codeAddress), matched[i]->runs.size(), matched[i]->words.size(), variant->runs.size(), variant->words.size());
            }
            if (insertCompare()) {
                const GuestMemory::ReadSiteScope site(GuestMemory::ReadSite::DrawCache);
                if (!captureStable(stageCapture.regions)) {
                    ++unstable;
                    continue;
                }
            }
            fresh[i] = std::move(variant);
        }
        insertDrawEntry(drawKey, fresh, registerKey ? decode : nullptr);
        if (unstable != 0 || mismatches != 0) {
            std::lock_guard cacheLock(drawCacheMutex);
            drawEntryCounters.unstable += unstable;
            drawEntryCounters.verifyMismatches += mismatches;
        }
        phaseTiming.Phase(DrawRowKeyLookupValidate);
    }
}

}
