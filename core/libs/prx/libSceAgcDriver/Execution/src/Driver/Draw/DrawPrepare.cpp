#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/IndirectDraw.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace AgcDriver::DriverDetail {

namespace {

enum AheadRecheck { RecheckDevice, RecheckMappings, RecheckPending, RecheckDiffers, RecheckUnreadable };

// The pending-write view of the recheck in progress on this thread (recheckPreparedDraw): one
// snapshot of the recorder's pending writes serves every page the prepared draw read, instead of
// one loaded per page by pendingOverlap. A write published while the recheck runs is missed for
// the pages already passed either way; with the shared view it is missed for the remaining ones
// too, the same window as a write published right after the recheck returns.
// APS5_RECHECK_VIEW_PER_PAGE=1 loads a view per page as before.
thread_local const PendingView* recheckView = nullptr;

// The draw front end's switches, all off unless set.
// APS5_TRACE_DRAW_AHEAD=1: the [drawahead] line, every 10 s (and the clock reads it needs).
// APS5_DRAW_AHEAD_THREADS=<n>: n threads prepare draws (1: the walk alone; see DrawAhead.hpp).
bool AheadTraced() {
    static const bool traced = std::getenv("APS5_TRACE_DRAW_AHEAD") != nullptr;
    return traced;
}

std::size_t AheadThreads() {
    static const std::size_t threads = [] {
        const char* text = std::getenv("APS5_DRAW_AHEAD_THREADS");
        return text != nullptr ? static_cast<std::size_t>(std::clamp<unsigned long long>(std::strtoull(text, nullptr, 10), 1ull, 8ull)) : std::size_t{1};
    }();
    return threads;
}

// What the front end's threads measured (APS5_TRACE_DRAW_AHEAD), summed since the start: the
// draws they were asked for and those they prepared, the stages of the latter, and where their
// time went, in nanoseconds. `captureNs` holds `fetchNs` (the fetches from guest memory are made
// inside the capture); the line shows the two apart.
struct AheadTimes {
    std::atomic<std::uint64_t> asked{0}, prepared{0}, stages{0};
    std::atomic<std::uint64_t> totalNs{0}, decodeNs{0}, handleNs{0}, captureNs{0}, fetchNs{0}, regionsNs{0}, compileNs{0};
    std::atomic<std::uint64_t> resultMemoHits{0}, resultMemoMisses{0};
};

AheadTimes& TheAheadTimes() {
    static AheadTimes times;
    return times;
}

}

bool Driver::drawAheadEnabled() {
    static const bool enabled = std::getenv("APS5_NO_DRAW_AHEAD") == nullptr;
    return enabled;
}

DrawAhead* Driver::frontEnd(std::uint32_t queue) {
    if (queue != 0 || !drawAheadEnabled()) return nullptr;
    if (drawAhead == nullptr) {
        drawAhead = std::make_unique<DrawAhead>(
            [this](const QueueState& state, std::span<const std::uint32_t> packet, const Submission& submission) { return prepareDrawAhead(state, packet, submission); },
            [](std::span<const std::uint32_t> packet) {
                const GuestMemory::UnhookedReadScope unhooked;
                return Pm4::ReadRegisterPairs(packet);
            },
            [] { PinWorkerThread("draw front end"); },
            AheadThreads() - 1);
    }
    return drawAhead.get();
}

std::shared_ptr<PreparedDraw> Driver::prepareDrawAhead(const QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission) {
    using Role = ShaderRecompiler::ProgramRole;
    static const bool locked = std::getenv("APS5_LOCKED_DRAW_PREPARE") != nullptr || std::getenv("APS5_DUMP_DRAW_SHADERS") != nullptr || std::getenv("APS5_DUMP_DRAW_SLOT1") != nullptr;
    if (locked || ShaderRecompiler::DebugProbeActive()) return nullptr;
    const GuestMemory::UnhookedReadScope unhooked;
    const bool timed = AheadTraced();
    const auto startedAt = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    if (timed) TheAheadTimes().asked.fetch_add(1, std::memory_order_relaxed);
    const auto localDevice = device.Load();
    if (localDevice == nullptr) return nullptr;
    auto prepared = std::make_shared<PreparedDraw>();
    prepared->forgetSerial = GuestMemory::ForgetSerial();
    if (prepared->forgetSerial % 2 != 0) return nullptr;
    prepared->deviceSerial = localDevice->Serial();

    auto drawParameters = Pm4::ResolveDraw(packet, queue);
    if (drawParameters.indirect && !IndirectDrawAheadEnabled()) return nullptr;
    if (!drawParameters.indirect && !drawParameters.indexed && (drawParameters.indexCount == 0 || drawParameters.instanceCount == 0)) return nullptr;
    static const bool metadataPasses = std::getenv("APS5_NO_METADATA_PASSES") == nullptr;
    if (metadataPasses && (Graphics::DecodeColorMetadataPass(queue) || Graphics::DepthMetadataBlit(queue))) return nullptr;
    {
        const auto targetMask = queue.context.find(0x8e);
        const auto shaderMask = queue.context.find(0x8f);
        const bool colorWrites = targetMask != queue.context.end() && shaderMask != queue.context.end() && (targetMask->second & shaderMask->second) != 0;
        if (!colorWrites && !queue.shader.contains(0x8)) return nullptr;
    }
    if (drawPrecheck() && !Graphics::DrawRejection(queue, drawParameters.indexed).empty()) return nullptr;

    const bool useDrawEntries = drawEntries();
    if (useDrawEntries && !registerKeyEnabled()) return nullptr;
    if (useDrawEntries) {
        prepared->drawKey = drawRegisterKey(queue, *submission.shaders, prepared->deviceSerial, &prepared->stateKey);
        static const bool skipKnown = std::getenv("APS5_DRAW_AHEAD_SKIP_KNOWN") != nullptr;
        std::lock_guard cacheLock(drawCacheMutex);
        prepared->keyKnown = drawCache.contains(prepared->drawKey);
        if (prepared->keyKnown && skipKnown) return nullptr;
    }

    auto decode = decodeDraw(queue, submission);
    const auto& graphics = decode->state;
    const auto& pixel = decode->pixel;
    const auto& roles = decode->roles;
    auto& programs = prepared->programs;
    programs = decode->programs;
    setMeshIndexWords(programs.front(), graphics, MeshIndexParameters(drawParameters));
    if (drawParameters.indirect) ResolveIndirectSgprs(programs, roles, *drawParameters.indirect);

    std::vector<ShaderRecompiler::MemoryRegion> memory;
    std::vector<ShaderRecompiler::LinkedProgram> linked;
    for (std::size_t i = 0; i < programs.size(); ++i) {
        const auto& program = programs[i];
        memory.insert(memory.end(), program.memory.begin(), program.memory.end());
        linked.push_back({roles[i], program.binary, program.userDataBase, program.firstUserSgpr, program.userData});
    }
    prepared->vertexInfos.resize(programs.size());
    prepared->decodeReads.resize(programs.size());
    for (std::size_t i = 0; i < programs.size(); ++i) decodeProgramVertexInfo(programs[i], roles[i], prepared->vertexInfos[i], prepared->decodeReads[i]);

    const auto decodedAt = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    prepared->shaderMemory = std::make_unique<ShaderMemory>(memory);
    prepared->shaderMemory->TimeFetches(timed);
    AheadStage stage;
    stage.timed = timed;
    auto& results = prepared->results;
    results.reserve(programs.size() + 2u);
    prepared->resultIndex.assign(programs.size(), PreparedDraw::NoResult);
    prepared->stageCaptures.resize(programs.size());
    std::vector<bool> recompiled(programs.size(), false);
    const std::vector<std::shared_ptr<DispatchVariant>> matched(programs.size());
    const std::vector<std::vector<ShaderRecompiler::MemoryRegion>> matchedRegions(programs.size());
    std::array<double, DrawDriverPhaseCount> phaseMs{};
    auto phaseLap = std::chrono::steady_clock::time_point{};
    DrawPhaseTiming phaseTiming{false, phaseMs, phaseLap};
    std::uint32_t pushCursorBytes = 0;
    std::string rejected;
    for (std::size_t i = 0; i < programs.size(); ++i) {
        if (roles[i] == Role::GeometryBack) continue;
        auto compiled = compileDrawStage(i, pushCursorBytes, queue, submission, programs, graphics, pixel, prepared->vertexInfos, memory, linked, drawParameters, localDevice, *prepared->shaderMemory, prepared->stageCaptures, recompiled, false, matched, matchedRegions, false, 0, 0, prepared->captures, phaseTiming, phaseMs, rejected, &stage);
        if (!rejected.empty()) return nullptr;
        prepared->resultIndex[i] = results.size();
        results.push_back(std::move(compiled));
        const auto& result = results.back();
        if (i == 0 && drawParameters.indirect) {
            if (IndirectAheadRule(ClassifyIndirectDraw(result, graphics, programs.front(), localDevice->DrawIndirectSupport(), drawParameters), true) != IndirectAhead::Prepare) return nullptr;
        } else if (i == 0) {
            foldDrawOffsets(result, programs.front(), drawParameters);
        }
        require(result.pushConstants.size() <= Graphics::PipelinePushConstantBytes - pushCursorBytes, "stage push constants exceed the pipeline push constant block");
        pushCursorBytes += static_cast<std::uint32_t>(result.pushConstants.size());
    }
    if (GuestMemory::ForgetSerial() != prepared->forgetSerial) return nullptr;
    prepared->decode = std::move(decode);
    prepared->drawParameters = drawParameters;
    prepared->memory = std::move(memory);
    if (timed) {
        const auto nanoseconds = [](std::chrono::steady_clock::duration span) { return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(span).count()); };
        auto& times = TheAheadTimes();
        times.prepared.fetch_add(1, std::memory_order_relaxed);
        times.stages.fetch_add(stage.stages, std::memory_order_relaxed);
        times.totalNs.fetch_add(nanoseconds(std::chrono::steady_clock::now() - startedAt), std::memory_order_relaxed);
        times.decodeNs.fetch_add(nanoseconds(decodedAt - startedAt), std::memory_order_relaxed);
        times.handleNs.fetch_add(stage.handleNs, std::memory_order_relaxed);
        times.captureNs.fetch_add(stage.captureNs, std::memory_order_relaxed);
        times.fetchNs.fetch_add(prepared->shaderMemory->FetchNanoseconds(), std::memory_order_relaxed);
        times.regionsNs.fetch_add(stage.regionsNs, std::memory_order_relaxed);
        times.compileNs.fetch_add(stage.compileNs, std::memory_order_relaxed);
        times.resultMemoHits.fetch_add(stage.resultMemoHits, std::memory_order_relaxed);
        times.resultMemoMisses.fetch_add(stage.resultMemoMisses, std::memory_order_relaxed);
    }
    return prepared;
}

ShaderMemory::PendingWrite Driver::pendingOverlap(std::uint64_t address, std::size_t bytes, std::span<std::byte>) {
    if (recheckView != nullptr) return recheckView->Overlaps(address, bytes) ? ShaderMemory::PendingWrite::Sync : ShaderMemory::PendingWrite::None;
    PendingView pending;
    pending.Load();
    return pending.Overlaps(address, bytes) ? ShaderMemory::PendingWrite::Sync : ShaderMemory::PendingWrite::None;
}

bool Driver::recheckPreparedDraw(const PreparedDraw& prepared, std::uint64_t deviceSerial) {
    const GuestMemory::CollectSiteScope collectSite(GuestMemory::CollectSite::Recheck);
    aheadRechecks.fetch_add(1, std::memory_order_relaxed);
    const auto fail = [&](AheadRecheck reason) {
        aheadRecheckFailures[reason].fetch_add(1, std::memory_order_relaxed);
        return false;
    };
    if (prepared.deviceSerial != deviceSerial) return fail(RecheckDevice);
    if (GuestMemory::ForgetSerial() != prepared.forgetSerial) return fail(RecheckMappings);
    static const bool viewPerPage = std::getenv("APS5_RECHECK_VIEW_PER_PAGE") != nullptr;
    PendingView view;
    struct ViewScope {
        explicit ViewScope(const PendingView* shared) { recheckView = shared; }
        ~ViewScope() { recheckView = nullptr; }
    };
    std::optional<ViewScope> viewScope;
    if (!viewPerPage) {
        view.Load();
        viewScope.emplace(&view);
        GuestMemory::CountTrace(GuestMemory::TraceCount::RecheckViewShared);
    }
    switch (prepared.shaderMemory->RecheckReads(&pendingOverlap)) {
        case ShaderMemory::Recheck::Same: break;
        case ShaderMemory::Recheck::Pending: return fail(RecheckPending);
        case ShaderMemory::Recheck::Differs: return fail(RecheckDiffers);
        case ShaderMemory::Recheck::Unreadable: return fail(RecheckUnreadable);
    }
    std::vector<std::byte> now;
    for (const auto& reads : prepared.decodeReads) {
        for (const auto& read : reads) {
            now.resize(read.bytes.size());
            try {
                GuestMemory::Read(read.address, now, 4);
            } catch (const std::runtime_error&) {
                return fail(RecheckUnreadable);
            }
            if (std::memcmp(now.data(), read.bytes.data(), now.size()) != 0) return fail(RecheckDiffers);
        }
    }
    if (GuestMemory::ForgetSerial() != prepared.forgetSerial) return fail(RecheckMappings);
    return true;
}

void Driver::reportDrawAhead() {
    static const bool profile = std::getenv("APS5_PROFILE_DRAW") != nullptr;
    if ((!profile && !AheadTraced()) || drawAhead == nullptr) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - aheadReported < std::chrono::seconds(10)) return;
    aheadReported = now;
    const auto totals = drawAhead->Totals();
    if (AheadTraced()) {
        // Over the 10 s (this is the queue 0 worker, the only caller): what the worker asked for
        // and got, where the front end's time went per draw it prepared, and what the result memo did.
        struct Seen {
            DrawAhead::Counters counters;
            std::uint64_t asked = 0, prepared = 0, stages = 0, totalNs = 0, decodeNs = 0, handleNs = 0, captureNs = 0, fetchNs = 0, regionsNs = 0, compileNs = 0, resultMemoHits = 0, resultMemoMisses = 0;
        };
        static Seen seen;
        auto& times = TheAheadTimes();
        Seen current;
        current.counters = totals;
        current.asked = times.asked.load(std::memory_order_relaxed);
        current.prepared = times.prepared.load(std::memory_order_relaxed);
        current.stages = times.stages.load(std::memory_order_relaxed);
        current.totalNs = times.totalNs.load(std::memory_order_relaxed);
        current.decodeNs = times.decodeNs.load(std::memory_order_relaxed);
        current.handleNs = times.handleNs.load(std::memory_order_relaxed);
        current.captureNs = times.captureNs.load(std::memory_order_relaxed);
        current.fetchNs = times.fetchNs.load(std::memory_order_relaxed);
        current.regionsNs = times.regionsNs.load(std::memory_order_relaxed);
        current.compileNs = times.compileNs.load(std::memory_order_relaxed);
        current.resultMemoHits = times.resultMemoHits.load(std::memory_order_relaxed);
        current.resultMemoMisses = times.resultMemoMisses.load(std::memory_order_relaxed);
        const auto count = [](std::uint64_t now, std::uint64_t before) { return static_cast<unsigned long long>(now - before); };
        const auto prepared = current.prepared - seen.prepared;
        const auto perDraw = [&](std::uint64_t now, std::uint64_t before) { return prepared != 0 ? static_cast<double>(now - before) / 1000.0 / static_cast<double>(prepared) : 0.0; };
        const auto total = perDraw(current.totalNs, seen.totalNs);
        const auto decode = perDraw(current.decodeNs, seen.decodeNs);
        const auto handle = perDraw(current.handleNs, seen.handleNs);
        const auto fetch = perDraw(current.fetchNs, seen.fetchNs);
        const auto evaluation = std::max(0.0, perDraw(current.captureNs, seen.captureNs) - fetch);
        const auto regions = perDraw(current.regionsNs, seen.regionsNs);
        const auto compile = perDraw(current.compileNs, seen.compileNs);
        const auto taken = current.counters.draws - seen.counters.draws;
        std::fprintf(stderr, "[drawahead] draws (10 s): %llu asked for by the worker, %llu handed over prepared (%llu after a wait), %llu left to the worker; the front end (%zu threads) was asked for %llu and prepared %llu (%llu on its helper threads), avg %.1f us each: state decode %.1f, source handle %.1f, SRT evaluation %.1f, fetches from guest memory %.1f, regions %.1f, compile lookup %.1f, rest %.1f; %llu stages: result memo %llu hits and %llu misses; the worker found %.1f draws ahead of it on average and waited %.1f ms in all; heap calls are not counted\n",
                              count(current.counters.draws, seen.counters.draws), count(current.counters.handed, seen.counters.handed), count(current.counters.waited, seen.counters.waited), count(current.counters.workerOwn, seen.counters.workerOwn), AheadThreads(), count(current.asked, seen.asked), static_cast<unsigned long long>(prepared), count(current.counters.helped, seen.counters.helped), total, decode, handle, evaluation, fetch, regions, compile, std::max(0.0, total - decode - handle - evaluation - fetch - regions - compile),
                              count(current.stages, seen.stages), count(current.resultMemoHits, seen.resultMemoHits), count(current.resultMemoMisses, seen.resultMemoMisses),
                              taken != 0 ? static_cast<double>(current.counters.depthSum - seen.counters.depthSum) / static_cast<double>(taken) : 0.0, static_cast<double>(current.counters.waitNanoseconds - seen.counters.waitNanoseconds) / 1e6);
        seen = current;
    }
    if (!profile) return;
    const auto failures = [&](AheadRecheck reason) { return static_cast<unsigned long long>(aheadRecheckFailures[reason].load(std::memory_order_relaxed)); };
    std::fprintf(stderr, "[ahead] %llu submissions, %llu draws: %llu prepared ahead, %llu handed over (%llu waited for), %llu the worker's own, %llu after a differing load; loads %llu read ahead, %llu behind, %llu differed; %llu stops; rechecks %llu, failed: device %llu, mappings %llu, pending GPU writes %llu, words differ %llu, unreadable %llu; cached keys: entry hit %llu, missed and recheck failed %llu, adopted %llu\n",
                 static_cast<unsigned long long>(totals.submissions), static_cast<unsigned long long>(totals.draws), static_cast<unsigned long long>(totals.prepared), static_cast<unsigned long long>(totals.handed), static_cast<unsigned long long>(totals.waited), static_cast<unsigned long long>(totals.workerOwn), static_cast<unsigned long long>(totals.poisonedDraws), static_cast<unsigned long long>(totals.loadsAhead), static_cast<unsigned long long>(totals.loadsBehind), static_cast<unsigned long long>(totals.loadMismatches), static_cast<unsigned long long>(totals.stopped), static_cast<unsigned long long>(aheadRechecks.load(std::memory_order_relaxed)), failures(RecheckDevice), failures(RecheckMappings), failures(RecheckPending), failures(RecheckDiffers), failures(RecheckUnreadable), static_cast<unsigned long long>(aheadKnownKeys[0].load(std::memory_order_relaxed)), static_cast<unsigned long long>(aheadKnownKeys[1].load(std::memory_order_relaxed)), static_cast<unsigned long long>(aheadKnownKeys[2].load(std::memory_order_relaxed)));
}

}
