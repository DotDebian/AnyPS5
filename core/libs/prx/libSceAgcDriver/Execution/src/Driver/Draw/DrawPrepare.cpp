#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Diagnostics.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include "Optimization/ResourceProgram.hpp"
#include <cstdlib>
#include <cstring>

namespace AgcDriver::DriverDetail {

namespace {

enum AheadRecheck { RecheckDevice, RecheckMappings, RecheckPending, RecheckDiffers, RecheckUnreadable };

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
            [] { PinWorkerThread("draw front end"); });
    }
    return drawAhead.get();
}

std::shared_ptr<PreparedDraw> Driver::prepareDrawAhead(const QueueState& queue, std::span<const std::uint32_t> packet, const Submission& submission) {
    using Role = ShaderRecompiler::ProgramRole;
    static const bool locked = std::getenv("APS5_LOCKED_DRAW_PREPARE") != nullptr || std::getenv("APS5_DUMP_DRAW_SHADERS") != nullptr || std::getenv("APS5_DUMP_DRAW_SLOT1") != nullptr;
    if (locked || ShaderRecompiler::DebugProbeActive()) return nullptr;
    const GuestMemory::UnhookedReadScope unhooked;
    const auto localDevice = device.Load();
    if (localDevice == nullptr) return nullptr;
    auto prepared = std::make_shared<PreparedDraw>();
    prepared->forgetSerial = GuestMemory::ForgetSerial();
    if (prepared->forgetSerial % 2 != 0) return nullptr;
    prepared->deviceSerial = localDevice->Serial();

    auto drawParameters = Pm4::ResolveDraw(packet, queue);
    if (drawParameters.indirect) return nullptr;
    if (!drawParameters.indexed && (drawParameters.indexCount == 0 || drawParameters.instanceCount == 0)) return nullptr;
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
        prepared->drawKey = drawRegisterKey(queue, *submission.shaders, prepared->deviceSerial);
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
    setMeshIndexWords(programs.front(), graphics, drawParameters);

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

    prepared->shaderMemory = std::make_unique<ShaderMemory>(memory);
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
    for (std::size_t i = 0; i < programs.size(); ++i) {
        if (roles[i] == Role::GeometryBack) continue;
        prepared->resultIndex[i] = results.size();
        const std::string* failed = nullptr;
        results.push_back(compileDrawStage(i, pushCursorBytes, queue, submission, programs, graphics, pixel, prepared->vertexInfos, memory, linked, drawParameters, localDevice, *prepared->shaderMemory, prepared->stageCaptures, recompiled, false, matched, matchedRegions, false, 0, 0, prepared->captures, phaseTiming, phaseMs, &failed));
        if (failed != nullptr) return nullptr;
        const auto& result = results.back();
        if (i == 0) foldDrawOffsets(result, programs.front(), drawParameters);
        require(result.pushConstants.size() <= Graphics::PipelinePushConstantBytes - pushCursorBytes, "stage push constants exceed the pipeline push constant block");
        pushCursorBytes += static_cast<std::uint32_t>(result.pushConstants.size());
    }
    if (GuestMemory::ForgetSerial() != prepared->forgetSerial) return nullptr;
    prepared->decode = std::move(decode);
    prepared->drawParameters = drawParameters;
    prepared->memory = std::move(memory);
    return prepared;
}

ShaderMemory::PendingWrite Driver::pendingOverlap(std::uint64_t address, std::size_t bytes, std::span<std::byte>) {
    PendingView pending;
    pending.Load();
    return pending.Overlaps(address, bytes) ? ShaderMemory::PendingWrite::Sync : ShaderMemory::PendingWrite::None;
}

bool Driver::recheckPreparedDraw(const PreparedDraw& prepared, std::uint64_t deviceSerial) {
    aheadRechecks.fetch_add(1, std::memory_order_relaxed);
    const auto fail = [&](AheadRecheck reason) {
        aheadRecheckFailures[reason].fetch_add(1, std::memory_order_relaxed);
        return false;
    };
    if (prepared.deviceSerial != deviceSerial) return fail(RecheckDevice);
    if (GuestMemory::ForgetSerial() != prepared.forgetSerial) return fail(RecheckMappings);
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
    if (!profile || drawAhead == nullptr) return;
    const auto now = std::chrono::steady_clock::now();
    if (now - aheadReported < std::chrono::seconds(10)) return;
    aheadReported = now;
    const auto totals = drawAhead->Totals();
    const auto failures = [&](AheadRecheck reason) { return static_cast<unsigned long long>(aheadRecheckFailures[reason].load(std::memory_order_relaxed)); };
    std::fprintf(stderr, "[ahead] %llu submissions, %llu draws: %llu prepared ahead, %llu handed over (%llu waited for), %llu the worker's own, %llu after a differing load; loads %llu read ahead, %llu behind, %llu differed; %llu stops; rechecks %llu, failed: device %llu, mappings %llu, pending GPU writes %llu, words differ %llu, unreadable %llu; cached keys: entry hit %llu, missed and recheck failed %llu, adopted %llu\n",
                 static_cast<unsigned long long>(totals.submissions), static_cast<unsigned long long>(totals.draws), static_cast<unsigned long long>(totals.prepared), static_cast<unsigned long long>(totals.handed), static_cast<unsigned long long>(totals.waited), static_cast<unsigned long long>(totals.workerOwn), static_cast<unsigned long long>(totals.poisonedDraws), static_cast<unsigned long long>(totals.loadsAhead), static_cast<unsigned long long>(totals.loadsBehind), static_cast<unsigned long long>(totals.loadMismatches), static_cast<unsigned long long>(totals.stopped), static_cast<unsigned long long>(aheadRechecks.load(std::memory_order_relaxed)), failures(RecheckDevice), failures(RecheckMappings), failures(RecheckPending), failures(RecheckDiffers), failures(RecheckUnreadable), static_cast<unsigned long long>(aheadKnownKeys[0].load(std::memory_order_relaxed)), static_cast<unsigned long long>(aheadKnownKeys[1].load(std::memory_order_relaxed)), static_cast<unsigned long long>(aheadKnownKeys[2].load(std::memory_order_relaxed)));
}

}
