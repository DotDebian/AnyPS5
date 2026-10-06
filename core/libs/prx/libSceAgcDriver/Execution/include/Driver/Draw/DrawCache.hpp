#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWCACHE_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAWCACHE_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchCache.hpp"
#include "prx/libSceAgcDriver/Execution/include/VulkanDevice.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <vector>

namespace AgcDriver::DriverDetail {

struct DrawProgram {
    ShaderRecompiler::ShaderBinary binary;
    std::uint32_t userDataBase;
    std::uint32_t firstUserSgpr = 8;
    std::vector<std::uint32_t> userData;
    std::array<ShaderRecompiler::MemoryRegion, 2> memory;

    std::shared_ptr<const ShaderSnapshot> snapshot;
    std::size_t codeOffset = 0;
};

struct DrawDecode {
    Graphics::State state;
    ShaderRecompiler::ShaderPixelStageInfo pixel;
    std::vector<DrawProgram> programs;
    std::vector<ShaderRecompiler::ProgramRole> roles;
};

struct DrawRecipeRecord {
    std::vector<std::weak_ptr<const DispatchVariant>> stages;
    std::shared_ptr<const DrawRecipe> recipe;
    bool Matches(const std::vector<std::shared_ptr<DispatchVariant>>& variants) const;
    bool Expired() const;
};

struct DrawEntry {

    std::shared_ptr<const DrawDecode> decode;

    std::vector<std::vector<std::shared_ptr<DispatchVariant>>> stages;

    std::atomic<std::shared_ptr<const std::vector<DrawRecipeRecord>>> recipes;
    std::uint64_t touched = 0;
    std::list<std::uint64_t>::iterator order;
};

enum class DrawMiss : std::size_t { FrontDiffering, FragmentDiffering, OtherDiffering, Layout, Gate, Stages, Count };

struct DrawEntryCounters {
    std::uint64_t lookups = 0, absent = 0, hits = 0, stageValidations = 0, stageEqual = 0, variantsCompared = 0, inserts = 0, variantsInserted = 0, variantsEvicted = 0, present = 0, unstable = 0, touches = 0, verifyHits = 0, verifyMismatches = 0;
    std::array<std::uint64_t, static_cast<std::size_t>(DrawMiss::Count)> misses{};
    std::array<std::uint64_t, MaxDispatchVariants> variantHitsByRank{};
    double validateUs = 0;

    std::uint64_t registerKeyLookups = 0, registerKeyHits = 0, decodeSkipped = 0, decodePartial = 0, facadeMismatches = 0, verifyDecodes = 0, verifyDecodeMismatches = 0;
    double keyUs = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

enum class DrawVerdict { Drawn, Nothing, Rejected };

struct StageCapture {
    std::shared_ptr<const ShaderRecompiler::RecompileResult> compiled;
    std::vector<ShaderRecompiler::MemoryRegion> regions;
    std::uint64_t forgetSerial = 0;
    std::uint32_t pushOffset = 0;
};

// What the draw front end asks of Driver::compileDrawStage for the stages of one draw, and what
// the stages measure for it (DrawPrepare.cpp; the worker's own draws pass none).
struct AheadStage {
    // APS5_TRACE_DRAW_AHEAD=1: the times below are taken.
    bool timed = false;
    // APS5_CAPTURE_SCRATCH=1: the capture is made into the thread's own ResourceCapture, which
    // keeps its storage from one stage to the next (it serves the Recompile call and nothing else).
    bool scratch = false;
    // APS5_CAPTURE_MEMO=1: a stage whose inputs repeat and whose walk reads the same words takes
    // the compiled result of the first time (DrawCapture.cpp).
    bool memo = false;
    // APS5_DRAW_AHEAD_SHARED_RESULTS=1 (not for an indirect draw, whose results are patched): the
    // stage's result stays in StageCapture::compiled, shared with every draw of the snapshot, and
    // compileDrawStage returns an empty one instead of a copy.
    bool sharedResult = false;
    // Nanoseconds: the request and its source handle; the capture (the SRT evaluation and its
    // fetches from guest memory, or the memo's replay); the regions; the compiled result's lookup
    // (and its copy).
    std::uint64_t handleNs = 0, captureNs = 0, regionsNs = 0, compileNs = 0;
    std::uint64_t stages = 0, resultMemoHits = 0, resultMemoMisses = 0;
};

// The capture memo's outcomes so far (APS5_CAPTURE_MEMO), for the [drawahead] line: a hit, or why
// not. NewProgram: the stage's program (its source, address and push constant place) was never
// seen. NewUserWords: the program was, with other user words (the SRT pointers of another draw
// or of another frame). Admitted: the same inputs came a second time, so this walk is recorded.
// Displaced: the same, their entry having been overwritten by another stage's. WordsDiffer: the
// inputs are an entry's but the walk read another word (the SRT changed in place). Unreadable:
// a word the walk read is no longer mapped.
enum class CaptureMemoOutcome : std::size_t { Hit, NewProgram, NewUserWords, Admitted, Displaced, WordsDiffer, Unreadable, Count };
std::array<std::uint64_t, static_cast<std::size_t>(CaptureMemoOutcome::Count)> CaptureMemoTotals();

}

#endif
