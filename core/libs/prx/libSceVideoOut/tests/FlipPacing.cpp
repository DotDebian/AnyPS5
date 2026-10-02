#include "prx/libSceVideoOut/include/FlipPacing.hpp"
#include <cstdio>
#include <cstdlib>
#include <limits>

static void Require(bool value, const char* what) {
    if (value) return;
    std::fprintf(stderr, "flip pacing test failed: %s\n", what);
    std::abort();
}

int main() {
    Require(FlipTargetVblank(0, 0) == 1, "first vsync flip waits for the first vblank");
    Require(FlipTargetVblank(10, 0) == 11 && FlipTargetVblank(10, 1) == 12 && FlipTargetVblank(10, 2) == 13, "flip rate interval");
    bool overflow = false;
    try { FlipTargetVblank(std::numeric_limits<std::uint64_t>::max(), 0); }
    catch (const std::overflow_error&) { overflow = true; }
    Require(overflow, "interval overflow is rejected");
    bool negative = false;
    try { FlipTargetVblank(0, -1); }
    catch (const std::invalid_argument&) { negative = true; }
    Require(negative, "negative flip rate is rejected");

    Require(LostVblanks(11, 10, 9, 0) == 0, "a flip queued ahead of its vblank released at it loses nothing");
    Require(LostVblanks(12, 10, 9, 0) == 1, "a flip queued ahead of vblank 11 but released at 12 lost one");
    Require(LostVblanks(14, 10, 14, 0) == 0, "a flip queued late and released at once loses nothing");
    Require(LostVblanks(15, 10, 14, 0) == 1, "a late flip released one vblank after it was queued lost one");
    Require(LostVblanks(13, 10, 9, 2) == 0 && LostVblanks(14, 10, 9, 2) == 1, "flip rate moves the earliest vblank");
    Require(HeldVblanks(11, 10, 9, 0) == 0 && HeldVblanks(12, 10, 9, 0) == 1, "a target past the earliest vblank holds the flip back");
    Require(HeldVblanks(12, 10, 12, 0) == 0 && HeldVblanks(13, 10, 12, 0) == 1, "a late flip is held only past the vblank it was queued at");

    std::uint64_t lastFlipVblank = 0;
    std::uint64_t lastPresentVblank = 0;
    std::uint64_t lost = 0;
    std::uint64_t heldByPresent = 0;
    struct Flip { std::uint64_t ready, released, afterPresent; };
    for (const Flip flip : {Flip{0, 1, 2}, Flip{1, 2, 3}, Flip{2, 3, 3}, Flip{3, 4, 5}}) {
        Require(flip.released >= FlipTargetVblank(lastFlipVblank, 0), "a flip is never released before its target");
        if (FlipTargetVblank(lastPresentVblank, 0) > flip.released) ++heldByPresent;
        lost += LostVblanks(flip.released, lastFlipVblank, flip.ready, 0);
        lastFlipVblank = flip.released;
        lastPresentVblank = flip.afterPresent;
    }
    Require(lost == 0 && lastFlipVblank == 4, "presents that cross the next vblank do not move later flips");
    Require(heldByPresent == 2, "pacing from the vblank after the present would hold two of these flips back");
    return 0;
}
