#include "prx/libSceAgcDriver/Graphics/include/GpuClock.hpp"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace AgcDriver::Graphics::GpuClock;

namespace {

int failures = 0;

void Expect(bool condition, const char* what) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

template <typename TAction>
void ExpectFailure(TAction action, const char* text) {
    try {
        action();
    } catch (const std::exception& error) {
        if (std::string(error.what()).find(text) != std::string::npos) return;
        std::fprintf(stderr, "FAIL: unexpected message: %s\n", error.what());
        ++failures;
        return;
    }
    std::fprintf(stderr, "FAIL: no exception for %s\n", text);
    ++failures;
}

std::uint64_t next(std::uint64_t& seed) {
    seed ^= seed << 13u;
    seed ^= seed >> 7u;
    seed ^= seed << 17u;
    return seed;
}

void testMulShift() {
    Expect(MulShift32(0, ~0ull) == 0 && MulShift32(~0ull, 0) == 0, "a zero operand gives zero");
    Expect(MulShift32(1ull << 32u, 1ull << 32u) == 1ull << 32u, "one in 32.32 is the identity");
    Expect(MulShift32(0xffffffffull, 0xffffffffull) == 0xfffffffeull, "the low product's carry is kept");
#ifdef __SIZEOF_INT128__
    std::uint64_t seed = 0x9e3779b97f4a7c15ull;
    int mismatches = 0;
    for (int round = 0; round < 100000; ++round) {
        const auto factor = next(seed) >> (next(seed) % 64u);
        const auto value = next(seed) >> (next(seed) % 64u);
        const auto exact = (static_cast<unsigned __int128>(value) * factor) >> 32u;
        if (exact >> 64u != 0) continue;
        if (MulShift32(value, factor) != static_cast<std::uint64_t>(exact)) ++mismatches;
    }
    Expect(mismatches == 0, "the 32.32 product matches 128-bit arithmetic");
#endif
}

void testUnits() {
    const auto nanosecond = MakeMapping(1.0, 0, 0);
    Expect(ToGuest(nanosecond, 1'000'000'000ull) == GuestTicksPerSecond, "one second of 1 ns ticks is 100,000,000 guest ticks");
    Expect(ToGuest(nanosecond, 16'666'667ull) == 1'666'666ull, "one frame of 1 ns ticks");
    const auto tenNanoseconds = MakeMapping(10.0, 0, 0);
    Expect(tenNanoseconds.factor == 1ull << 32u, "10 ns ticks are guest ticks");
    Expect(ToGuest(tenNanoseconds, 123'456'789ull) == 123'456'789ull, "10 ns ticks map one to one");
    const auto forty = MakeMapping(40.0, 0, 0);
    Expect(ToGuest(forty, 25'000'000ull) == GuestTicksPerSecond, "one second of 40 ns ticks");
    const auto crystal = MakeMapping(1e9 / 19.2e6, 0, 0);
    const auto second = ToGuest(crystal, 19'200'000ull);
    Expect(second >= GuestTicksPerSecond - 1 && second <= GuestTicksPerSecond + 1, "one second of 19.2 MHz ticks");
    const std::uint64_t day = 86'400ull * 1'000'000'000ull;
    const auto daily = ToGuest(nanosecond, day);
    constexpr std::uint64_t guestDay = 8'640'000'000'000ull;
    Expect(daily >= guestDay - guestDay / 400'000'000ull && daily <= guestDay + guestDay / 400'000'000ull, "a day of 1 ns ticks keeps the rate within the 32.32 factor's rounding (2.5e-9)");
    ExpectFailure([] { MakeMapping(0.0, 0, 0); }, "not positive");
    ExpectFailure([] { MakeMapping(-1.0, 0, 0); }, "not positive");
    ExpectFailure([] { MakeMapping(1e30, 0, 0); }, "does not fit");
}

void testOrigin() {
    const auto mapping = MakeMapping(1.0, 5'000'000ull, 7'000'000'000ull);
    Expect(mapping.guestOrigin == 700'000'000ull, "the calibrated host time becomes the guest origin");
    Expect(ToGuest(mapping, 5'000'000ull) == 700'000'000ull, "the calibration point maps to the origin");
    Expect(ToGuest(mapping, 5'000'000ull + 1'000'000'000ull) == 700'000'000ull + GuestTicksPerSecond, "a second after calibration");
    Expect(ToGuest(mapping, 5'000'000ull - 1'000'000ull) == 700'000'000ull - 100'000ull, "a host stamp before calibration maps before the origin");
}

void testMonotonic() {
    for (const double period : {1.0, 10.0, 40.0, 1e9 / 19.2e6, 0.25}) {
        const auto mapping = MakeMapping(period, 1'000'000ull, 12'345'678'900ull);
        std::uint64_t seed = 0x2545f4914f6cdd1dull;
        std::uint64_t host = 900'000ull, previous = ToGuest(mapping, host);
        bool monotonic = true;
        for (int step = 0; step < 100000; ++step) {
            host += next(seed) % 5000u;
            const auto guest = ToGuest(mapping, host);
            if (guest < previous) monotonic = false;
            previous = guest;
        }
        Expect(monotonic, "guest timestamps never go backwards while host timestamps advance");
    }
}

}

int main() {
    testMulShift();
    testUnits();
    testOrigin();
    testMonotonic();
    if (failures != 0) {
        std::fprintf(stderr, "%d GPU clock checks failed\n", failures);
        return 1;
    }
    std::printf("GPU clock tests passed\n");
    return 0;
}
