#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GPUCLOCK_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_GRAPHICS_INCLUDE_GPUCLOCK_HPP

#include <cstdint>

namespace AgcDriver::Graphics::GpuClock {

constexpr std::uint64_t GuestTicksPerSecond = 100'000'000;
constexpr std::uint64_t NanosecondsPerGuestTick = 1'000'000'000 / GuestTicksPerSecond;

struct Mapping {
    std::uint64_t hostOrigin = 0;
    std::uint64_t guestOrigin = 0;
    std::uint64_t factor = 0;
};

std::uint64_t MulShift32(std::uint64_t value, std::uint64_t factor);

Mapping MakeMapping(double timestampPeriodNanoseconds, std::uint64_t hostOrigin, std::uint64_t originNanoseconds);

std::uint64_t ToGuest(const Mapping& mapping, std::uint64_t hostTicks);

}

#endif
