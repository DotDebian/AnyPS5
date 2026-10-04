#include "prx/libSceAgcDriver/Graphics/include/GpuClock.hpp"
#include <cmath>
#include <stdexcept>

namespace AgcDriver::Graphics::GpuClock {

std::uint64_t MulShift32(std::uint64_t value, std::uint64_t factor) {
    const std::uint64_t v0 = value & 0xffffffffu, v1 = value >> 32u;
    const std::uint64_t f0 = factor & 0xffffffffu, f1 = factor >> 32u;
    return ((v0 * f0) >> 32u) + v0 * f1 + v1 * f0 + ((v1 * f1) << 32u);
}

Mapping MakeMapping(double timestampPeriodNanoseconds, std::uint64_t hostOrigin, std::uint64_t originNanoseconds) {
    if (!(timestampPeriodNanoseconds > 0.0) || !std::isfinite(timestampPeriodNanoseconds)) throw std::runtime_error("GPU timestamps: the Vulkan timestamp period is not positive");
    const double factor = std::round(timestampPeriodNanoseconds / static_cast<double>(NanosecondsPerGuestTick) * 4294967296.0);
    if (factor < 1.0 || factor >= 18446744073709551616.0) throw std::runtime_error("GPU timestamps: the Vulkan timestamp period does not fit the guest clock conversion");
    return Mapping{hostOrigin, originNanoseconds / NanosecondsPerGuestTick, static_cast<std::uint64_t>(factor)};
}

std::uint64_t ToGuest(const Mapping& mapping, std::uint64_t hostTicks) {
    if (hostTicks >= mapping.hostOrigin) return mapping.guestOrigin + MulShift32(hostTicks - mapping.hostOrigin, mapping.factor);
    return mapping.guestOrigin - MulShift32(mapping.hostOrigin - hostTicks, mapping.factor);
}

}
