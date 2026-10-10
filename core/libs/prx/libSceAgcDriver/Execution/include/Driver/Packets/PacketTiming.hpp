#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_PACKETTIMING_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_PACKETTIMING_HPP

#include "prx/libSceAgcDriver/Execution/include/Driver/Dispatch/DispatchTiming.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>

namespace AgcDriver::DriverDetail {

struct WorkerProfile {
    double dispatchMs = 0;
    double drawMs = 0;
    double waitMs = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point reported = start;
};

struct PacketProfile {
    std::map<std::uint32_t, std::pair<std::uint64_t, double>> byOpcode;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
    std::uint64_t submissions = 0;

    double flushMs = 0;

    std::array<std::pair<std::uint64_t, double>, static_cast<std::size_t>(DispatchOutcome::Count)> dispatchOutcomes{};
};

// APS5_PROFILE_PACKETS=1: the worker's time by packet kind, one [packet-times] line per queue
// every 10 s. Two clock reads per packet and nothing else, unlike APS5_PROFILE_DRAW's rows, so
// the frame rate under it is the frame rate.
struct LightPacketTimes {
    std::map<std::uint32_t, std::pair<std::uint64_t, double>> byOpcode;
    double flushMs = 0;
    std::chrono::steady_clock::time_point lastReport = std::chrono::steady_clock::now();
};

struct LightPacketTimer {
    bool enabled; std::uint32_t key; std::uint32_t queue; LightPacketTimes& times; std::chrono::steady_clock::time_point start;
    ~LightPacketTimer();
};

struct PacketTimer {
    bool enabled; std::uint32_t key; std::uint32_t queue; PacketProfile& profile; std::chrono::steady_clock::time_point start;
    ~PacketTimer();
};

}

#endif
