#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PRESENTPACING_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_PRESENTPACING_HPP

#include <atomic>
#include <cstdint>

namespace AgcDriver {

struct PresentPacing {
    std::atomic<std::uint64_t> releases{0};
    std::atomic<std::uint64_t> lostVblanks{0};
    std::atomic<std::uint64_t> heldVblanks{0};
    std::atomic<std::uint64_t> presentsCrossingVblank{0};
    std::atomic<std::uint64_t> vblankLateMaxNs{0};
};

}

#endif
