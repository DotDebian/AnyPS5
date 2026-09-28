#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_PRECISEWAIT_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_PRECISEWAIT_HPP

#include <chrono>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

// std::condition_variable::wait_for on this MinGW toolchain cannot return in under about one system tick,
// whatever the budget says: measured 15.6 ms average for a 0 us wait and for a 1 us wait alike, and raising
// the process timer resolution to 0.5 ms with NtSetTimerResolution does not change it. A high-resolution
// waitable timer honours the same request in about 0.5 ms instead.
//
// This matters for any guest-side poll that requests a sub-tick wait budget: without this helper, such a
// poll runs at the OS tick rate rather than at the requested budget, which can turn a tight wait loop into
// a multi-millisecond stall per iteration.
//
// Callers must hold no lock that the awaited event's producer needs: the point of the helper is to wait
// precisely and cheaply, so the caller re-tests its own condition afterwards.
inline constexpr unsigned long long kSubTickWaitUs = 2000;

inline void PreciseSleepUs(unsigned long long micros) {
    if (micros == 0) {
        return;
    }
#ifdef _WIN32
    struct Timer {
        HANDLE handle = CreateWaitableTimerExW(nullptr, nullptr,
            CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        ~Timer() {
            if (handle != nullptr) {
                CloseHandle(handle);
            }
        }
    };
    thread_local Timer timer;
    if (timer.handle != nullptr) {
        LARGE_INTEGER due{};
        due.QuadPart = -static_cast<LONGLONG>(micros * 10ULL);  // relative, 100 ns units
        if (SetWaitableTimerEx(timer.handle, &due, 0, nullptr, nullptr, nullptr, 0) != FALSE) {
            WaitForSingleObject(timer.handle, INFINITE);
            return;
        }
    }
#endif
    std::this_thread::sleep_for(std::chrono::microseconds(micros));
}

#endif
