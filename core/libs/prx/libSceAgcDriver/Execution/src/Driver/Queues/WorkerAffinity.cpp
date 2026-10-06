#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Report.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/WorkerAffinity.hpp"
#include "prx/libc/include/CpuTopology.hpp"
#include <cstdlib>
#include <cstring>
#include <mutex>
#ifdef _WIN32
#include <windows.h>
#endif

namespace AgcDriver::DriverDetail {

std::uint64_t WorkerAffinityMask() {
    static const std::uint64_t mask = [] {
        if (std::getenv("APS5_NO_WORKER_AFFINITY") != nullptr) return std::uint64_t{0};
        const auto requested = CpuTopology::MaskFromEnvironment("APS5_WORKER_AFFINITY_MASK");
        if (requested != 0) return requested;
#ifndef __linux__
        if (std::getenv("APS5_WORKER_AFFINITY") == nullptr) return std::uint64_t{0};
#endif
        const auto& layout = CpuTopology::Get();
        return layout.hybrid ? layout.performant : std::uint64_t{0};
    }();
    return mask;
}

// APS5_WORKER_PRIORITY=<n> (local, not for upstream): SetThreadPriority level of the driver's pinned
// threads (queue workers, presenter, draw front end): 1 ABOVE_NORMAL, 2 HIGHEST, 15 TIME_CRITICAL.
// APS5_PROCESS_PRIORITY=above|high: the process priority class, set once.
static void RaiseWorkerPriority(const char* role) {
#ifdef _WIN32
    static const int level = [] {
        const char* text = std::getenv("APS5_WORKER_PRIORITY");
        return text != nullptr ? std::atoi(text) : 0;
    }();
    static std::once_flag processClass;
    std::call_once(processClass, [] {
        const char* text = std::getenv("APS5_PROCESS_PRIORITY");
        if (text == nullptr) return;
        const DWORD wanted = std::strcmp(text, "high") == 0 ? HIGH_PRIORITY_CLASS : ABOVE_NORMAL_PRIORITY_CLASS;
        const bool set = SetPriorityClass(GetCurrentProcess(), wanted) != 0;
        AgcDriver::ReportLine("[affinity] process priority class %s: %s\n", text, set ? "set" : "refused");
    });
    if (level == 0) return;
    const bool set = SetThreadPriority(GetCurrentThread(), level) != 0;
    AgcDriver::ReportLine("[affinity] %s priority %d: %s\n", role, level, set ? "set" : "refused");
#else
    static_cast<void>(role);
#endif
}

void PinWorkerThread(const char* role) {
    RaiseWorkerPriority(role);
    const auto mask = WorkerAffinityMask();
    if (mask == 0) return;
    static std::once_flag summary;
    std::call_once(summary, [mask] {
        const auto& layout = CpuTopology::Get();
        AgcDriver::ReportLine("[affinity] queue workers and presenter -> 0x%llx (hybrid=%d efficient=0x%llx performant=0x%llx process=0x%llx)\n", static_cast<unsigned long long>(mask), layout.hybrid ? 1 : 0, static_cast<unsigned long long>(layout.efficient), static_cast<unsigned long long>(layout.performant), static_cast<unsigned long long>(layout.process));
    });
    CpuTopology::PinTraced(role, nullptr, mask);
}

}
