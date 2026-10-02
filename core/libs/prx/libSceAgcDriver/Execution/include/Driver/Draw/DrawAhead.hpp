#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWAHEAD_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_DRIVER_DRAW_DRAWAHEAD_HPP

#include "prx/libSceAgcDriver/Execution/include/QueueState.hpp"
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace AgcDriver::DriverDetail {

struct Submission;
struct PreparedDraw;

class DrawAhead {
public:
    using Prepare = std::function<std::shared_ptr<PreparedDraw>(const QueueState& state, std::span<const std::uint32_t> packet, const Submission& submission)>;
    using ReadPairs = std::function<std::vector<std::uint32_t>(std::span<const std::uint32_t> packet)>;
    struct Counters {
        std::uint64_t submissions = 0;
        std::uint64_t draws = 0;
        std::uint64_t prepared = 0;
        std::uint64_t handed = 0;
        std::uint64_t waited = 0;
        std::uint64_t workerOwn = 0;
        std::uint64_t poisonedDraws = 0;
        std::uint64_t loadsAhead = 0;
        std::uint64_t loadsBehind = 0;
        std::uint64_t loadMismatches = 0;
        std::uint64_t stopped = 0;
    };

    DrawAhead(Prepare prepare, ReadPairs readPairs, std::function<void()> started = {});
    ~DrawAhead();
    DrawAhead(const DrawAhead&) = delete;
    DrawAhead& operator=(const DrawAhead&) = delete;

    void Begin(const Submission& submission, const QueueState& state);
    std::shared_ptr<PreparedDraw> TakeDraw();
    void ConfirmLoad(std::span<const std::uint32_t> pairs);
    void Poison();
    void End();
    Counters Totals() const;

private:
    enum class State : std::uint8_t { Pending, Working, Ready, Taken };
    struct Event {
        State state = State::Pending;
        bool failed = false;
        std::shared_ptr<PreparedDraw> prepared;
        std::vector<std::uint32_t> pairs;
    };
    void run();
    void walk(const Submission& submission, QueueState state);
    static Event& at(std::deque<Event>& events, std::size_t index);

    Prepare prepare;
    ReadPairs readPairs;
    std::function<void()> started;
    mutable std::mutex mutex;
    std::condition_variable changed;
    const Submission* submission = nullptr;
    QueueState start;
    bool pendingJob = false;
    bool active = false;
    bool poisoned = false;
    bool shutdown = false;
    std::atomic<bool> stop{false};
    std::deque<Event> draws;
    std::deque<Event> loads;
    std::size_t workerDraw = 0;
    std::size_t workerLoad = 0;
    Counters counters;
    std::thread thread;
};

}

#endif
