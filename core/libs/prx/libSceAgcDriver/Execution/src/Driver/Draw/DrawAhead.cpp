#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawAhead.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/Submission.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include <utility>

namespace AgcDriver::DriverDetail {

DrawAhead::DrawAhead(Prepare prepare, ReadPairs readPairs, std::function<void()> started) : prepare(std::move(prepare)), readPairs(std::move(readPairs)), started(std::move(started)) {
    thread = std::thread([this] { run(); });
}

DrawAhead::~DrawAhead() {
    {
        std::lock_guard lock(mutex);
        shutdown = true;
        stop.store(true, std::memory_order_relaxed);
    }
    changed.notify_all();
    if (thread.joinable()) thread.join();
}

DrawAhead::Event& DrawAhead::at(std::deque<Event>& events, std::size_t index) {
    while (events.size() <= index) events.emplace_back();
    return events[index];
}

void DrawAhead::Begin(const Submission& job, const QueueState& state) {
    {
        std::lock_guard lock(mutex);
        submission = &job;
        start = state;
        pendingJob = true;
        poisoned = false;
        stop.store(false, std::memory_order_relaxed);
        draws.clear();
        loads.clear();
        workerDraw = 0;
        workerLoad = 0;
        ++counters.submissions;
    }
    changed.notify_all();
}

std::shared_ptr<PreparedDraw> DrawAhead::TakeDraw() {
    std::unique_lock lock(mutex);
    ++counters.draws;
    auto& event = at(draws, workerDraw++);
    if (event.state == State::Working) {
        ++counters.waited;
        changed.wait(lock, [&] { return event.state != State::Working; });
    }
    const bool ready = event.state == State::Ready;
    event.state = State::Taken;
    if (!ready) {
        ++counters.workerOwn;
        return nullptr;
    }
    auto prepared = std::move(event.prepared);
    if (prepared == nullptr) return nullptr;
    if (poisoned) {
        ++counters.poisonedDraws;
        return nullptr;
    }
    ++counters.handed;
    return prepared;
}

void DrawAhead::ConfirmLoad(std::span<const std::uint32_t> pairs) {
    {
        std::unique_lock lock(mutex);
        auto& event = at(loads, workerLoad++);
        if (event.state == State::Working) changed.wait(lock, [&] { return event.state != State::Working; });
        if (event.state == State::Pending) {
            event.pairs.assign(pairs.begin(), pairs.end());
            ++counters.loadsBehind;
        } else if (event.failed || !std::equal(event.pairs.begin(), event.pairs.end(), pairs.begin(), pairs.end())) {
            ++counters.loadMismatches;
            poisoned = true;
            stop.store(true, std::memory_order_relaxed);
        } else {
            ++counters.loadsAhead;
        }
        event.state = State::Taken;
    }
    changed.notify_all();
}

void DrawAhead::Poison() {
    std::lock_guard lock(mutex);
    poisoned = true;
    stop.store(true, std::memory_order_relaxed);
}

void DrawAhead::End() {
    std::unique_lock lock(mutex);
    stop.store(true, std::memory_order_relaxed);
    pendingJob = false;
    changed.notify_all();
    changed.wait(lock, [&] { return !active; });
    submission = nullptr;
    draws.clear();
    loads.clear();
}

DrawAhead::Counters DrawAhead::Totals() const {
    std::lock_guard lock(mutex);
    return counters;
}

void DrawAhead::run() {
    if (started) started();
    for (;;) {
        const Submission* job = nullptr;
        QueueState state;
        {
            std::unique_lock lock(mutex);
            changed.wait(lock, [&] { return shutdown || pendingJob; });
            if (shutdown) return;
            pendingJob = false;
            active = true;
            job = submission;
            state = std::move(start);
        }
        walk(*job, std::move(state));
        {
            std::lock_guard lock(mutex);
            active = false;
        }
        changed.notify_all();
    }
}

void DrawAhead::walk(const Submission& job, QueueState state) {
    const auto& commands = job.commands;
    std::size_t drawOrdinal = 0, loadOrdinal = 0;
    const auto halt = [&] {
        std::lock_guard lock(mutex);
        ++counters.stopped;
    };
    for (std::size_t cursor = 0; cursor < commands.size();) {
        if (stop.load(std::memory_order_relaxed)) return;
        const auto header = commands[cursor];
        if (Pm4::FillerPacket(header)) {
            ++cursor;
            continue;
        }
        const auto count = Pm4::PacketWords(header);
        if (count > commands.size() - cursor) return;
        const auto packet = std::span(commands).subspan(cursor, count);
        const auto opcode = (header >> 8u) & 0xffu;
        cursor += count;
        if (Pm4::DrawOpcode(opcode)) {
            Event* event = nullptr;
            {
                std::lock_guard lock(mutex);
                event = &at(draws, drawOrdinal++);
                if (event->state != State::Pending || stop.load(std::memory_order_relaxed)) continue;
                event->state = State::Working;
            }
            std::shared_ptr<PreparedDraw> prepared;
            try {
                prepared = prepare(state, packet, job);
            } catch (...) {
                prepared = nullptr;
            }
            {
                std::lock_guard lock(mutex);
                event->prepared = std::move(prepared);
                if (event->prepared != nullptr) ++counters.prepared;
                event->state = State::Ready;
            }
            changed.notify_all();
            continue;
        }
        const auto effect = Pm4::PacketStateEffect(header);
        if (effect == Pm4::StateEffect::None) continue;
        if (effect == Pm4::StateEffect::Unknown) {
            halt();
            return;
        }
        if (effect == Pm4::StateEffect::Registers) {
            try {
                Pm4::Execute(packet, state);
            } catch (...) {
                halt();
                return;
            }
            continue;
        }
        std::vector<std::uint32_t> pairs;
        bool own = false;
        Event* event = nullptr;
        {
            std::lock_guard lock(mutex);
            event = &at(loads, loadOrdinal++);
            if (event->state == State::Taken) {
                pairs = event->pairs;
            } else {
                event->state = State::Working;
                own = true;
            }
        }
        if (own) {
            bool failed = false;
            try {
                pairs = readPairs(packet);
            } catch (...) {
                failed = true;
            }
            {
                std::lock_guard lock(mutex);
                event->failed = failed;
                if (!failed) event->pairs = pairs;
                event->state = State::Ready;
            }
            changed.notify_all();
            if (failed) {
                halt();
                return;
            }
        }
        try {
            Pm4::ApplyRegisterPairs(packet, state, pairs);
        } catch (...) {
            halt();
            return;
        }
    }
}

}
