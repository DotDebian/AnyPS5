#include "prx/libSceAgcDriver/Execution/include/Driver/Draw/DrawAhead.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Queues/Submission.hpp"
#include "prx/libSceAgcDriver/Execution/include/Pm4.hpp"
#include <algorithm>
#include <chrono>
#include <utility>

namespace AgcDriver::DriverDetail {

DrawAhead::DrawAhead(Prepare prepare, ReadPairs readPairs, std::function<void()> started, std::size_t helperCount) : prepare(std::move(prepare)), readPairs(std::move(readPairs)), started(std::move(started)) {
    for (std::size_t index = 0; index < helperCount; ++index) helpers.push_back(std::make_unique<Helper>());
    for (auto& helper : helpers) helper->thread = std::thread([this, target = helper.get()] { help(*target); });
    thread = std::thread([this] { run(); });
}

DrawAhead::~DrawAhead() {
    {
        std::lock_guard lock(mutex);
        shutdown = true;
        stop.store(true, std::memory_order_relaxed);
    }
    changed.notify_all();
    work.notify_all();
    if (thread.joinable()) thread.join();
    for (auto& helper : helpers) {
        if (helper->thread.joinable()) helper->thread.join();
    }
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
        walkerDraw = 0;
        ++counters.submissions;
    }
    changed.notify_all();
}

std::shared_ptr<PreparedDraw> DrawAhead::TakeDraw() {
    std::unique_lock lock(mutex);
    ++counters.draws;
    if (walkerDraw > workerDraw) counters.depthSum += walkerDraw - workerDraw;
    auto& event = at(draws, workerDraw++);
    if (event.state == State::Working) {
        ++counters.waited;
        const auto waitStarted = std::chrono::steady_clock::now();
        changed.wait(lock, [&] { return event.state != State::Working; });
        counters.waitNanoseconds += static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - waitStarted).count());
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
            // The helpers still hold the submission and its events: End waits for them too.
            std::unique_lock lock(mutex);
            changed.wait(lock, [&] { return std::all_of(helpers.begin(), helpers.end(), [](const std::unique_ptr<Helper>& helper) { return helper->stage == Helper::Stage::Idle; }); });
            active = false;
        }
        changed.notify_all();
    }
}

void DrawAhead::help(Helper& helper) {
    if (started) started();
    for (;;) {
        {
            std::unique_lock lock(mutex);
            work.wait(lock, [&] { return shutdown || helper.stage == Helper::Stage::Loaded; });
            // A job loaded before the shutdown is still prepared: the walk waits for it.
            if (helper.stage != Helper::Stage::Loaded) return;
            helper.stage = Helper::Stage::Busy;
        }
        std::shared_ptr<PreparedDraw> prepared;
        try {
            prepared = prepare(helper.state, helper.packet, *helper.submission);
        } catch (...) {
            prepared = nullptr;
        }
        {
            std::lock_guard lock(mutex);
            helper.event->prepared = std::move(prepared);
            if (helper.event->prepared != nullptr) {
                ++counters.prepared;
                ++counters.helped;
            }
            helper.event->state = State::Ready;
            helper.event = nullptr;
            helper.stage = Helper::Stage::Idle;
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
            Helper* helper = nullptr;
            {
                std::lock_guard lock(mutex);
                event = &at(draws, drawOrdinal++);
                walkerDraw = drawOrdinal;
                if (event->state != State::Pending || stop.load(std::memory_order_relaxed)) continue;
                event->state = State::Working;
                for (auto& candidate : helpers) {
                    if (candidate->stage != Helper::Stage::Idle) continue;
                    helper = candidate.get();
                    helper->stage = Helper::Stage::Reserved;
                    break;
                }
            }
            if (helper != nullptr) {
                // The state as it stands at this draw: the walk goes on changing its own.
                bool loaded = true;
                try {
                    helper->state = state;
                } catch (...) {
                    loaded = false;
                }
                {
                    std::lock_guard lock(mutex);
                    if (loaded) {
                        helper->packet = packet;
                        helper->submission = &job;
                        helper->event = event;
                    }
                    helper->stage = loaded ? Helper::Stage::Loaded : Helper::Stage::Idle;
                }
                if (loaded) {
                    work.notify_all();
                    continue;
                }
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
