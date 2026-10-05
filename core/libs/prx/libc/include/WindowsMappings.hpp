#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSMAPPINGS_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_WINDOWSMAPPINGS_HPP

#ifdef _WIN32
#include <windows.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <map>
#include <memory>
#include <vector>
#include <stdexcept>
#include <string>
#include <system_error>

namespace GuestArena {

class WindowsMappings {
public:
    static WindowsMappings& Get() {
        static WindowsMappings mappings;
        return mappings;
    }

    void* Reserve(void* address, std::size_t bytes) {
        return allocate(GetCurrentProcess(), address, bytes, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
    }

    void Commit(void* address, std::size_t bytes, DWORD protection, std::size_t granule, bool watched) {
        std::lock_guard lock(mutex);
        const auto end = reinterpret_cast<std::uintptr_t>(address) + bytes;
        for (auto cursor = reinterpret_cast<std::uintptr_t>(address); cursor < end;) {
            const auto memory = query(cursor);
            const auto stop = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
            if (memory.State == MEM_RESERVE) {
                const auto limit = std::min(end, cursor + granule);
                auto placeholderEnd = stop;
                while (placeholderEnd < limit) {
                    const auto next = query(placeholderEnd);
                    if (next.State != MEM_RESERVE) break;
                    placeholderEnd = reinterpret_cast<std::uintptr_t>(next.BaseAddress) + next.RegionSize;
                }
                const auto size = std::min(limit, placeholderEnd) - cursor;
                reset(cursor, size);
                const DWORD flags = MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | (watched ? MEM_WRITE_WATCH : 0);
                if (!allocate(GetCurrentProcess(), reinterpret_cast<void*>(cursor), size, flags, protection, nullptr, 0)) fail("replace guest placeholder with private memory");
                cursor += size;
            } else {
                if (memory.State != MEM_COMMIT) throw std::runtime_error("guest memory is not committed");
                const auto mapped = views.find(cursor & ~(pageBytes - 1));
                if (mapped != views.end()) {
                    mapped->second.protection = protection;
                    mapped->second.armed = false;
                    invalidate(*mapped->second.page);
                }
                DWORD previous;
                if (!VirtualProtect(reinterpret_cast<void*>(cursor), stop - cursor, protection, &previous)) fail("protect guest memory");
                cursor = stop;
            }
        }
    }

    void Reset(void* address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        reset(reinterpret_cast<std::uintptr_t>(address), bytes);
    }

    // A section page mapped at one guest address only is resident: its bytes live in private memory
    // at that address, watched with MEM_WRITE_WATCH like flexible memory, so tracking its writes
    // raises no fault on the guest thread (a resumed fault overwrites the System V red zone). A page
    // mapped at a second address moves to the section and both addresses become views of it, tracked
    // by write protection. APS5_SHARED_DIRECT_MEMORY=1 maps every page as a view as before.
    void Map(void* address, std::size_t bytes, HANDLE section, std::uint64_t offset, DWORD protection, bool watched) {
        std::lock_guard lock(mutex);
        auto cursor = reinterpret_cast<std::uintptr_t>(address);
        reset(cursor, bytes);
        watchResident = watched;
        HANDLE duplicate = nullptr;
        if (!DuplicateHandle(GetCurrentProcess(), section, GetCurrentProcess(), &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS)) fail("keep shared guest section");
        const auto owned = std::make_shared<Section>(duplicate);
        const auto slotOf = [&](std::size_t done) -> Physical& { return physical[std::make_pair(reinterpret_cast<std::uintptr_t>(section), offset + done)]; };
        const auto pageOf = [](Physical& slot) {
            auto shared = slot.page.lock();
            if (!shared) {
                shared = std::make_shared<SharedPage>();
                slot.page = shared;
            }
            return shared;
        };
        for (std::size_t done = 0; done < bytes;) {
            const auto base = cursor + done;
            if (residentDirect && pageOf(slotOf(done))->aliases.empty()) {
                auto runBytes = pageBytes;
                while (done + runBytes < bytes && (base + runBytes) % chunkBytes != 0) {
                    const auto next = slotOf(done + runBytes).page.lock();
                    if (next && !next->aliases.empty()) break;
                    runBytes += pageBytes;
                }
                split(base, runBytes);
                commitResident(base, runBytes);
                residentRuns.emplace(base, makeRun(base, base + runBytes));
                Window window;
                for (std::size_t at = 0; at < runBytes; at += pageBytes) {
                    auto& slot = slotOf(done + at);
                    const auto shared = pageOf(slot);
                    if (slot.stored) {
                        if (window.view == nullptr) window = open(section, offset + done, runBytes);
                        commitSection(window.bytes + at);
                        std::memcpy(reinterpret_cast<void*>(base + at), window.bytes + at, pageBytes);
                    }
                    shared->aliases.push_back(base + at);
                    views.emplace(base + at, View{shared, protection, 0, false, owned, offset + done + at, 0, true, &slot});
                    invalidate(*shared);
                }
                close(window);
                DWORD previous;
                if (protection != PAGE_READWRITE && !VirtualProtect(reinterpret_cast<void*>(base), runBytes, protection, &previous)) fail("protect resident guest pages");
                done += runBytes;
                continue;
            }
            auto& slot = slotOf(done);
            const auto shared = pageOf(slot);
            const auto aliases = shared->aliases;
            for (const auto alias : aliases) {
                if (views.at(alias).resident) share(alias);
            }
            split(base, pageBytes);
            mapShared(base, section, offset + done, protection);
            slot.stored = true;
            shared->aliases.push_back(base);
            views.emplace(base, View{shared, protection, 0, false, owned, offset + done, 0, false, &slot});
            invalidate(*shared);
            done += pageBytes;
        }
    }

    void SetProtection(std::uintptr_t address, std::size_t bytes, DWORD protection) {
        std::lock_guard lock(mutex);
        for (auto it = views.lower_bound(address); it != views.end() && it->first < address + bytes; ++it) {
            it->second.protection = protection;
            it->second.armed = false;
            invalidate(*it->second.page);
        }
    }

    void Pin(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            auto& page = *it->second.page;
            ++page.pins;
            for (const auto alias : page.aliases) {
                auto& view = views.at(alias);
                if (!view.armed) continue;
                DWORD previous;
                if (!VirtualProtect(reinterpret_cast<void*>(alias), pageBytes, view.protection, &previous)) fail("pin shared guest page writable");
                view.armed = false;
            }
            invalidate(page);
        }
    }

    void Unpin(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < address + bytes; ++it) {
            auto& page = *it->second.page;
            if (page.pins != 0) --page.pins;
            invalidate(page);
        }
    }

    bool HandleWrite(std::uintptr_t address) {
        std::lock_guard lock(mutex);
        const auto base = address & ~(pageBytes - 1);
        const auto found = views.find(base);
        if (found == views.end() || !writable(found->second.protection)) return false;
        auto& view = found->second;
        if (view.resident) {
            // Resident pages are never write-protected for tracking: the write raced a page that
            // was being moved, and is retried only now that the page accepts it.
            const auto memory = query(address);
            return memory.State == MEM_COMMIT && writable(memory.Protect & 0xffu);
        }
        invalidate(*view.page);
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(base), pageBytes, view.protection, &previous)) fail("resume shared memory write");
        view.armed = false;
        return true;
    }

    bool BeginHostWrite(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto first = views.lower_bound(address & ~(pageBytes - 1));
        const auto end = address + bytes;
        for (auto it = first; it != views.end() && it->first < end; ++it) {
            if (!writable(it->second.protection)) return false;
        }
        for (auto it = first; it != views.end() && it->first < end; ++it) {
            auto& view = it->second;
            ++view.hostWrites;
            invalidate(*view.page);
            if (!view.armed) continue;
            DWORD previous;
            if (!VirtualProtect(reinterpret_cast<void*>(it->first), pageBytes, view.protection, &previous)) fail("open shared memory to a host write");
            view.armed = false;
        }
        return true;
    }

    void EndHostWrite(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto end = address + bytes;
        for (auto it = views.lower_bound(address & ~(pageBytes - 1)); it != views.end() && it->first < end; ++it) {
            --it->second.hostWrites;
            invalidate(*it->second.page);
        }
    }

    void* MapAlias(std::uintptr_t address, std::size_t bytes) {
        std::lock_guard lock(mutex);
        const auto refuse = [&](const char* reason) {
            char text[192];
            std::snprintf(text, sizeof(text), "read-write alias of shared guest memory 0x%llx+0x%llx: %s", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes), reason);
            return std::runtime_error(text);
        };
        if (address % pageBytes != 0 || bytes % pageBytes != 0 || bytes == 0) throw refuse("the range is not made of whole shared pages");
        for (std::size_t done = 0; done < bytes; done += pageBytes) {
            const auto found = views.find(address + done);
            if (found != views.end() && found->second.resident) share(address + done);
        }
        auto view = views.find(address);
        if (view == views.end()) throw refuse("the range does not start at a shared view");
        const auto section = view->second.section;
        const auto offset = view->second.offset;
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        for (std::size_t done = 0; done < bytes; done += pageBytes, ++view) {
            if (view == views.end() || view->first != address + done || view->second.offset != offset + done) throw refuse("the range is not one contiguous run of views of a section");
            if (view->second.section != section && !sameSection(view->second.section->handle, section->handle)) throw refuse("the range spans several sections");
        }
        const auto lead = offset % system.dwAllocationGranularity;
        void* alias = map(section->handle, GetCurrentProcess(), nullptr, offset - lead, lead + bytes, 0, PAGE_READWRITE, nullptr, 0);
        if (alias == nullptr) {
            char text[160];
            std::snprintf(text, sizeof(text), "MapViewOfFile3 of a read-write alias of shared guest memory 0x%llx+0x%llx", static_cast<unsigned long long>(address), static_cast<unsigned long long>(bytes));
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), text);
        }
        return static_cast<char*>(alias) + lead;
    }

    void UnmapAlias(void* alias) {
        if (alias == nullptr) return;
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        const auto base = reinterpret_cast<std::uintptr_t>(alias) & ~(static_cast<std::uintptr_t>(system.dwAllocationGranularity) - 1);
        if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(base), 0)) fail("unmap shared guest alias");
    }

    bool Protection(std::uintptr_t address, std::uint32_t* protection) {
        std::lock_guard lock(mutex);
        const auto found = views.find(address & ~(pageBytes - 1));
        if (found == views.end()) return false;
        *protection = found->second.protection;
        return true;
    }

    struct CollectStats {
        std::uint64_t calls = 0, bytes = 0, residentCalls = 0, residentBytes = 0, privateCalls = 0, privateBytes = 0, viewPages = 0, dirty = 0;
        std::uint64_t ticks = 0, watchTicks = 0, lockTicks = 0, last = 0;
        std::uint64_t sizeCalls[6] = {}, sizeBytes[6] = {}, sizeTicks[6] = {};
        std::uint64_t tookCalls[5] = {}, tookTicks[5] = {}, tookBytes[5] = {}, repeatTicks = 0, firstTicks = 0, repeats = 0;
    };
    CollectStats stats;
    static std::uint64_t now() {
        LARGE_INTEGER value;
        QueryPerformanceCounter(&value);
        return static_cast<std::uint64_t>(value.QuadPart);
    }

    bool Collect(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear) {
        static const bool trace = std::getenv("APS5_TRACE_COLLECT") != nullptr;
        if (!trace) return collect(address, bytes, pages, count, clear);
        const auto before = now();
        std::unique_lock lock(mutex);
        const auto locked = now();
        lock.unlock();
        const bool result = collect(address, bytes, pages, count, clear);
        const auto after = now();
        lock.lock();
        ++stats.calls;
        stats.bytes += bytes;
        stats.dirty += *count;
        stats.ticks += after - before;
        stats.lockTicks += locked - before;
        const int bucket = bytes <= 0x10000 ? 0 : bytes <= 0x100000 ? 1 : bytes <= 0x1000000 ? 2 : bytes <= 0x4000000 ? 3 : bytes <= 0x10000000 ? 4 : 5;
        ++stats.sizeCalls[bucket];
        stats.sizeBytes[bucket] += bytes;
        stats.sizeTicks[bucket] += after - locked;
        if (after - stats.last > 100000000ull) {
            stats.last = after;
            std::fprintf(stderr, "[collect] %llu calls %llu MiB in %.0f ms (lock wait %.0f ms, GetWriteWatch %.0f ms); resident %llu calls %llu MiB, private %llu calls %llu MiB, view pages %llu, dirty pages %llu, runs %zu, views %zu\n", static_cast<unsigned long long>(stats.calls), static_cast<unsigned long long>(stats.bytes >> 20), stats.ticks / 10000.0, stats.lockTicks / 10000.0, stats.watchTicks / 10000.0, static_cast<unsigned long long>(stats.residentCalls), static_cast<unsigned long long>(stats.residentBytes >> 20), static_cast<unsigned long long>(stats.privateCalls), static_cast<unsigned long long>(stats.privateBytes >> 20), static_cast<unsigned long long>(stats.viewPages), static_cast<unsigned long long>(stats.dirty), residentRuns.size(), views.size());
            {
                static char* const probe = [] {
                    char* memory = static_cast<char*>(VirtualAlloc(nullptr, 0x10000, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE));
                    std::memset(memory, 1, 0x10000);
                    ResetWriteWatch(memory, 0x10000);
                    return memory;
                }();
                void* found[16];
                const auto time = [&](DWORD flags) {
                    const auto start = now();
                    for (int i = 0; i < 200; ++i) {
                        ULONG_PTR available = 16;
                        DWORD granule = 0;
                        GetWriteWatch(flags, probe, 0x10000, found, &available, &granule);
                    }
                    return (now() - start) / 2000.0;
                };
                const double reset = time(WRITE_WATCH_FLAG_RESET);
                const double plain = time(0);
                const auto start = now();
                MEMORY_BASIC_INFORMATION information{};
                for (int i = 0; i < 200; ++i) VirtualQuery(probe, &information, sizeof(information));
                const double queried = (now() - start) / 2000.0;
                const auto run = residentRuns.begin();
                const auto startRun = now();
                for (int i = 0; i < 200 && run != residentRuns.end(); ++i) {
                    ULONG_PTR available = 16;
                    DWORD granule = 0;
                    GetWriteWatch(0, reinterpret_cast<void*>(run->first), 0x10000, found, &available, &granule);
                }
                std::fprintf(stderr, "[collect]   probe: GetWriteWatch on 64 KiB of a separate allocation %.2f us with reset, %.2f us without; VirtualQuery %.2f us; GetWriteWatch without reset on 64 KiB of a resident run %.2f us\n", reset, plain, queried, (now() - startRun) / 2000.0);
            }
            static const char* const spans[5] = {"under 1 us", "1 to 4 us", "4 to 16 us", "16 to 64 us", "over 64 us"};
            for (int i = 0; i < 5; ++i) std::fprintf(stderr, "[collect]   GetWriteWatch %s: %llu calls, %llu MiB, %.0f ms\n", spans[i], static_cast<unsigned long long>(stats.tookCalls[i]), static_cast<unsigned long long>(stats.tookBytes[i] >> 20), stats.tookTicks[i] / 10000.0);
            std::fprintf(stderr, "[collect]   repeat: %llu sampled calls took %.2f us each, the same range again without reset %.2f us\n", static_cast<unsigned long long>(stats.repeats), stats.repeats ? stats.firstTicks / 10.0 / stats.repeats : 0.0, stats.repeats ? stats.repeatTicks / 10.0 / stats.repeats : 0.0);            static const char* const names[6] = {"64K", "1M", "16M", "64M", "256M", "more"};
            for (int i = 0; i < 6; ++i) std::fprintf(stderr, "[collect]   up to %s: %llu calls, %llu MiB, %.0f ms\n", names[i], static_cast<unsigned long long>(stats.sizeCalls[i]), static_cast<unsigned long long>(stats.sizeBytes[i] >> 20), stats.sizeTicks[i] / 10000.0);
            const auto last = stats.last;
            stats = {};
            stats.last = last;
        }
        return result;
    }

    bool collect(std::uintptr_t address, std::size_t bytes, void** pages, std::size_t* count, bool clear) {
        std::lock_guard lock(mutex);
        const auto capacity = *count;
        *count = 0;
        const auto end = address + bytes;
        for (auto cursor = address; cursor < end;) {
            const auto nextClean = cleanRanges.upper_bound(cursor);
            if (nextClean != cleanRanges.begin()) {
                const auto clean = std::prev(nextClean);
                if (cursor < clean->second) {
                    cursor = std::min(end, clean->second);
                    continue;
                }
            }
            const auto base = cursor & ~(pageBytes - 1);
            const auto found = views.find(base);
            if (found != views.end() && found->second.resident) {
                const auto run = std::prev(residentRuns.upper_bound(cursor));
                auto& state = run->second;
                const auto stop = std::min(end, state.end);
                const auto firstBit = (cursor - run->first) >> 12;
                const auto lastBit = (stop - 1 - run->first) >> 12;
                bool fresh = false;
                for (auto bit = firstBit; state.fresh != 0 && bit <= lastBit && !fresh; ++bit) fresh = (state.collected[bit >> 6] >> (bit & 63) & 1) == 0;
                if (fresh) {
                    // Pages never collected since they were mapped report as written, as a view does.
                    if (lastBit - firstBit + 1 > capacity - *count) {
                        for (auto at = cursor; *count < capacity; at += 4096) pages[(*count)++] = reinterpret_cast<void*>(at);
                        return true;
                    }
                    for (auto at = cursor; at < stop; at += 4096) pages[(*count)++] = reinterpret_cast<void*>(at);
                    if (clear) {
                        if (watchResident && ResetWriteWatch(reinterpret_cast<void*>(cursor), stop - cursor) != 0) fail("reset resident guest writes");
                        for (auto bit = firstBit; bit <= lastBit; ++bit) {
                            auto& word = state.collected[bit >> 6];
                            if ((word >> (bit & 63) & 1) != 0) continue;
                            word |= 1ull << (bit & 63);
                            --state.fresh;
                        }
                    }
                } else {
                    ULONG_PTR available = capacity - *count;
                    if (available == 0) return true;
                    DWORD granularity = 0;
                    const auto watchStart = now();
                    if (GetWriteWatch(clear ? WRITE_WATCH_FLAG_RESET : 0, reinterpret_cast<void*>(cursor), stop - cursor, pages + *count, &available, &granularity) != 0) fail("collect resident guest writes");
                    stats.watchTicks += now() - watchStart; ++stats.residentCalls; stats.residentBytes += stop - cursor;
                    {
                        const auto took = now() - watchStart;
                        const int slot = took < 10 ? 0 : took < 40 ? 1 : took < 160 ? 2 : took < 640 ? 3 : 4;
                        ++stats.tookCalls[slot];
                        stats.tookTicks[slot] += took;
                        stats.tookBytes[slot] += stop - cursor;
                        if ((stats.residentCalls & 63) == 0) {
                            void* again[64];
                            ULONG_PTR some = 64;
                            DWORD granule = 0;
                            const auto repeatStart = now();
                            GetWriteWatch(0, reinterpret_cast<void*>(cursor), stop - cursor, again, &some, &granule);
                            stats.repeatTicks += now() - repeatStart;
                            stats.firstTicks += took;
                            ++stats.repeats;
                        }
                    }
                    *count += available;
                    if (*count == capacity) return true;
                }
                cursor = stop;
            } else if (found != views.end()) {
                auto& view = found->second;
                const auto stop = std::min(end, base + pageBytes);
                if (view.protection == PAGE_NOACCESS) return false;
                const bool pinned = view.page->pins != 0;
                if (pinned || view.seen != view.page->generation) {
                    const auto needed = (stop - cursor + 4095) / 4096;
                    if (needed > capacity - *count) {
                        for (auto at = cursor; *count < capacity; at += 4096) pages[(*count)++] = reinterpret_cast<void*>(at);
                        return true;
                    }
                    for (auto at = cursor; at < stop; at += 4096) pages[(*count)++] = reinterpret_cast<void*>(at);
                }
                if (clear && !pinned) {
                    for (const auto alias : view.page->aliases) {
                        auto& other = views.at(alias);
                        if (!writable(other.protection) || other.armed || other.hostWrites != 0) continue;
                        DWORD previous;
                        const DWORD protection = other.protection == PAGE_EXECUTE_READWRITE ? PAGE_EXECUTE_READ : PAGE_READONLY;
                        if (!VirtualProtect(reinterpret_cast<void*>(alias), pageBytes, protection, &previous)) fail("arm shared memory write tracking");
                        other.armed = true;
                    }
                    view.seen = view.page->generation;
                    rememberClean(base, base + pageBytes);
                }
                cursor = stop;
            } else {
                const auto memory = query(cursor);
                if (memory.State != MEM_COMMIT || memory.Type != MEM_PRIVATE) return false;
                const auto stop = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
                ULONG_PTR available = capacity - *count;
                if (available == 0) return true;
                DWORD granularity = 0;
                const auto watchStart = now();
                if (GetWriteWatch(clear ? WRITE_WATCH_FLAG_RESET : 0, reinterpret_cast<void*>(cursor), stop - cursor, pages + *count, &available, &granularity) != 0) fail("collect private guest writes");
                stats.watchTicks += now() - watchStart; ++stats.privateCalls; stats.privateBytes += stop - cursor;
                *count += available;
                if (*count == capacity) return true;
                cursor = stop;
            }
        }
        return true;
    }

private:
    static constexpr std::size_t pageBytes = 0x4000;
    static constexpr std::size_t chunkBytes = 0x4000000;
    struct SharedPage {
        std::uint64_t generation = 1;
        std::vector<std::uintptr_t> aliases;
        std::uint32_t pins = 0;
    };
    struct Section {
        HANDLE handle;
        explicit Section(HANDLE handle) : handle(handle) {}
        Section(const Section&) = delete;
        Section& operator=(const Section&) = delete;
        ~Section() { CloseHandle(handle); }
    };
    // stored: the section holds bytes of this page that a new mapping has to read back.
    struct Physical {
        std::weak_ptr<SharedPage> page;
        bool stored = false;
    };
    struct View {
        std::shared_ptr<SharedPage> page;
        DWORD protection;
        std::uint64_t seen;
        bool armed;
        std::shared_ptr<Section> section;
        std::uint64_t offset;
        std::uint32_t hostWrites;
        bool resident;
        Physical* slot;
    };
    // One private allocation of resident pages, at most chunkBytes long; collected has a bit per
    // 4 KiB page that a clearing collect has reported, fresh counts the pages still without it.
    struct Run {
        std::uintptr_t end;
        std::vector<std::uint64_t> collected;
        std::uint32_t fresh;
    };
    static Run makeRun(std::uintptr_t start, std::uintptr_t end) {
        const auto pages = (end - start) >> 12;
        return Run{end, std::vector<std::uint64_t>((pages + 63) / 64), static_cast<std::uint32_t>(pages)};
    }
    struct Window {
        void* view = nullptr;
        unsigned char* bytes = nullptr;
    };

    Window open(HANDLE section, std::uint64_t offset, std::size_t bytes) {
        const auto lead = offset % granularity;
        Window window;
        window.view = map(section, GetCurrentProcess(), nullptr, offset - lead, lead + bytes, 0, PAGE_READWRITE, nullptr, 0);
        if (window.view == nullptr) fail("open a window on direct memory");
        window.bytes = static_cast<unsigned char*>(window.view) + lead;
        return window;
    }

    void close(Window& window) {
        if (window.view != nullptr && !unmap(GetCurrentProcess(), window.view, 0)) fail("close a window on direct memory");
        window = {};
    }

    static void commitSection(void* page) {
        if (!VirtualAlloc(page, pageBytes, MEM_COMMIT, PAGE_READWRITE)) fail("commit a direct memory page");
    }

    void commitResident(std::uintptr_t base, std::size_t bytes) {
        const DWORD flags = MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER | (watchResident ? MEM_WRITE_WATCH : 0);
        if (!allocate(GetCurrentProcess(), reinterpret_cast<void*>(base), bytes, flags, PAGE_READWRITE, nullptr, 0)) fail("replace guest placeholder with resident direct memory");
    }

    void mapShared(std::uintptr_t base, HANDLE section, std::uint64_t offset, DWORD protection) {
        void* page = reinterpret_cast<void*>(base);
        if (!map(section, GetCurrentProcess(), page, offset, pageBytes, MEM_REPLACE_PLACEHOLDER, PAGE_EXECUTE_READWRITE, nullptr, 0)) fail("map shared guest page");
        if (!VirtualAlloc(page, pageBytes, MEM_COMMIT, PAGE_EXECUTE_READWRITE)) fail("commit shared guest page");
        DWORD previous;
        if (!VirtualProtect(page, pageBytes, protection, &previous)) fail("protect shared guest page");
    }

    static bool blank(std::uintptr_t page) {
        const auto* words = reinterpret_cast<const std::uint64_t*>(page);
        for (std::size_t i = 0; i < pageBytes / sizeof(std::uint64_t); ++i) {
            if (words[i] != 0) return false;
        }
        return true;
    }

    // Makes `at` an allocation boundary of the resident run that spans it. A private allocation
    // cannot be split in place, so the run is copied out, released and committed again as two.
    void cut(std::uintptr_t at) {
        auto it = residentRuns.upper_bound(at);
        if (it == residentRuns.begin()) return;
        --it;
        const auto start = it->first;
        const auto stop = it->second.end;
        if (at <= start || at >= stop) return;
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(start), stop - start, PAGE_READWRITE, &previous)) fail("open resident guest pages");
        const std::vector<unsigned char> bytes(reinterpret_cast<const unsigned char*>(start), reinterpret_cast<const unsigned char*>(stop));
        if (!VirtualFree(reinterpret_cast<void*>(start), stop - start, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release resident guest pages");
        if (!VirtualFree(reinterpret_cast<void*>(start), at - start, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split resident guest placeholder");
        commitResident(start, at - start);
        commitResident(at, stop - at);
        std::memcpy(reinterpret_cast<void*>(start), bytes.data(), bytes.size());
        for (auto page = start; page < stop; page += pageBytes) {
            const auto protection = views.at(page).protection;
            if (protection != PAGE_READWRITE && !VirtualProtect(reinterpret_cast<void*>(page), pageBytes, protection, &previous)) fail("protect resident guest pages");
        }
        residentRuns.erase(it);
        residentRuns.emplace(start, makeRun(start, at));
        residentRuns.emplace(at, makeRun(at, stop));
    }

    // Moves a resident page into the section and maps the section in its place.
    void share(std::uintptr_t base) {
        cut(base);
        cut(base + pageBytes);
        auto& view = views.at(base);
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(base), pageBytes, PAGE_READWRITE, &previous)) fail("open a resident guest page");
        auto window = open(view.section->handle, view.offset, pageBytes);
        commitSection(window.bytes);
        std::memcpy(window.bytes, reinterpret_cast<const void*>(base), pageBytes);
        close(window);
        view.slot->stored = true;
        if (!VirtualFree(reinterpret_cast<void*>(base), pageBytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release a resident guest page");
        residentRuns.erase(base);
        mapShared(base, view.section->handle, view.offset, view.protection);
        view.resident = false;
        view.seen = 0;
        view.armed = false;
        invalidate(*view.page);
        if ((++sharedPages & (sharedPages - 1)) == 0) std::fprintf(stderr, "[memory] %llu direct memory pages mapped at a second address so far (last 0x%llx); their writes are tracked by write protection\n", static_cast<unsigned long long>(sharedPages), static_cast<unsigned long long>(base));
    }

    // Stores the pages of a resident run that is about to be released back into the section.
    void evict(std::map<std::uintptr_t, Run>::iterator run) {
        const auto start = run->first;
        const auto stop = run->second.end;
        DWORD previous;
        if (!VirtualProtect(reinterpret_cast<void*>(start), stop - start, PAGE_READWRITE, &previous)) fail("open resident guest pages");
        const auto section = views.at(start).section;
        auto window = open(section->handle, views.at(start).offset, stop - start);
        for (auto page = start; page < stop; page += pageBytes) {
            const auto found = views.find(page);
            auto& view = found->second;
            if (view.slot->stored || !blank(page)) {
                commitSection(window.bytes + (page - start));
                std::memcpy(window.bytes + (page - start), reinterpret_cast<const void*>(page), pageBytes);
                view.slot->stored = true;
            }
            std::erase(view.page->aliases, page);
            views.erase(found);
        }
        close(window);
        residentRuns.erase(run);
    }
    void forgetClean(std::uintptr_t start, std::uintptr_t end) {
        auto it = cleanRanges.lower_bound(start);
        if (it != cleanRanges.begin() && std::prev(it)->second > start) --it;
        while (it != cleanRanges.end() && it->first < end) {
            const auto first = it->first;
            const auto last = it->second;
            it = cleanRanges.erase(it);
            if (first < start) cleanRanges.emplace(first, start);
            if (last > end) it = cleanRanges.emplace(end, last).first;
        }
    }

    void rememberClean(std::uintptr_t start, std::uintptr_t end) {
        auto it = cleanRanges.lower_bound(start);
        if (it != cleanRanges.begin() && std::prev(it)->second >= start) --it;
        while (it != cleanRanges.end() && it->first <= end) {
            start = std::min(start, it->first);
            end = std::max(end, it->second);
            it = cleanRanges.erase(it);
        }
        cleanRanges.emplace(start, end);
    }

    void invalidate(SharedPage& page) {
        ++page.generation;
        for (const auto alias : page.aliases) forgetClean(alias, alias + pageBytes);
    }

    bool sameSection(HANDLE first, HANDLE second) const {
        return compare != nullptr && compare(first, second);
    }

    static bool writable(DWORD protection) {
        return protection == PAGE_READWRITE || protection == PAGE_EXECUTE_READWRITE;
    }
    using AllocateFunction = PVOID (WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using MapFunction = PVOID (WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
    using UnmapFunction = BOOL (WINAPI*)(HANDLE, PVOID, ULONG);
    using CompareFunction = BOOL (WINAPI*)(HANDLE, HANDLE);

    WindowsMappings() {
        const auto module = GetModuleHandleW(L"KernelBase.dll");
        if (!module) fail("load Windows memory API");
        allocate = reinterpret_cast<AllocateFunction>(GetProcAddress(module, "VirtualAlloc2"));
        map = reinterpret_cast<MapFunction>(GetProcAddress(module, "MapViewOfFile3"));
        unmap = reinterpret_cast<UnmapFunction>(GetProcAddress(module, "UnmapViewOfFile2"));
        if (!allocate || !map || !unmap) throw std::runtime_error("Windows placeholder memory APIs are required");
        compare = reinterpret_cast<CompareFunction>(GetProcAddress(module, "CompareObjectHandles"));
        SYSTEM_INFO system{};
        GetSystemInfo(&system);
        granularity = system.dwAllocationGranularity;
        residentDirect = std::getenv("APS5_SHARED_DIRECT_MEMORY") == nullptr;
    }

    [[noreturn]] static void fail(const char* operation) {
        throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation);
    }

    static MEMORY_BASIC_INFORMATION query(std::uintptr_t address) {
        MEMORY_BASIC_INFORMATION memory{};
        if (VirtualQuery(reinterpret_cast<void*>(address), &memory, sizeof(memory)) != sizeof(memory)) fail("query guest memory");
        return memory;
    }

    static void split(std::uintptr_t address, std::size_t bytes) {
        auto memory = query(address);
        memory = query(reinterpret_cast<std::uintptr_t>(memory.AllocationBase));
        if (memory.State != MEM_RESERVE) throw std::runtime_error("guest mapping requires a placeholder");
        const auto base = reinterpret_cast<std::uintptr_t>(memory.BaseAddress);
        if (address != base) {
            if (!VirtualFree(reinterpret_cast<void*>(base), address - base, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split guest placeholder prefix");
            memory = query(address);
        }
        if (memory.RegionSize < bytes) throw std::runtime_error("guest placeholder is too small");
        if (memory.RegionSize != bytes && !VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("split guest placeholder suffix");
    }

    void reset(std::uintptr_t address, std::size_t bytes) {
        const auto end = address + bytes;
        forgetClean(address, end);
        cut(address);
        cut(end);
        for (auto cursor = address; cursor < end;) {
            const auto memory = query(cursor);
            if (memory.State == MEM_RESERVE) {
                cursor = std::min(end, reinterpret_cast<std::uintptr_t>(memory.BaseAddress) + memory.RegionSize);
                continue;
            }
            if (reinterpret_cast<std::uintptr_t>(memory.AllocationBase) != cursor) throw std::runtime_error("cannot release part of a host allocation");
            auto allocationEnd = cursor;
            do {
                const auto part = query(allocationEnd);
                if (part.AllocationBase != memory.AllocationBase) break;
                allocationEnd = reinterpret_cast<std::uintptr_t>(part.BaseAddress) + part.RegionSize;
            } while (allocationEnd < end);
            if (allocationEnd > end || query(allocationEnd).AllocationBase == memory.AllocationBase) throw std::runtime_error("guest release truncates a host allocation");
            if (memory.Type == MEM_MAPPED) {
                if (!unmap(GetCurrentProcess(), reinterpret_cast<void*>(cursor), MEM_PRESERVE_PLACEHOLDER)) fail("unmap shared guest page");
                const auto found = views.find(cursor);
                if (found != views.end()) {
                    std::erase(found->second.page->aliases, cursor);
                    views.erase(found);
                }
            } else if (memory.Type == MEM_PRIVATE) {
                if (const auto run = residentRuns.find(cursor); run != residentRuns.end()) evict(run);
                if (!VirtualFree(reinterpret_cast<void*>(cursor), allocationEnd - cursor, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) fail("release private guest memory");
            } else {
                throw std::runtime_error("unsupported guest mapping type");
            }
            cursor = allocationEnd;
        }
        const auto last = query(reinterpret_cast<std::uintptr_t>(query(end - 1).AllocationBase));
        const auto lastBase = reinterpret_cast<std::uintptr_t>(last.BaseAddress);
        if (lastBase + last.RegionSize > end) split(lastBase, end - lastBase);
        const auto first = query(address);
        split(address, std::min(bytes, reinterpret_cast<std::uintptr_t>(first.BaseAddress) + first.RegionSize - address));
        if (query(address).RegionSize != bytes && !VirtualFree(reinterpret_cast<void*>(address), bytes, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) fail("coalesce guest placeholders");
    }

    std::map<std::uintptr_t, std::uintptr_t> cleanRanges;
    std::map<std::uintptr_t, View> views;
    std::map<std::pair<std::uintptr_t, std::uint64_t>, Physical> physical;
    std::map<std::uintptr_t, Run> residentRuns;
    std::uint64_t sharedPages = 0;
    std::size_t granularity = 0x10000;
    bool residentDirect = true;
    bool watchResident = false;
    std::mutex mutex;
    AllocateFunction allocate = nullptr;
    MapFunction map = nullptr;
    UnmapFunction unmap = nullptr;
    CompareFunction compare = nullptr;
};

}
#endif

#endif
