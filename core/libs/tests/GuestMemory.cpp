#include "prx/libc/include/general/VabiMacros.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "SceTypes.hpp"
#include <cstring>
#include <exception>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <source_location>
#include <thread>
#include <utility>
#include <vector>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

extern "C" {
void* APS5_VABI mmap_nid_postfix(void*, std::size_t, int, int, int, std::int64_t) noexcept;
int APS5_VABI munmap_nid_postfix(void*, std::size_t) noexcept;
int* APS5_VABI __error_nid_postfix();
int APS5_VABI sceKernelMapNamedFlexibleMemory(void**, std::size_t, int, int, const char*);
int APS5_VABI sceKernelMapFlexibleMemory(void**, std::size_t, int, int);
int APS5_VABI sceKernelMunmap(void*, std::size_t);
int APS5_VABI sceKernelMprotect(const void*, std::size_t, int);
int APS5_VABI sceKernelVirtualQuery(const void*, int, VirtualQueryInfo*, std::uint64_t);
int APS5_VABI sceKernelSetVirtualRangeName(const void*, std::uint64_t, const char*);
int APS5_VABI sceKernelClearVirtualRangeName(const void*, std::uint64_t);
}

static void Require(bool condition, std::source_location location = std::source_location::current()) {
    if (!condition) {
        std::fprintf(stderr, "Guest memory check failed at %s:%u\n", location.file_name(), static_cast<unsigned>(location.line()));
        std::abort();
    }
}

static const char* NameAt(const void* address) {
    static VirtualQueryInfo info;
    Require(sceKernelVirtualQuery(address, 0, &info, sizeof(info)) == 0);
    return info.name;
}

static void CheckNamedAndHintedMappings() {
    constexpr std::size_t length = 0x10000;
    void* first = nullptr;
    Require(sceKernelMapNamedFlexibleMemory(&first, length, 3, 0, "first mapping") == 0);
    Require(std::strcmp(NameAt(first), "first mapping") == 0);
    auto* middle = static_cast<unsigned char*>(first) + 0x4000;
    Require(sceKernelSetVirtualRangeName(middle, 0x4000, "middle") == 0);
    Require(std::strcmp(NameAt(first), "first mapping") == 0);
    Require(std::strcmp(NameAt(middle), "middle") == 0);
    Require(sceKernelClearVirtualRangeName(first, length) == 0);
    Require(NameAt(middle)[0] == '\0');
    Require(sceKernelSetVirtualRangeName(nullptr, length, "x") != 0);
#if defined(__linux__)
    void* hinted = first;
    Require(sceKernelMapFlexibleMemory(&hinted, length, 3, 0) == 0);
    Require(hinted > first && (reinterpret_cast<std::uintptr_t>(hinted) & 0x3fff) == 0);
    bool rejected = false;
    void* overwrite = first;
    try { sceKernelMapFlexibleMemory(&overwrite, 0x4000, 3, 0x90); } catch (const std::exception&) { rejected = true; }
    Require(rejected && overwrite == first);
    Require(sceKernelMunmap(hinted, length) == 0);
#endif
    Require(sceKernelMunmap(first, length) == 0);
}

#if defined(__linux__)
using PageRuns = std::vector<std::pair<std::uintptr_t, std::uintptr_t>>;

static bool CollectRuns(const void* base, std::size_t offset, std::size_t bytes, PageRuns& runs) {
    runs.clear();
    const auto address = reinterpret_cast<std::uintptr_t>(base);
    std::pair<std::uintptr_t, PageRuns*> context{address, &runs};
    return GuestWriteWatch::GuestWriteWatchCollect_nid_postfix(address + offset, bytes, [](void* context, std::uintptr_t begin, std::uintptr_t end) {
        auto& [origin, into] = *static_cast<std::pair<std::uintptr_t, PageRuns*>*>(context);
        if (!into->empty() && into->back().second == (begin - origin) / 4096) into->back().second = (end - origin) / 4096;
        else into->emplace_back((begin - origin) / 4096, (end - origin) / 4096);
    }, &context);
}

static bool Written(const void* base, std::size_t bytes, PageRuns expected) {
    PageRuns runs;
    const bool complete = CollectRuns(base, 0, bytes, runs);
    if (complete && runs == expected) return true;
    std::fprintf(stderr, "write watch collect %s, written pages:", complete ? "complete" : "incomplete");
    for (const auto& [first, last] : runs) std::fprintf(stderr, " [%zu, %zu)", static_cast<std::size_t>(first), static_cast<std::size_t>(last));
    std::fputs("\n", stderr);
    return false;
}

static void CheckWriteWatch() {
    if (!GuestWriteWatch::GuestWriteWatchAvailable_nid_postfix()) {
        std::puts("write watch unavailable: not tested");
        return;
    }
    constexpr std::size_t length = 0x100000;
    constexpr std::size_t small = 4096;
    void* mapping = nullptr;
    Require(sceKernelMapFlexibleMemory(&mapping, length, 3, 0) == 0);
    auto* bytes = static_cast<volatile unsigned char*>(mapping);
    const auto address = reinterpret_cast<std::uintptr_t>(mapping);
    Require(GuestWriteWatch::GuestWriteWatchCovers_nid_postfix(address, length));
    Require(GuestWriteWatch::GuestWriteWatchCovers_nid_postfix(address + small, small));
    Require(!GuestWriteWatch::GuestWriteWatchCovers_nid_postfix(address, length + small));
    Require(!GuestWriteWatch::GuestWriteWatchCovers_nid_postfix(reinterpret_cast<std::uintptr_t>(&length), sizeof(length)));
    Require(Written(mapping, length, {{0, length / small}}));
    Require(Written(mapping, length, {}));
    bytes[5 * small + 17] = 1;
    Require(Written(mapping, length, {{5, 6}}));
    Require(Written(mapping, length, {}));
    static_cast<void>(bytes[10 * small]);
    Require(Written(mapping, length, {}));
    bytes[7 * small] = 1;
    bytes[9 * small] = 1;
    PageRuns runs;
    Require(CollectRuns(mapping, 8 * small, small, runs) && runs.empty());
    Require(CollectRuns(mapping, 7 * small + 100, 1, runs) && runs == PageRuns{{7, 8}});
    Require(Written(mapping, length, {{9, 10}}));
    int pipe[2];
    Require(::pipe(pipe) == 0);
    Require(::write(pipe[1], "kernel", 6) == 6);
    Require(::read(pipe[0], const_cast<unsigned char*>(bytes + 20 * small + 8), 6) == 6);
    ::close(pipe[0]);
    ::close(pipe[1]);
    Require(bytes[20 * small + 8] == 'k' && Written(mapping, length, {{20, 21}}));
    std::thread([&] { bytes[30 * small + 5] = 3; }).join();
    Require(Written(mapping, length, {{30, 31}}));
    std::vector<unsigned char> source(2 * small, 0xab);
    std::memcpy(const_cast<unsigned char*>(bytes + 40 * small + 2048), source.data(), source.size());
    Require(Written(mapping, length, {{40, 43}}));
    Require(sceKernelMprotect(mapping, length, 1) == 0);
    Require(sceKernelMprotect(mapping, length, 3) == 0);
    Require(Written(mapping, length, {}));
    bytes[50 * small] = 1;
    Require(Written(mapping, length, {{50, 51}}));
    constexpr std::size_t guestPage = 0x4000;
    auto* middle = const_cast<unsigned char*>(bytes + 4 * guestPage);
    Require(sceKernelMunmap(middle, guestPage) == 0);
    Require(!GuestWriteWatch::GuestWriteWatchCovers_nid_postfix(address, length));
    Require(!CollectRuns(mapping, 0, length, runs) && runs.empty());
    void* fixed = middle;
    Require(sceKernelMapFlexibleMemory(&fixed, guestPage, 3, 0x10) == 0 && fixed == middle);
    Require(GuestWriteWatch::GuestWriteWatchCovers_nid_postfix(address, length));
    Require(Written(mapping, length, {{16, 20}}));
    Require(Written(mapping, length, {}));
    middle[1] = 1;
    Require(Written(mapping, length, {{16, 17}}));
    Require(sceKernelMunmap(mapping, length) == 0);
    Require(!GuestWriteWatch::GuestWriteWatchCovers_nid_postfix(address, small));
    constexpr std::size_t tableSpan = 0x200000;
    constexpr std::size_t spanned = 2 * tableSpan;
    void* raw = mmap(nullptr, spanned + tableSpan, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    Require(raw != MAP_FAILED);
    const auto rawAddress = reinterpret_cast<std::uintptr_t>(raw);
    void* region = reinterpret_cast<void*>((rawAddress + tableSpan - 1) & ~(tableSpan - 1));
    GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(region, spanned);
    Require(GuestWriteWatch::GuestWriteWatchCovers_nid_postfix(reinterpret_cast<std::uintptr_t>(region), spanned));
    Require(Written(region, spanned, {{0, spanned / small}}));
    Require(Written(region, spanned, {}));
    static_cast<volatile unsigned char*>(region)[tableSpan + 3 * small] = 1;
    static_cast<volatile unsigned char*>(region)[7 * small] = 1;
    Require(Written(region, spanned, {{7, 8}, {tableSpan / small + 3, tableSpan / small + 4}}));
    Require(Written(region, spanned, {}));
    GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(region, spanned);
    Require(munmap(raw, spanned + tableSpan) == 0);
}
#endif

int main() {
    CheckNamedAndHintedMappings();
#if defined(__linux__)
    CheckWriteWatch();
#endif
    constexpr std::size_t page = 0x4000;
    const auto failed = reinterpret_cast<void*>(static_cast<std::uintptr_t>(-1));
    const auto reject = [&](std::size_t length, int protection, int flags, int fd,
                            std::int64_t offset, int error) {
        *__error_nid_postfix() = 0;
        Require(mmap_nid_postfix(nullptr, length, protection, flags, fd, offset) == failed);
        Require(*__error_nid_postfix() == error);
    };
    reject(0, 3, 0x1002, -1, 0, 22);
    reject(std::numeric_limits<std::size_t>::max(), 3, 0x1002, -1, 0, 22);
    reject(page, 8, 0x1002, -1, 0, 22);
    reject(page, 3, 0x1002, 0, 0, 22);
    reject(page, 3, 0x1002, -1, 1, 22);
    reject(page, 3, 0x1001, -1, 0, 45); // shared
    reject(page, 3, 0x1012, -1, 0, 45); // fixed
    reject(page, 3, 0x2, 0, 0, 45);    // file-backed
    reject(page, 3, 0x22, -1, 0, 45);  // Linux MAP_ANON is not guest MAP_ANON

    auto* memory = static_cast<unsigned char*>(mmap_nid_postfix(nullptr, page * 3 - 1, 3, 0x1002, -1, 0));
    Require(memory != failed && (reinterpret_cast<std::uintptr_t>(memory) & (page - 1)) == 0);
    for (std::size_t i = 0; i < page * 3; ++i) Require(memory[i] == 0);
    memory[0] = 42;
    memory[page * 2] = 73;
    {
        GuestAllocations::Mutation mutation;
        const auto range = mutation.Find(memory);
        Require(range.bytes == page * 3 && range.readable && range.writable);
    }
    Require(munmap_nid_postfix(memory + 1, page) == -1 && *__error_nid_postfix() == 22);
    Require(munmap_nid_postfix(memory, 0) == -1 && *__error_nid_postfix() == 22);
    Require(memory[0] == 42);
    Require(munmap_nid_postfix(memory + page, 1) == 0); // round to one guest page
    Require(memory[0] == 42 && memory[page * 2] == 73);
    Require(munmap_nid_postfix(memory, page) == 0);
    Require(memory[page * 2] == 73);
    Require(munmap_nid_postfix(memory + page * 2, page) == 0);
    Require(munmap_nid_postfix(memory, page) == -1);
    for (int protection : {0, 1, 3, 5}) {
        void* mapped = mmap_nid_postfix(memory, 1, protection, 0x1002, -1, 0);
        Require(mapped != failed);
        {
            GuestAllocations::Mutation mutation;
            const auto range = mutation.Find(mapped);
            Require(range.readable == ((protection & 3) != 0));
            Require(range.writable == ((protection & 2) != 0));
        }
        Require(munmap_nid_postfix(mapped, 1) == 0);
    }
}
