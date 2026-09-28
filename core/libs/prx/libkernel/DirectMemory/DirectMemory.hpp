#ifndef CORE_LIBS_PRX_LIBKERNEL_DIRECTMEMORY_DIRECTMEMORY_HPP
#define CORE_LIBS_PRX_LIBKERNEL_DIRECTMEMORY_DIRECTMEMORY_HPP

#include <cstdint>
#include <cstddef>

static constexpr size_t DIRECT_MEMORY_SIZE = 13824ULL * 1024 * 1024;
static constexpr size_t PS5_PAGE_SIZE = 0x4000;

static constexpr int SCE_KERNEL_ERROR_EINVAL = 0x80020016;
static constexpr int SCE_KERNEL_ERROR_EAGAIN = 0x80020023;
static constexpr int SCE_KERNEL_ERROR_ENOMEM = 0x8002000C;
static constexpr int SCE_KERNEL_ERROR_EACCES = 0x8002000D;
static constexpr int SCE_KERNEL_ERROR_EFAULT = 0x8002000E;

// A block of GPU-visible ("direct") memory the guest asked for, remembered so that later
// queries can answer with the real extent and the type it was allocated as. A title that gets
// back a one-page extent or a type it never requested rejects its own graphics allocations.
struct DirectMemoryBlock {
    uint64_t start;
    uint64_t end;
    int memoryType;
};

// memoryType is what sceKernelAllocateDirectMemory was called with; -1 for a range the caller
// has no type for (the pool API and the available-size probe).
int DirectMemoryAlloc(int64_t searchStart, int64_t searchEnd, size_t len, size_t alignment, int memoryType, int64_t* physOut);
void DirectMemoryFree(int64_t start, size_t len);
// Block containing offset, as the guest was given it at allocation time.
bool DirectMemoryQueryBlock(uint64_t offset, DirectMemoryBlock* block);
// Contiguous unallocated bytes starting at offset, capped at limit (what the available-size
// query reports; a whole-aperture figure would let a title plan allocations that cannot fit).
size_t DirectMemoryFreeRun(uint64_t offset, uint64_t limit);
int DoMapDirect(void** addr, size_t len, int prot, int flags, int64_t physStart, size_t alignment);
int DoMapAnon(void** addr, size_t len, int prot, int flags);
int DoMprotect(const void* addr, size_t len, int prot);
int DoMunmap(void* addr, size_t len);
int DoReserveVirtual(void** addr, size_t len, size_t alignment);

#endif