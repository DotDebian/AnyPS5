#ifndef CORE_LIBS_PRX_LIBC_INCLUDE_GUESTWRITEWATCH_HPP
#define CORE_LIBS_PRX_LIBC_INCLUDE_GUESTWRITEWATCH_HPP

#include <cstddef>
#include <cstdint>

// Page write watching of guest mappings on Linux, the counterpart of the Windows arena's
// MEM_WRITE_WATCH: guest mappings are registered with a userfaultfd in asynchronous write-protect
// mode (UFFD_FEATURE_WP_ASYNC), so the kernel lifts a page's protection on its first write without
// a fault handler, and PAGEMAP_SCAN reports the pages written since the last scan of a range while
// protecting them again (GetWriteWatch with WRITE_WATCH_FLAG_RESET). Pages never protected since
// they were mapped (a fresh mapping, one replaced by a fixed mapping, pages dropped by the kernel)
// report as written, so a mapping change reads as a write of its pages. Needs Linux 6.7; a process
// without CAP_SYS_PTRACE opens the userfaultfd with UFFD_USER_MODE_ONLY when unprivileged
// userfaultfds are restricted to it (vm.unprivileged_userfaultfd = 0), which the asynchronous mode
// does not need either way: kernel-mode writes (a read() into guest memory) lift protection too.
// APS5_NO_WRITE_WATCH=1 turns it off. Windows reports unavailable: its arena is watched instead.
namespace GuestWriteWatch {

extern "C" {

bool GuestWriteWatchAvailable_nid_postfix();
// Registers a guest mapping made with mmap (its pages report as written until first scanned), or
// forgets a range that was unmapped. A mapping the kernel refuses to register stays unwatched.
void GuestWriteWatchRegister_nid_postfix(const void* pointer, std::size_t bytes);
void GuestWriteWatchUnregister_nid_postfix(const void* pointer, std::size_t bytes);
// Whether every page of the range belongs to a registered mapping.
bool GuestWriteWatchCovers_nid_postfix(std::uintptr_t address, std::size_t bytes);
// Calls `written` with the runs of pages of [address, address + bytes) (page-rounded outward) that
// were written since they were last collected, and write-protects them again. Returns false when a
// page of the range is not registered (nothing is scanned) or the scan failed (a mapping changed
// under it): the pages of the range not reported yet are then reported as written, since the scan
// may have reset them.
bool GuestWriteWatchCollect_nid_postfix(std::uintptr_t address, std::size_t bytes, void (*written)(void* context, std::uintptr_t begin, std::uintptr_t end), void* context);

}

}

#endif
