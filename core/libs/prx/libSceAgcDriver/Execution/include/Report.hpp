#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_REPORT_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_EXECUTION_INCLUDE_REPORT_HPP

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace AgcDriver {

// The driver's lines on stderr, one write each (local, not for upstream). stderr is unbuffered,
// and redirected to a file or a pipe the C runtime gives it no temporary buffer either, so a
// std::fprintf wrote its line one character per system call: a 700-character report line cost
// hundreds of ZwWriteFile calls on the thread that printed it (tens of milliseconds through a
// pipe), under the device lock for most reports. The line is formatted into memory first and
// handed to the stream by one fwrite, under the stream's own lock as before, so lines of different
// threads still do not interleave. Only the driver's own reports go through here; nothing about
// the stream itself changes, so every other writer of stderr behaves as it did.
// APS5_UNBUFFERED_REPORTS=1 prints through vfprintf as before.
#if defined(__MINGW32__)
inline void ReportLine(const char* format, ...) __attribute__((format(__MINGW_PRINTF_FORMAT, 1, 2)));
#elif defined(__GNUC__)
inline void ReportLine(const char* format, ...) __attribute__((format(printf, 1, 2)));
#endif

inline void ReportLine(const char* format, ...) {
    static const bool unbuffered = std::getenv("APS5_UNBUFFERED_REPORTS") != nullptr;
    va_list arguments;
    va_start(arguments, format);
    if (unbuffered) {
        std::vfprintf(stderr, format, arguments);
        va_end(arguments);
        return;
    }
    va_list again;
    va_copy(again, arguments);
    char line[2048];
    const int length = std::vsnprintf(line, sizeof(line), format, arguments);
    va_end(arguments);
    if (length > 0 && static_cast<std::size_t>(length) < sizeof(line)) {
        std::fwrite(line, 1, static_cast<std::size_t>(length), stderr);
    } else if (length > 0) {
        std::string longer(static_cast<std::size_t>(length) + 1, '\0');
        std::vsnprintf(longer.data(), longer.size(), format, again);
        std::fwrite(longer.data(), 1, static_cast<std::size_t>(length), stderr);
    }
    va_end(again);
}

}

#endif
