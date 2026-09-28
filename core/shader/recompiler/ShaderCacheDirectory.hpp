#ifndef CORE_SHADER_RECOMPILER_SHADERCACHEDIRECTORY_HPP
#define CORE_SHADER_RECOMPILER_SHADERCACHEDIRECTORY_HPP

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

// Where the persistent caches live, shared by the shader disk cache (ShaderDiskCache.hpp) and the
// AGC driver's Vulkan pipeline cache (Graphics/include/PipelineCache.hpp). Header only, so the
// driver's standalone test builds that do not link the recompiler can use it too.
namespace ShaderRecompiler {

// $APS5_SHADER_CACHE_DIR, else $XDG_CACHE_HOME/anyps5/shaders, else ~/.cache/anyps5/shaders; empty
// when APS5_NO_SHADER_DISK_CACHE is set (to anything but 0) or no home directory is known.
inline std::filesystem::path ShaderCacheDirectory() {
    const auto set = [](const char* name) -> const char* {
        const char* value = std::getenv(name);
        return value != nullptr && *value != '\0' ? value : nullptr;
    };
    if (const char* disabled = set("APS5_NO_SHADER_DISK_CACHE"); disabled != nullptr && std::strcmp(disabled, "0") != 0) return {};
    if (const char* directory = set("APS5_SHADER_CACHE_DIR")) return std::filesystem::path(directory);
    if (const char* xdg = set("XDG_CACHE_HOME")) return std::filesystem::path(xdg) / "anyps5" / "shaders";
    if (const char* home = set("HOME")) return std::filesystem::path(home) / ".cache" / "anyps5" / "shaders";
    return {};
}

// Writes `bytes` to a temporary file beside `path` and renames it over `path`, so a reader (this
// process, a concurrent one, or the next run after a kill) sees either the old file or the whole
// new one. The directory is created when missing.
inline bool WriteFileAtomically(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    static std::atomic<std::uint64_t> serial{0};
    const auto unique = std::hash<std::thread::id>{}(std::this_thread::get_id()) ^ static_cast<std::size_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    auto temporary = path;
    temporary += ".tmp." + std::to_string(unique) + "." + std::to_string(serial.fetch_add(1, std::memory_order_relaxed));
    std::FILE* file = std::fopen(temporary.string().c_str(), "wb");
    if (file == nullptr) return false;
    const bool written = std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
    const bool closed = std::fclose(file) == 0;
    if (!written || !closed) {
        std::filesystem::remove(temporary, error);
        return false;
    }
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        return false;
    }
    return true;
}

// The whole file, or nothing when it cannot be opened or read.
inline bool ReadWholeFile(const std::filesystem::path& path, std::vector<std::byte>& bytes) {
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    if (file == nullptr) return false;
    bytes.clear();
    bool ok = std::fseek(file, 0, SEEK_END) == 0;
    const long size = ok ? std::ftell(file) : -1;
    ok = ok && size >= 0 && std::fseek(file, 0, SEEK_SET) == 0;
    if (ok) {
        bytes.resize(static_cast<std::size_t>(size));
        ok = std::fread(bytes.data(), 1, bytes.size(), file) == bytes.size();
    }
    std::fclose(file);
    if (!ok) bytes.clear();
    return ok;
}

// A 64-bit hash of `bytes` (a checksum and file name source, not a cryptographic digest).
inline std::uint64_t HashBytes(std::span<const std::byte> bytes, std::uint64_t seed = 0) {
    constexpr std::uint64_t multiplier = 0x9e3779b97f4a7c15ull;
    std::uint64_t hash = seed ^ (static_cast<std::uint64_t>(bytes.size()) * multiplier);
    const auto mix = [&](std::uint64_t chunk) {
        chunk *= 0xff51afd7ed558ccdull;
        chunk ^= chunk >> 32u;
        hash = (hash ^ chunk) * multiplier;
        hash = (hash << 27u) | (hash >> 37u);
    };
    std::size_t index = 0;
    for (; index + 8 <= bytes.size(); index += 8) {
        std::uint64_t chunk;
        std::memcpy(&chunk, bytes.data() + index, sizeof(chunk));
        mix(chunk);
    }
    if (index < bytes.size()) {
        std::uint64_t chunk = 0;
        std::memcpy(&chunk, bytes.data() + index, bytes.size() - index);
        mix(chunk ^ 0x8000000000000000ull);
    }
    // The fmix64 finalizer of MurmurHash3.
    hash ^= hash >> 33u;
    hash *= 0xff51afd7ed558ccdull;
    hash ^= hash >> 33u;
    hash *= 0xc4ceb9fe1a85ec53ull;
    hash ^= hash >> 33u;
    return hash;
}

}

#endif
