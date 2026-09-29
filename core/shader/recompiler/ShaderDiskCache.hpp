#ifndef CORE_SHADER_RECOMPILER_SHADERDISKCACHE_HPP
#define CORE_SHADER_RECOMPILER_SHADERDISKCACHE_HPP

#include "CompiledVariant.hpp"
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

// The shader disk cache: compiled variants (CompiledVariant.hpp) persisted across runs, so a
// program the title used before loads its SPIR-V and metadata instead of running the front end,
// the emitter and the optimizer again (hundreds of ms for the largest compute programs).
//
// What is cached is the variant, not a materialized result: a variant depends only on the program
// (the code, the stage registers, the user data count, the target and the binding layout) and on
// the resource specialization, which is the part of the captured memory the compile reads (formats,
// strides, image shapes and swizzles). The captured words themselves (V#/T# addresses, flattened
// SRT data) only reach the per-snapshot binding population, which runs on every load as on every
// in-memory hit; guest addresses also move between runs, so keying on them would never hit.
//
// Key: every byte the variant depends on (BuildKey). An entry's file name is a 128-bit hash of the
// key, and the file carries the key itself, so a name collision reads as a miss.
// File: a header (magic, format, source version, sizes, checksums), the key, the payload. An entry
// whose magic, format, version, size, key or checksum disagrees, or whose payload does not decode
// exactly, is ignored (and replaced by the next compile's write).
// Version: Generated::SourceVersion, a hash of the recompiler and driver sources made at build
// time (ShaderCacheVersion.cmake); entries live under a directory per version, and directories no
// build used for two weeks are removed.
//
// Location: ShaderCacheDirectory() (ShaderCacheDirectory.hpp); APS5_NO_SHADER_DISK_CACHE=1
// disables the cache. Loads run on the compiling thread (a file read, ~0.1 ms per 100 KiB); writes go
// to a background thread.
namespace ShaderRecompiler::ShaderDiskCache {

// The file format; bumped by hand only when the layout below changes (the source version already
// changes with every source edit).
inline constexpr std::uint32_t FormatVersion = 2;

enum class LoadStatus {
    Loaded,
    // No entry for the key.
    Absent,
    // A file for another key under the same name (a hash collision).
    KeyMismatch,
    // Truncated, corrupt, another format or another source version.
    Rejected,
};

struct Counters {
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t writes = 0;
    std::uint64_t loadFailures = 0;
    std::uint64_t writeFailures = 0;
    std::uint64_t bytesRead = 0;
    std::uint64_t bytesWritten = 0;
};

// The source version entries are written under (Generated::SourceVersion).
[[nodiscard]] std::uint64_t SourceVersion();

// The key of the variant a request compiles to under `specialization`: the format and source
// versions, the recompiler's in-memory key (stage, code hash, stage registers, user data count,
// target), every code word, the host subgroup size, the binding layout, the specialization and the
// values of the APS5_* switches the recompiler reads that may change its output.
void BuildKey(const RecompileRequest& request, std::uint32_t hostSubgroupSize, const ResourceSpecialization& specialization, std::vector<std::byte>& key);

// The entry's file name (32 hex digits and ".bin") for a key.
[[nodiscard]] std::string EntryName(std::span<const std::byte> key);

// The serialized result: the SPIR-V and every field the driver reads (bindings with their guest
// descriptors and per-element flags, push constants, vertex attributes, offset SGPRs, parameter
// exports, fragment parameters). `cacheHit` and `variantId` are per-process and not stored.
void EncodeResult(const RecompileResult& result, std::vector<std::byte>& out);
// False unless `bytes` is exactly one encoded result.
[[nodiscard]] bool DecodeResult(std::span<const std::byte> bytes, RecompileResult& result);

// A whole entry file for `key`: the variant's result, compiled shader info and binding allocation
// (its specialization and layout are part of the key).
[[nodiscard]] std::vector<std::byte> EncodeEntry(std::span<const std::byte> key, const CompiledVariant& variant);
// Decodes an entry file for `key` into `variant` (all but its specialization and layout, which the
// caller supplies, and the result's variantId).
[[nodiscard]] LoadStatus DecodeEntry(std::span<const std::byte> file, std::span<const std::byte> key, CompiledVariant& variant);

// Whether the cache is on (a directory is known and APS5_NO_SHADER_DISK_CACHE is not set); fixed
// at the first call.
[[nodiscard]] bool Enabled();
// The directory entries of this source version go to (empty when disabled).
[[nodiscard]] std::filesystem::path EntryDirectory();

// The stored variant for `key`, when there is a valid one; false (a miss, or a load failure that is
// counted and reported) otherwise.
[[nodiscard]] bool Load(std::span<const std::byte> key, CompiledVariant& variant);
// Queues the variant for the background writer.
void Store(std::vector<std::byte> key, std::shared_ptr<const CompiledVariant> variant);
// Waits until every queued write finished (tests).
void Flush();
// The totals since the process started.
[[nodiscard]] Counters Totals();

}

#endif
