// The NGG geometry program the GPU tests draw with (agc_driver_mesh_tests, agc_driver_gds_tests):
// triangles fetched from a vertex buffer of Vertex records, recompiled to a mesh shader.
#ifndef CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_NGGPROGRAM_HPP
#define CORE_LIBS_PRX_LIBSCEAGCDRIVER_TESTS_NGGPROGRAM_HPP

#include "Recompiler.hpp"
#include <array>
#include <cstdint>

namespace NggTest {

// The merged program (llvm-mc -triple=amdgcn -mcpu=gfx1030), laid out as the PS5 compiler's: the
// ES half stores its vertex id (v5) at LDS[thread * 4]; after the barrier each GS thread below its
// wave's primitive count (s3 [15:8]) reads its three vertex ids through the vertex offsets in v0/v1,
// fetches position and color (32-byte records, V# in s[8:11]), writes them to LDS at
// 1024 + (3 * thread + k) * 32 and packs its triangle; wave 0 sends GS_ALLOC_REQ for the subgroup's
// primitives (GS_TG_INFO, s2 [30:22]) and three vertices each; the primitive threads export `prim`,
// the vertex threads read the LDS records back and export pos0 and param0.
//
//   s_lshl_b32 vcc_lo, s3, 16 / s_bfe_u64 exec, -1, vcc_lo / s_cbranch_execz es_done
//   v_mbcnt_lo_u32_b32 v6, -1, 0 / v_mbcnt_hi_u32_b32 v6, -1, v6 / s_bfe_u32 vcc_hi, s3, 0x40018
//   v_lshl_add_u32 v6, vcc_hi, 6, v6 / v_lshlrev_b32 v6, 2, v6 / ds_write_b32 v6, v5
// es_done:
//   s_waitcnt lgkmcnt(0) / s_mov_b64 exec, -1 / s_barrier
//   s_bfe_u32 s13, s2, 0x90016 / s_bfe_u32 s2, s3, 0x40018 / s_bfe_u32 s12, s3, 0x80008
//   v_mbcnt_lo_u32_b32 v9, -1, 0 / v_mbcnt_hi_u32_b32 v9, -1, v9 / v_lshl_add_u32 v10, s2, 6, v9
//   v_cmpx_gt_u32 s12, v9 / s_cbranch_execz gs_done
//   v_and_b32 v11, 0xffff, v0 / v_lshrrev_b32 v12, 16, v0 / v_and_b32 v13, 0xffff, v1
//   ds_read_b32 v11, v11 / ds_read_b32 v12, v12 / ds_read_b32 v13, v13 / s_waitcnt lgkmcnt(0)
//   buffer_load_dwordx4 v[16:19], v11, s[8:11], 0 idxen (and offset:16 into v[20:23]; v12 into
//   v[24:31], v13 into v[32:39])
//   v_mul_u32_u24 v14, 3, v10 / v_mul_u32_u24 v15, 0x60, v10 / s_waitcnt vmcnt(0)
//   ds_write_b128 v15, v[16:19] offset:1024 ... v[36:39] offset:1104
//   v_add_nc_u32 v40, 1, v14 / v_add_nc_u32 v41, 2, v14 / v_lshlrev_b32 v40, 10, v40
//   v_lshlrev_b32 v41, 20, v41 / v_or3_b32 v42, v14, v40, v41
// gs_done:
//   s_waitcnt lgkmcnt(0) / s_mov_b64 exec, -1 / s_barrier / s_mul_i32 s14, s13, 3
//   s_cmp_lg_u32 s2, 0 / s_cbranch_scc1 alloc_done / s_lshl_b32 s15, s13, 12 / s_or_b32 m0, s14, s15
//   s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
// alloc_done:
//   v_cmpx_gt_u32 s13, v10 / s_cbranch_execz prim_done / exp prim v42, off, off, off done
// prim_done:
//   s_mov_b64 exec, -1 / v_cmpx_gt_u32 s14, v10 / s_cbranch_execz end / v_lshlrev_b32 v43, 5, v10
//   ds_read_b128 v[44:47], v43 offset:1024 / ds_read_b128 v[48:51], v43 offset:1040
//   s_waitcnt lgkmcnt(0) / exp pos0 v44, v45, v46, v47 done / exp param0 v48, v49, v50, v51
// end:
//   s_endpgm
alignas(256) inline constexpr std::array<std::uint32_t, 104> GeometryCode{
    0x8f6a9003, 0x94fe6ac1, 0xbf88000b, 0xd7650006, 0x000100c1, 0xd7660006, 0x00020cc1, 0x93ebff03,
    0x00040018, 0xd7460006, 0x04190c6b, 0x340c0c82, 0xd8340000, 0x00000506, 0xbf8cc07f, 0xbefe04c1,
    0xbf8a0000, 0x938dff02, 0x00090016, 0x9382ff03, 0x00040018, 0x938cff03, 0x00080008, 0xd7650009,
    0x000100c1, 0xd7660009, 0x000212c1, 0xd746000a, 0x04250c02, 0x7da8120c, 0xbf88002e, 0x361600ff,
    0x0000ffff, 0x2c180090, 0x361a02ff, 0x0000ffff, 0xd8d80000, 0x0b00000b, 0xd8d80000, 0x0c00000c,
    0xd8d80000, 0x0d00000d, 0xbf8cc07f, 0xe0382000, 0x8002100b, 0xe0382010, 0x8002140b, 0xe0382000,
    0x8002180c, 0xe0382010, 0x80021c0c, 0xe0382000, 0x8002200d, 0xe0382010, 0x8002240d, 0x161c1483,
    0x161e14ff, 0x00000060, 0xbf8c3f70, 0xdb7c0400, 0x0000100f, 0xdb7c0410, 0x0000140f, 0xdb7c0420,
    0x0000180f, 0xdb7c0430, 0x00001c0f, 0xdb7c0440, 0x0000200f, 0xdb7c0450, 0x0000240f, 0x4a501c81,
    0x4a521c82, 0x3450508a, 0x34525294, 0xd772002a, 0x04a6510e, 0xbf8cc07f, 0xbefe04c1, 0xbf8a0000,
    0x930e830d, 0xbf078002, 0xbf850003, 0x8f0f8c0d, 0x887c0f0e, 0xbf900009, 0x7da8140d, 0xbf880002,
    0xf8000941, 0x0000002a, 0xbefe04c1, 0x7da8140e, 0xbf88000a, 0x34561485, 0xdbfc0400, 0x2c00002b,
    0xdbfc0410, 0x3000002b, 0xbf8cc07f, 0xf80008cf, 0x2f2e2d2c, 0xf800020f, 0x33323130, 0xbf810000,
};

struct Vertex {
    std::array<float, 4> position;
    std::array<float, 4> color;
};

inline std::array<std::uint32_t, 4> VertexBufferDescriptor(const void* vertices, std::uint32_t count) {
    const auto address = reinterpret_cast<std::uintptr_t>(vertices);
    // Stride 32, structured (OOB_SELECT 0), DST_SEL XYZW, FORMAT 32_FLOAT, RESOURCE_LEVEL.
    return {static_cast<std::uint32_t>(address), static_cast<std::uint32_t>((address >> 32u) & 0xffffu) | (32u << 16u), count, 0x01016facu};
}

// Four triangles (twelve ES vertices) per subgroup of one wave: forty triangles take ten workgroups.
inline constexpr ShaderRecompiler::MeshConfiguration SmallSubgroup{4u, 4u, 12u, 12u, 4u, 64u, 1024u, 0u, 4u};

}

#endif
