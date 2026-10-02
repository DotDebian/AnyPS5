#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "prx/libc/include/General.hpp"
#include "prx/libSceNgs2.native/include/Ngs2Types.hpp"

// Geom and Pan helpers: listener/source math and speaker panning over the caller's work buffers.
// They keep no state in the system, rack or voice objects.

namespace {

constexpr std::int32_t kOk = 0;
constexpr std::int32_t kErrInvalidParam = static_cast<std::int32_t>(0x804A0001);

// Geom: source/listener helper math (documented layouts from the reference).
struct GVec { float x, y, z; };
struct GListenerParam { GVec pos, front, up, vel; float sound_speed; uint32_t reserved[2]; };
struct GListenerWork { float pos[3], right[3], up[3], front[3], vel[3], sound_speed; uint32_t coordinate; };
struct GSourceParam {
    GVec pos, vel, dir; float cone[4]; uint32_t rolloff_model; float max_dist, rolloff_factor, ref_dist;
    float doppler, fbw, lfe, max_level, min_level, radius; uint32_t num_speakers, matrix_format, reserved[2];
};

struct PanWorkL { float angles[8]; float unit; uint32_t n; };
struct PanParamL { float angle, distance, fbw, lfe; };

}  // namespace

#pragma GCC visibility push(default)

extern "C" {

int APS5_VABI sceNgs2GeomResetListenerParam(Ngs2GeomListenerParam* out) {
    if (out == nullptr) return kErrInvalidParam;
    auto* p = reinterpret_cast<GListenerParam*>(out);
    std::memset(p, 0, sizeof(GListenerParam));
    p->front = {0, 0, 1};
    p->up = {0, 1, 0};
    p->sound_speed = 343.0f;
    return kOk;
}
int APS5_VABI sceNgs2GeomResetSourceParam(Ngs2GeomSourceParam* out) {
    if (out == nullptr) return kErrInvalidParam;
    auto* p = reinterpret_cast<GSourceParam*>(out);
    std::memset(p, 0, sizeof(GSourceParam));
    p->dir = {0, 0, 1};
    p->cone[0] = 1.0f; p->cone[1] = 360.0f; p->cone[2] = 1.0f; p->cone[3] = 360.0f;
    p->max_dist = 1000000.0f; p->rolloff_factor = 1.0f; p->ref_dist = 1.0f;
    p->doppler = 1.0f; p->fbw = 1.0f; p->lfe = 1.0f; p->max_level = 1.0f; p->min_level = 0.0f;
    p->num_speakers = 2; p->matrix_format = 2;
    return kOk;
}
int APS5_VABI sceNgs2GeomCalcListener(const Ngs2GeomListenerParam* param, Ngs2GeomListenerWork* out_work, uint32_t flags) {
    if (param == nullptr || out_work == nullptr) return kErrInvalidParam;
    const auto* p = reinterpret_cast<const GListenerParam*>(param);
    auto* w = reinterpret_cast<GListenerWork*>(out_work);
    std::memset(out_work, 0, sizeof(GListenerWork));
    auto norm = [](GVec v) { float l = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); return l > 1e-6f ? GVec{v.x / l, v.y / l, v.z / l} : GVec{0, 0, 1}; };
    const GVec f = norm(p->front);
    GVec u = norm(p->up);
    GVec r = {u.y * f.z - u.z * f.y, u.z * f.x - u.x * f.z, u.x * f.y - u.y * f.x};  // up x front
    r = norm(r);
    u = {f.y * r.z - f.z * r.y, f.z * r.x - f.x * r.z, f.x * r.y - f.y * r.x};
    w->pos[0] = p->pos.x; w->pos[1] = p->pos.y; w->pos[2] = p->pos.z;
    w->right[0] = r.x; w->right[1] = r.y; w->right[2] = r.z;
    w->up[0] = u.x; w->up[1] = u.y; w->up[2] = u.z;
    w->front[0] = f.x; w->front[1] = f.y; w->front[2] = f.z;
    w->vel[0] = p->vel.x; w->vel[1] = p->vel.y; w->vel[2] = p->vel.z;
    w->sound_speed = p->sound_speed > 0 ? p->sound_speed : 343.0f;
    w->coordinate = flags & 1u;
    return kOk;
}
int APS5_VABI sceNgs2GeomApply(const Ngs2GeomListenerWork* listener, const Ngs2GeomSourceParam* source, Ngs2GeomAttribute* out_attrib, uint32_t flags) {
    (void)flags;
    if (listener == nullptr || source == nullptr || out_attrib == nullptr) return kErrInvalidParam;
    const auto* w = reinterpret_cast<const GListenerWork*>(listener);
    const auto* s = reinterpret_cast<const GSourceParam*>(source);
    // out: float pitch_ratio; float level[64]; ...
    auto* out = reinterpret_cast<float*>(out_attrib);
    std::memset(out_attrib, 0, sizeof(Ngs2GeomAttribute));
    float* level = out + 1;
    float d[3] = {s->pos.x - w->pos[0], s->pos.y - w->pos[1], s->pos.z - w->pos[2]};
    const float dist = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    const float lx = d[0] * w->right[0] + d[1] * w->right[1] + d[2] * w->right[2];
    const float lz = d[0] * w->front[0] + d[1] * w->front[1] + d[2] * w->front[2];
    // attenuation (inverse-distance rolloff between reference and max distance)
    float att = 1.0f;
    const float ref = s->ref_dist > 0 ? s->ref_dist : 1.0f;
    if (dist > ref) att = ref / (ref + s->rolloff_factor * (std::min(dist, s->max_dist > 0 ? s->max_dist : dist) - ref));
    const float maxl = s->max_level > 0 ? s->max_level : 1.0f;
    att = std::clamp(att * maxl, s->min_level, std::max(maxl, s->min_level));
    // constant-power stereo pan from the azimuth
    float pan = 0.0f;  // -1 left .. +1 right
    if (dist > 1e-4f) pan = std::clamp(lx / std::sqrt(lx * lx + lz * lz + 1e-12f), -1.0f, 1.0f);
    if (dist <= std::max(s->radius, 1e-4f)) pan = 0.0f;
    const float ang = (pan + 1.0f) * 0.25f * 3.14159265f;  // 0..pi/2
    const float gl = std::cos(ang) * 1.41421356f * 0.7071f * att * (s->fbw > 0 ? s->fbw : 1.0f);
    const float gr = std::sin(ang) * 1.41421356f * 0.7071f * att * (s->fbw > 0 ? s->fbw : 1.0f);
    level[0] = gl; level[1] = gr;      // in0 -> L,R
    level[2] = 0.0f; level[3] = gr;    // in1 -> R (second channel of a stereo source), stride-2 reading
    level[8] = 0.0f; level[9] = gr;    // same for a stride-8 reading
    // doppler
    float pitch = 1.0f;
    if (s->doppler > 0 && dist > 1e-4f) {
        const float c = w->sound_speed > 0 ? w->sound_speed : 343.0f;
        const float vl = (w->vel[0] * d[0] + w->vel[1] * d[1] + w->vel[2] * d[2]) / dist;
        const float vs = (s->vel.x * d[0] + s->vel.y * d[1] + s->vel.z * d[2]) / dist;
        const float denom = c - std::min(vs, c * 0.9f) * s->doppler;
        if (denom > 1.0f) pitch = std::clamp((c - std::min(vl, c * 0.9f) * s->doppler) / denom, 0.5f, 2.0f);
    }
    out[0] = pitch;
    return kOk;
}

int APS5_VABI sceNgs2PanInit(Ngs2PanWork* work, const float* speaker_angles, float unit_angle, uint32_t num_speakers) {
    if (work == nullptr) return kErrInvalidParam;
    std::memset(work, 0, sizeof(Ngs2PanWork));
    auto* w = reinterpret_cast<PanWorkL*>(work);
    w->unit = unit_angle;
    w->n = std::min<uint32_t>(num_speakers, 8);
    if (speaker_angles != nullptr) for (uint32_t i = 0; i < w->n; ++i) w->angles[i] = speaker_angles[i];
    return kOk;
}
int APS5_VABI sceNgs2PanGetVolumeMatrix(Ngs2PanWork* work, const Ngs2PanParam* params, uint32_t num_params, uint32_t matrix_format, float* out) {
    if (work == nullptr || (num_params && (params == nullptr || out == nullptr))) return kErrInvalidParam;
    const auto* w = reinterpret_cast<const PanWorkL*>(work);
    const auto* pp = reinterpret_cast<const PanParamL*>(params);
    const uint32_t ns = w->n ? w->n : (matrix_format ? std::min<uint32_t>(matrix_format, 8) : 2);
    const float pi2 = 6.28318531f;
    for (uint32_t p = 0; p < num_params; ++p) {
        float* row = out + static_cast<size_t>(p) * ns;
        for (uint32_t i = 0; i < ns; ++i) row[i] = 0;
        const float a = pp[p].angle;
        // find the two speakers bracketing the angle (angles in radians, wrapped)
        float best0 = 1e9f, best1 = 1e9f;
        int i0 = -1, i1 = -1;
        for (uint32_t i = 0; i < ns; ++i) {
            float diff = std::fmod(a - (w->n ? w->angles[i] : (i == 0 ? 0.5236f : -0.5236f)), pi2);
            if (diff > pi2 / 2) diff -= pi2;
            if (diff < -pi2 / 2) diff += pi2;
            const float ad = std::fabs(diff);
            if (ad < best0) { best1 = best0; i1 = i0; best0 = ad; i0 = static_cast<int>(i); }
            else if (ad < best1) { best1 = ad; i1 = static_cast<int>(i); }
        }
        const float lvl = pp[p].fbw > 0 ? pp[p].fbw : 1.0f;
        if (i0 >= 0 && i1 >= 0 && best0 + best1 > 1e-6f) {
            const float t = best0 / (best0 + best1);  // 0 -> fully speaker0
            row[i0] = std::cos(t * 1.5707963f) * lvl;
            row[i1] = std::sin(t * 1.5707963f) * lvl;
        } else if (i0 >= 0) row[i0] = lvl;
        if (ns >= 4 && pp[p].lfe > 0) row[3] = pp[p].lfe;
    }
    return kOk;
}

}

#pragma GCC visibility pop
