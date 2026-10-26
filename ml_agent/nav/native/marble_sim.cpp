// Compiled port of the planner's marble physics (nav/marble_dynamics/physics.py GeneralPhysics.step, its contact
// generation and native contact order, and the compiled swept-sphere query of nav/collision/experimental_sweep.py).
// 2026-10-16 (operator: "extend the horizon ... while remaining completely performant and accurate to the physics").
//
// Goal: the SAME numbers as the Python port, step for step. Every expression keeps the Python evaluation order, and
// NumPy's reductions are reproduced as measured on this machine (scratchpad np_order*.py, 2026-10-16):
//   1-D float64 dot and np.linalg.norm:     (a0*b0 + a1*b1) + a2*b2
//   (N,3)@(3,) float64 (sweep pre-terms):   N >= 2: fma(a2,b2, fma(a0,b0, a1*b1));  N == 1: sequential
//   1-D float32 dot (OpenBLAS sdot):        float( double(f(a0*b0)) + double(f(a1*b1)) + double(f(a2*b2)) )
//   float32 3x3 matrix-vector:              sequential float32 sums
// Build with /fp:strict (no contraction; explicit std::fma only). cos/sin/pow are the CRT functions Python calls.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <cstdio>
#include <cstdlib>
#define NOMINMAX
#include <windows.h>

#pragma function(cos, sin, pow, sqrt, fmod)

#define DLL extern "C" __declspec(dllexport)

namespace {

struct V3 { double x, y, z; };
inline V3 add(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 sub(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 mul(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline V3 dvs(V3 a, double s) { return {a.x / s, a.y / s, a.z / s}; }
inline V3 neg(V3 a) { return {-a.x, -a.y, -a.z}; }
inline V3 load(const double* p) { return {p[0], p[1], p[2]}; }
inline void store(double* p, V3 a) { p[0] = a.x; p[1] = a.y; p[2] = a.z; }
// numpy 1-D float64 dot / norm
inline double dot(V3 a, V3 b) { return (a.x * b.x + a.y * b.y) + a.z * b.z; }
inline double norm(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 cross3(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline V3 unit(V3 v) { double n = norm(v); return n > 1e-12 ? dvs(v, n) : V3{0, 0, 0}; }
inline double f32(double x) { return (double)(float)x; }
// Python's min/max on two values (first argument wins ties)
inline double pymin(double a, double b) { return b < a ? b : a; }
inline double pymax(double a, double b) { return b > a ? b : a; }
inline double pymod(double x, double y) {
    double m = std::fmod(x, y);
    if (m != 0.0) { if ((y < 0) != (m < 0)) m += y; }
    else m = std::copysign(0.0, y);
    return m;
}
// (N,3)@(3,) row of a numpy float64 matmul
inline double mv_row(V3 a, V3 b, bool single) {
    if (single) return dot(a, b);
    return std::fma(a.z, b.z, std::fma(a.x, b.x, a.y * b.y));
}
// sweep kernel's internal einsum order (x, z, then y) -- kept from nav/collision/sweep_kernel.cpp
inline double kdot(V3 a, V3 b) { return (a.x * b.x + a.z * b.z) + a.y * b.y; }
inline double positive(double x) { return x > 0 ? x : 0; }
inline double f32root(double x) { return double(std::sqrt(float(x))); }

struct Material { double friction, restitution, force; };

struct Contact {
    V3 normal, point; double distance; int triangle; int polygon; Material mat; V3 plane;
};

struct Event {   // kind: 1 ambiguous, 2 support, 3 jump, 4 collision, 5 solver limit
    int kind; int sub; int triangle; V3 a, b; double value;
};

struct Poly {
    int nv, vstart, first_triangle, object; V3 normal; double d;
};

struct HullObject {
    float inv[16]; float scale[3]; float bbox_lo[3], bbox_hi[3];
    std::vector<float> hull_lo, hull_hi; int nhull = 0;
    int bin_start[256], bin_count[256]; std::vector<int> indices;
    std::vector<int> hp_off, hp;
};

struct Map {
    int ntri = 0;
    std::vector<double> tri, source, planes, lo, hi, ab, ac, fcross, nn;
    std::vector<uint8_t> boundary;
    std::vector<int> material_id, polygon_id; std::vector<long long> object_id;
    std::vector<Material> materials;
    std::vector<Poly> polys; std::vector<V3> pverts, pedges; std::vector<uint8_t> pvalid;
    std::unordered_map<long long, HullObject> objects;
    // broad phase: uniform xy grid of triangle ids (ascending in each cell)
    double gx0 = 0, gy0 = 0, cell = 2.0; int gnx = 0, gny = 0;
    std::vector<int> gstart, gids;
    std::vector<int> stamp; int stamp_id = 0;
};

// ------------------------------------------------------------------ broad phase (same set and order as np.flatnonzero)
void build_grid(Map& m, double cell) {
    m.cell = cell;
    double x0 = 1e300, y0 = 1e300, x1 = -1e300, y1 = -1e300;
    for (int i = 0; i < m.ntri; i++) {
        x0 = std::min(x0, m.lo[i * 3]); y0 = std::min(y0, m.lo[i * 3 + 1]);
        x1 = std::max(x1, m.hi[i * 3]); y1 = std::max(y1, m.hi[i * 3 + 1]);
    }
    m.gx0 = x0 - cell; m.gy0 = y0 - cell;
    m.gnx = (int)std::ceil((x1 - m.gx0) / cell) + 2; m.gny = (int)std::ceil((y1 - m.gy0) / cell) + 2;
    std::vector<std::vector<int>> cells((size_t)m.gnx * m.gny);
    for (int i = 0; i < m.ntri; i++) {
        int ax = (int)std::floor((m.lo[i * 3] - m.gx0) / cell), bx = (int)std::floor((m.hi[i * 3] - m.gx0) / cell);
        int ay = (int)std::floor((m.lo[i * 3 + 1] - m.gy0) / cell), by = (int)std::floor((m.hi[i * 3 + 1] - m.gy0) / cell);
        ax = std::max(0, ax); ay = std::max(0, ay); bx = std::min(m.gnx - 1, bx); by = std::min(m.gny - 1, by);
        for (int x = ax; x <= bx; x++) for (int y = ay; y <= by; y++) cells[(size_t)x * m.gny + y].push_back(i);
    }
    m.gstart.assign((size_t)m.gnx * m.gny + 1, 0); m.gids.clear();
    for (size_t c = 0; c < cells.size(); c++) { m.gstart[c] = (int)m.gids.size(); m.gids.insert(m.gids.end(), cells[c].begin(), cells[c].end()); }
    m.gstart[cells.size()] = (int)m.gids.size();
    m.stamp.assign(m.ntri, 0); m.stamp_id = 0;
}

inline bool box_hit(const Map& m, int i, const double* lo, const double* hi) {
    return m.hi[i * 3] >= lo[0] && m.lo[i * 3] <= hi[0] && m.hi[i * 3 + 1] >= lo[1] && m.lo[i * 3 + 1] <= hi[1]
        && m.hi[i * 3 + 2] >= lo[2] && m.lo[i * 3 + 2] <= hi[2];
}

void query_aabb(Map& m, const double* lo, const double* hi, std::vector<int>& out) {
    out.clear();
    if (!(std::isfinite(lo[0]) && std::isfinite(lo[1]) && std::isfinite(hi[0]) && std::isfinite(hi[1]))) {
        for (int i = 0; i < m.ntri; i++) if (box_hit(m, i, lo, hi)) out.push_back(i);
        return;
    }
    int ax = (int)std::floor((lo[0] - m.gx0) / m.cell), bx = (int)std::floor((hi[0] - m.gx0) / m.cell);
    int ay = (int)std::floor((lo[1] - m.gy0) / m.cell), by = (int)std::floor((hi[1] - m.gy0) / m.cell);
    ax = std::max(0, ax); ay = std::max(0, ay); bx = std::min(m.gnx - 1, bx); by = std::min(m.gny - 1, by);
    if (ax > bx || ay > by) return;
    if ((long long)(bx - ax + 1) * (by - ay + 1) > 64) {
        for (int i = 0; i < m.ntri; i++) if (box_hit(m, i, lo, hi)) out.push_back(i);
        return;
    }
    if (++m.stamp_id == 0x7fffffff) { std::fill(m.stamp.begin(), m.stamp.end(), 0); m.stamp_id = 1; }
    for (int x = ax; x <= bx; x++) for (int y = ay; y <= by; y++) {
        size_t c = (size_t)x * m.gny + y;
        for (int k = m.gstart[c]; k < m.gstart[c + 1]; k++) {
            int i = m.gids[k];
            if (m.stamp[i] == m.stamp_id) continue;
            m.stamp[i] = m.stamp_id;
            if (box_hit(m, i, lo, hi)) out.push_back(i);
        }
    }
    std::sort(out.begin(), out.end());
}

// ------------------------------------------------------------------ native contact order (contact_order.py)
inline float f32dot3_row(const float* M, int row, const float* x) {   // sum_j M[row][j]*x[j], float32 sequential
    float s = M[row * 4 + 0] * x[0];
    s = s + M[row * 4 + 1] * x[1];
    s = s + M[row * 4 + 2] * x[2];
    return s;
}

void ranks(const HullObject& o, V3 p, double r, std::unordered_map<int, int>& rank) {
    rank.clear();
    float lo[3] = {(float)((p.x - r) - .0001), (float)((p.y - r) - .0001), (float)((p.z - r) - .0001)};
    float hi[3] = {(float)((p.x + r) + .0001), (float)((p.y + r) + .0001), (float)((p.z + r) + .0001)};
    float mid[3], width[3], hw[3], absinv[16];
    for (int i = 0; i < 3; i++) { mid[i] = (lo[i] + hi[i]) * 0.5f; width[i] = hi[i] - lo[i]; hw[i] = width[i] * 0.5f; }
    for (int i = 0; i < 16; i++) absinv[i] = std::fabs(o.inv[i]);
    float center[3], broad[3], half[3], qlo[3], qhi[3], alo[3], ahi[3];
    for (int i = 0; i < 3; i++) {
        center[i] = (f32dot3_row(o.inv, i, mid) + o.inv[i * 4 + 3]) / o.scale[i];
        broad[i] = f32dot3_row(absinv, i, width) / o.scale[i];
        half[i] = f32dot3_row(absinv, i, hw) / o.scale[i];
        qlo[i] = center[i] - broad[i]; qhi[i] = center[i] + broad[i];
        alo[i] = center[i] - half[i]; ahi[i] = center[i] + half[i];
    }
    int starts[2], ends[2];
    for (int k = 0; k < 2; k++) {
        float bin_size = (o.bbox_hi[k] - o.bbox_lo[k]) / 16.0f;
        float a = (std::max(o.bbox_lo[k], qlo[k]) - o.bbox_lo[k]) / bin_size;
        float b = (std::min(o.bbox_hi[k], qhi[k]) - o.bbox_lo[k]) / bin_size;
        starts[k] = std::max(0, (int)std::floor(a));
        ends[k] = std::min(16, (int)std::ceil(b));
    }
    static thread_local std::vector<uint8_t> seen; seen.assign(o.nhull, 0);
    for (int x = starts[0]; x < ends[0]; x++) for (int y = starts[1]; y < ends[1]; y++) {
        int b = x * 16 + y;
        for (int k = o.bin_start[b]; k < o.bin_start[b] + o.bin_count[b]; k++) {
            int h = o.indices[k];
            if (seen[h]) continue;
            seen[h] = 1;
            const float* hl = &o.hull_lo[h * 3]; const float* hh = &o.hull_hi[h * 3];
            bool culled = false;
            for (int i = 0; i < 3; i++)
                if (hh[i] < qlo[i] || hl[i] > qhi[i] || hh[i] < alo[i] || hl[i] > ahi[i]) culled = true;
            if (culled) continue;
            for (int q = o.hp_off[h]; q < o.hp_off[h + 1]; q++) {
                int pid = o.hp[q];
                if (rank.find(pid) == rank.end()) { int n = (int)rank.size(); rank[pid] = n; }
            }
        }
    }
}

void reorder(Map& m, std::vector<Contact>& cs, V3 p, double r) {
    if (cs.size() < 2) {
        if (cs.size() == 1) {   // keep the rank computation's side effects out: a single contact needs no order
        }
        return;
    }
    std::vector<long long> order_oids;
    std::vector<std::vector<Contact>> groups;
    for (auto& c : cs) {
        long long oid = m.object_id[c.triangle];
        size_t g = 0;
        for (; g < order_oids.size(); g++) if (order_oids[g] == oid) break;
        if (g == order_oids.size()) { order_oids.push_back(oid); groups.emplace_back(); }
        groups[g].push_back(c);
    }
    std::vector<Contact> result;
    static thread_local std::unordered_map<int, int> rk;
    for (size_t g = 0; g < groups.size(); g++) {
        auto it = m.objects.find(order_oids[g]);
        if (it != m.objects.end() && groups[g].size() > 1) {
            ranks(it->second, p, r, rk);
            int nr = (int)rk.size();
            std::stable_sort(groups[g].begin(), groups[g].end(), [&](const Contact& a, const Contact& b) {
                auto fa = rk.find(a.polygon), fb = rk.find(b.polygon);
                long long ka = fa != rk.end() ? fa->second : (long long)nr + a.polygon;
                long long kb = fb != rk.end() ? fb->second : (long long)nr + b.polygon;
                return ka < kb;
            });
        }
        result.insert(result.end(), groups[g].begin(), groups[g].end());
    }
    cs.swap(result);
}

// ------------------------------------------------------------------ contacts (physics.py GeneralPhysics.contacts)
void contacts(Map& m, V3 p, double radius, std::vector<Contact>& out, std::vector<Event>* amb) {
    out.clear();
    static thread_local std::vector<int> ids, pids;
    double qlo[3] = {(p.x - radius) - .0001, (p.y - radius) - .0001, (p.z - radius) - .0001};
    double qhi[3] = {(p.x + radius) + .0001, (p.y + radius) + .0001, (p.z + radius) + .0001};
    query_aabb(m, qlo, qhi, ids);
    if (ids.empty()) return;
    pids.clear();
    for (int i : ids) pids.push_back(m.polygon_id[i]);
    std::sort(pids.begin(), pids.end()); pids.erase(std::unique(pids.begin(), pids.end()), pids.end());
    for (int pid : pids) {
        const Poly& poly = m.polys[pid]; V3 n = poly.normal;
        double distance = ((n.y * p.y + n.z * p.z) + n.x * p.x) + poly.d;
        if (std::fabs((double)(float)distance) > radius + .0001) continue;
        double projected = ((n.x * p.x + n.y * p.y) + n.z * p.z) + poly.d;
        V3 contact = sub(p, mul(n, projected)); V3 point = contact; bool rejected = false;
        float native_radius = (float)radius;
        float rr = native_radius * native_radius;
        double radicand = (double)rr - distance * distance;
        double separation = radicand >= 0. ? (double)(float)std::sqrt(radicand) : std::numeric_limits<double>::quiet_NaN();
        bool boundary = std::fabs(std::fabs(distance) - radius) <= 1e-12;
        for (int j = 0; j < poly.nv; j++) {
            if (!m.pvalid[poly.vstart + j]) continue;
            V3 edge = m.pedges[poly.vstart + j]; V3 vertex = m.pverts[poly.vstart + j];
            double vd = dot(edge, sub(contact, vertex));
            if (vd < 0) {
                if (vd < -(separation + .0001)) {
                    rejected = true;
                    if (amb == nullptr || !boundary) break;
                }
                V3 before = m.pverts[poly.vstart + (j - 1 + poly.nv) % poly.nv];
                V3 along = unit(cross3(edge, n));
                if (dot(along, sub(contact, vertex)) >= 0) {
                    if (dot(neg(along), sub(contact, before)) >= 0) { point = sub(contact, mul(edge, vd)); break; }
                    point = before;
                } else point = vertex;
            }
        }
        if (rejected && (amb == nullptr || !boundary)) continue;
        V3 delta = sub(p, point); double dist = norm(delta);
        if (dist > radius + .0001) continue;
        int idx = poly.first_triangle;
        V3 normal = dist != 0.0 ? dvs(delta, dist) : n;
        Material mat = m.materials[m.material_id[idx]];
        if (rejected) {
            amb->push_back(Event{1, 1, idx, normal, point, projected});
            continue;
        }
        out.push_back(Contact{normal, point, dist, idx, pid, mat, n});
    }
    reorder(m, out, p, radius);
}

// ------------------------------------------------------------------ swept sphere (experimental_sweep + sweep_kernel)
struct Hit { int triangle; double t; V3 center, point, normal; };

bool sweep_sphere(Map& m, V3 start, V3 end, double radius, double native_dt, Hit& hit) {
    V3 velocity = sub(end, start);
    double speed2 = dot(velocity, velocity);
    if (speed2 < 1e-18) return false;
    double qlo[3] = {std::min(start.x, end.x) - radius, std::min(start.y, end.y) - radius, std::min(start.z, end.z) - radius};
    double qhi[3] = {std::max(start.x, end.x) + radius, std::max(start.y, end.y) + radius, std::max(start.z, end.z) + radius};
    static thread_local std::vector<int> ids;
    query_aabb(m, qlo, qhi, ids);
    if (ids.empty()) return false;
    const bool single = ids.size() == 1;
    const double inf = std::numeric_limits<double>::infinity();
    const double approach_limit = -.001 * std::sqrt(speed2);
    double best_t_all = inf; int best_k = -1; V3 best_point{0, 0, 0};
    for (size_t k = 0; k < ids.size(); k++) {
        const int id = ids[k];
        V3 normal = load(&m.planes[id * 4]);
        const double approach = mv_row(normal, velocity, single);
        const double distance = mv_row(normal, start, single) + m.planes[id * 4 + 3];
        const bool valid = approach < approach_limit;
        double best = inf; V3 closest{0, 0, 0};
        if (valid) {
            const double t = (radius - distance) / approach;
            const V3 point = sub(add(start, mul(velocity, t)), mul(normal, radius));
            const V3 rel = sub(point, load(&m.tri[id * 9]));
            const double nnv = m.nn[id];
            const double u = nnv > 0 ? kdot(cross3(rel, load(&m.ac[id * 3])), load(&m.fcross[id * 3])) / nnv : 0;
            const double v = nnv > 0 ? kdot(cross3(load(&m.ab[id * 3]), rel), load(&m.fcross[id * 3])) / nnv : 0;
            if (nnv > 0 && t > 0 && t < 1 && u >= -1e-7 && v >= -1e-7 && u + v <= 1 + 1e-7) { best = t; closest = point; }
        }
        if (valid) for (int j = 0; j < 3; j++) {
            const V3 a = load(&m.source[id * 9 + j * 3]), b = load(&m.source[id * 9 + ((j + 1) % 3) * 3]);
            const V3 edge = sub(b, a), pv = sub(start, a);
            const double edge_dot = mv_row(edge, velocity, single);
            const double vertex_dot = mv_row(pv, velocity, single);
            const double length2 = kdot(edge, edge);
            const double ve = length2 > 0 ? edge_dot / length2 : 0;
            const double pe = length2 > 0 ? kdot(pv, edge) / length2 : 0;
            const V3 vr = sub(velocity, mul(edge, ve)), pr = sub(pv, mul(edge, pe));
            const double qa = kdot(vr, vr), qb = 2 * kdot(pr, vr), qc = kdot(pr, pr) - radius * radius;
            const double disc = qb * qb - 4 * qa * qc;
            double root = std::sqrt(positive(disc));
            if (native_dt > 0) {
                const double scaled = positive(disc) * length2 * length2 / (native_dt * native_dt);
                root = length2 > 0 ? f32root(scaled) * native_dt / length2 : 0;
            }
            const double t = qa > 1e-18 ? (-qb - root) / (2 * qa) : inf;
            const double exit_t = qa > 1e-18 ? (-qb + root) / (2 * qa) : -inf;
            const double along = pe + (std::isfinite(t) ? t : 0) * ve;
            if (m.boundary[id * 3 + j] && length2 > 0 && disc >= 0 && t > 0 && t < 1 && t < best && along >= 0 && along <= 1 &&
                (native_dt <= 0 || exit_t * native_dt > .0001)) {
                best = t; closest = add(a, mul(edge, along));
            }
            const double vqb = 2 * vertex_dot, vqc = kdot(pv, pv) - radius * radius;
            const double vdisc = vqb * vqb - 4 * speed2 * vqc;
            double vroot = std::sqrt(positive(vdisc));
            if (native_dt > 0) vroot = f32root(positive(vdisc) / (native_dt * native_dt)) * native_dt;
            const double vt = (-vqb - vroot) / (2 * speed2), vexit = (-vqb + vroot) / (2 * speed2);
            if (vdisc >= 0 && vt > 0 && vt < 1 && vt < best && (native_dt <= 0 || vexit * native_dt > .0001)) { best = vt; closest = a; }
        }
        if (best < best_t_all) { best_t_all = best; best_k = (int)k; best_point = closest; }   // np.argmin: first minimum
    }
    if (best_k < 0) return false;
    hit.triangle = ids[best_k]; hit.t = best_t_all;
    hit.center = add(start, mul(velocity, best_t_all));
    hit.point = best_point;
    V3 nrm = sub(hit.center, best_point);
    hit.normal = dvs(nrm, norm(nrm));
    return true;
}

// ------------------------------------------------------------------ state
struct Params {
    double max_roll_velocity, angular_acceleration, braking_acceleration, static_friction, kinetic_friction,
        bounce_kinetic_friction, gravity, max_dot_slide, bounce_restitution, air_acceleration, jump_impulse,
        min_bounce_velocity, mass; bool has_mass;
    long long effect_duration[6]; bool has_effect_duration;
};
struct State {
    V3 p, v, w; double radius, yaw, next_yaw, pitch; bool smooth_camera, has_next_yaw;
    V3 gravity; int held, mode; bool eff_active[6]; long long eff_remaining[6];
    long long eff_end[6]; bool has_eff_end; long long clock_ms;
    V3 best_normal; bool has_best_normal; Params prm; V3 camera_offset;
};

// packed layout (doubles), shared with the Python wrapper nav/native/marble_sim.py
enum {
    S_P = 0, S_V = 3, S_W = 6, S_RADIUS = 9, S_YAW = 10, S_NEXT_YAW = 11, S_PITCH = 12, S_SMOOTH = 13, S_GRAV = 14,
    S_HELD = 17, S_MODE = 18, S_EFF_ACTIVE = 19, S_EFF_REM = 25, S_EFF_END = 31, S_HAS_EFF_END = 37, S_CLOCK = 38,
    S_BEST = 39, S_HAS_BEST = 42, S_PRM = 43, /* 13 params */ S_HAS_MASS = 56, S_EFF_DUR = 57, S_HAS_EFF_DUR = 63,
    S_HAS_NEXT_YAW = 64, S_CAM = 65, S_SIZE = 68
};

void unpack(const double* d, State& s) {
    s.p = load(d + S_P); s.v = load(d + S_V); s.w = load(d + S_W); s.radius = d[S_RADIUS]; s.yaw = d[S_YAW];
    s.next_yaw = d[S_NEXT_YAW]; s.pitch = d[S_PITCH]; s.smooth_camera = d[S_SMOOTH] != 0; s.gravity = load(d + S_GRAV);
    s.held = (int)d[S_HELD]; s.mode = (int)d[S_MODE];
    for (int i = 0; i < 6; i++) { s.eff_active[i] = d[S_EFF_ACTIVE + i] != 0; s.eff_remaining[i] = (long long)d[S_EFF_REM + i]; s.eff_end[i] = (long long)d[S_EFF_END + i]; }
    s.has_eff_end = d[S_HAS_EFF_END] != 0; s.clock_ms = (long long)d[S_CLOCK];
    s.best_normal = load(d + S_BEST); s.has_best_normal = d[S_HAS_BEST] != 0;
    const double* q = d + S_PRM; Params& P = s.prm;
    P.max_roll_velocity = q[0]; P.angular_acceleration = q[1]; P.braking_acceleration = q[2]; P.static_friction = q[3];
    P.kinetic_friction = q[4]; P.bounce_kinetic_friction = q[5]; P.gravity = q[6]; P.max_dot_slide = q[7];
    P.bounce_restitution = q[8]; P.air_acceleration = q[9]; P.jump_impulse = q[10]; P.min_bounce_velocity = q[11]; P.mass = q[12];
    P.has_mass = d[S_HAS_MASS] != 0;
    for (int i = 0; i < 6; i++) P.effect_duration[i] = (long long)d[S_EFF_DUR + i];
    P.has_effect_duration = d[S_HAS_EFF_DUR] != 0; s.has_next_yaw = d[S_HAS_NEXT_YAW] != 0; s.camera_offset = load(d + S_CAM);
}

void pack(const State& s, double* d) {
    store(d + S_P, s.p); store(d + S_V, s.v); store(d + S_W, s.w); d[S_RADIUS] = s.radius; d[S_YAW] = s.yaw;
    d[S_NEXT_YAW] = s.next_yaw; d[S_PITCH] = s.pitch; d[S_SMOOTH] = s.smooth_camera; store(d + S_GRAV, s.gravity);
    d[S_HELD] = s.held; d[S_MODE] = s.mode;
    for (int i = 0; i < 6; i++) { d[S_EFF_ACTIVE + i] = s.eff_active[i]; d[S_EFF_REM + i] = (double)s.eff_remaining[i]; d[S_EFF_END + i] = (double)s.eff_end[i]; }
    d[S_HAS_EFF_END] = s.has_eff_end; d[S_CLOCK] = (double)s.clock_ms; store(d + S_BEST, s.best_normal); d[S_HAS_BEST] = s.has_best_normal;
    double* q = d + S_PRM; const Params& P = s.prm;
    q[0] = P.max_roll_velocity; q[1] = P.angular_acceleration; q[2] = P.braking_acceleration; q[3] = P.static_friction;
    q[4] = P.kinetic_friction; q[5] = P.bounce_kinetic_friction; q[6] = P.gravity; q[7] = P.max_dot_slide;
    q[8] = P.bounce_restitution; q[9] = P.air_acceleration; q[10] = P.jump_impulse; q[11] = P.min_bounce_velocity; q[12] = P.mass;
    d[S_HAS_MASS] = P.has_mass;
    for (int i = 0; i < 6; i++) d[S_EFF_DUR + i] = (double)P.effect_duration[i];
    d[S_HAS_EFF_DUR] = P.has_effect_duration; d[S_HAS_NEXT_YAW] = s.has_next_yaw; store(d + S_CAM, s.camera_offset);
}

struct Move { double x, y, yaw; bool jump, use, trigger1; };

enum { E_OK = 0, E_MODE = 1, E_GRAVITY = 2, E_SS_NORMAL = 3, E_SCRIPT_POWER = 4, E_RANDOM_FORCE = 5, E_DIVERGED = 6 };

// ------------------------------------------------------------------ cancel (GeneralPhysics.cancel)
void cancel(V3& v, V3& w, const std::vector<Contact>& cs, bool centered, bool no_bounce, const State& s, double r) {
    bool slide = centered;
    const Params& P = s.prm;
    for (int iteration = 0; iteration < 7; iteration++) {
        bool changed = false;
        for (const Contact& c : cs) {
            V3 n = c.normal;
            double dt = (v.z * n.z + v.y * n.y) + v.x * n.x;
            if (!((iteration == 0 && dt < 0) || dt < -.0001)) continue;
            double speed = norm(v); V3 normal_v = mul(n, dt);
            if (no_bounce) v = sub(v, normal_v);
            else if (!slide && dt > -P.max_dot_slide * speed) {
                v = sub(v, normal_v); v = mul(unit(v), speed); slide = true;
            } else if (dt > -P.min_bounce_velocity) v = sub(v, normal_v);
            else {
                double base = s.eff_active[4] ? .01 : s.eff_active[3] ? .9 : P.bounce_restitution;
                double restitution = c.mat.restitution * base;
                V3 tangent = sub(add(cross3(w, mul(neg(n), r)), v), mul(n, dt));
                double mag = norm(tangent);
                if (mag > 1e-12) {
                    double friction = f32(P.bounce_kinetic_friction * c.mat.friction);
                    double delta_mag = pymin(((friction * 2.5) * (-dt)) / r, mag / r);
                    V3 dw = mul(cross3(n, dvs(tangent, mag)), delta_mag);
                    w = add(w, dw); v = sub(v, cross3(dw, mul(n, r)));
                }
                v = sub(v, mul(normal_v, 1 + restitution));
            }
            changed = true;
        }
        if (!changed) break;
    }
}

// ------------------------------------------------------------------ step (GeneralPhysics.step)
int step(Map& m, State& s, const Move& move, int dt_ms, std::vector<Event>& events) {
    Params& P = s.prm;
    P.max_roll_velocity = f32(P.max_roll_velocity); P.angular_acceleration = f32(P.angular_acceleration);
    P.braking_acceleration = f32(P.braking_acceleration); P.static_friction = f32(P.static_friction);
    P.kinetic_friction = f32(P.kinetic_friction); P.bounce_kinetic_friction = f32(P.bounce_kinetic_friction);
    P.gravity = f32(P.gravity); P.max_dot_slide = f32(P.max_dot_slide); P.bounce_restitution = f32(P.bounce_restitution);
    P.air_acceleration = f32(P.air_acceleration); P.jump_impulse = f32(P.jump_impulse);
    P.min_bounce_velocity = f32(P.min_bounce_velocity); if (P.has_mass) P.mass = f32(P.mass);
    const double r = f32(s.radius); s.radius = r;
    V3 p = s.p, v = s.v, w = s.w;
    if (s.mode != 0) return E_MODE;
    // axes()
    V3 gravity = s.gravity;
    if (norm(sub(gravity, V3{0, 0, -1})) > 1e-6) return E_GRAVITY;
    double yaw = s.yaw;
    const double twopi = 2 * 3.141592653589793;
    if (s.smooth_camera) {
        double next_yaw = (s.has_next_yaw ? s.next_yaw : yaw) + move.yaw;
        double diff = pymod(next_yaw - yaw + 3.141592653589793, twopi) - 3.141592653589793;
        yaw += diff * (1 - std::pow(.5, dt_ms / 16.0));
        s.next_yaw = next_yaw; s.has_next_yaw = true;
    } else yaw += move.yaw;
    s.yaw = pymod(yaw, twopi);
    const V3 side{std::cos(yaw), -std::sin(yaw), 0.}, forward{std::sin(yaw), std::cos(yaw), 0.};
    {   // camera_offset = forward*cos(pitch) - [0,0,1]*sin(pitch), elementwise as NumPy does it
        double c = std::cos(s.pitch), sn = std::sin(s.pitch);
        s.camera_offset = V3{forward.x * c - 0.0 * sn, forward.y * c - 0.0 * sn, forward.z * c - 1.0 * sn};
    }
    const V3 direction = add(mul(side, move.x), mul(forward, move.y));
    const long long clock = s.clock_ms;
    long long deadlines[6];
    for (int i = 0; i < 6; i++) deadlines[i] = s.has_eff_end ? s.eff_end[i] : clock + s.eff_remaining[i];
    const int held = s.held;
    if (move.use && held) {
        if (held == 1) v = sub(v, mul(gravity, 20.));
        else if (held == 2) {
            if (!s.has_best_normal) return E_SS_NORMAL;
            V3 previous = s.best_normal;
            float aim[3] = {(float)forward.x, (float)forward.y, (float)forward.z};
            float pf[3] = {(float)previous.x, (float)previous.y, (float)previous.z};
            double acc = 0.0; for (int i = 0; i < 3; i++) acc += (double)(pf[i] * aim[i]);
            double projection = (double)(float)acc;
            float boost[3] = {(float)((double)aim[0] - previous.x * projection), (float)((double)aim[1] - previous.y * projection),
                              (float)((double)aim[2] - previous.z * projection)};
            double acc2 = 0.0; for (int i = 0; i < 3; i++) acc2 += (double)(boost[i] * boost[i]);
            float length = std::sqrt((float)acc2);
            if (length != 0.0f) { for (int i = 0; i < 3; i++) boost[i] = boost[i] / length; }
            else { boost[0] = 1.f; boost[1] = 0.f; boost[2] = 0.f; }
            v = add(v, V3{(double)(boost[0] * 25.f), (double)(boost[1] * 25.f), (double)(boost[2] * 25.f)});
        } else if (held >= 3 && held <= 5) {
            // effects[held] set below with the duration
        } else return E_SCRIPT_POWER;
        s.held = 0;
        static const long long def[6] = {5000, 1000, 1000, 5000, 5000, 5000};
        long long duration = P.has_effect_duration ? P.effect_duration[held] : def[held];
        s.eff_active[held] = true; s.eff_remaining[held] = duration; deadlines[held] = clock + duration;
    }
    double total = dt_ms / 1000.; int iterations = 0;
    static thread_local std::vector<Contact> cs;
    std::vector<double> nf;
    while (total > 1e-10) {
        double chunk = pymin(.008, total), left = chunk;
        while (left > 1e-10) {
            contacts(m, p, r, cs, &events);
            V3 control{0, 0, 0}, desired{0, 0, 0};
            const bool centered = move.x == 0 && move.y == 0;
            const V3 radius_up = mul(neg(gravity), r);
            if (!centered) {
                V3 rolling = cross3(w, radius_up);
                double current[2] = {dot(rolling, side), dot(rolling, forward)};
                double target[2] = {move.x * P.max_roll_velocity, move.y * P.max_roll_velocity};
                for (int j = 0; j < 2; j++)
                    if ((target[j] > 0 && current[j] > target[j]) || (target[j] < 0 && current[j] < target[j])) target[j] = current[j];
                desired = dvs(cross3(radius_up, add(mul(side, target[0]), mul(forward, target[1]))), r * r);
                control = sub(desired, w);
                control = mul(control, pymin(1., P.angular_acceleration / pymax(norm(control), 1e-12)));
            }
            if (!centered && cs.size() > 1) {   // grazing_contacts
                std::vector<double> dots;
                for (auto& c : cs) dots.push_back((v.z * c.normal.z + v.y * c.normal.y) + v.x * c.normal.x);
                bool any_fragile = false;
                for (size_t i = 0; i + 1 < dots.size(); i++) {
                    double d0 = dots[i];
                    double mn = dots[i + 1]; for (size_t k = i + 2; k < dots.size(); k++) mn = pymin(mn, dots[k]);
                    if (std::fabs(d0) >= 1e-6 || mn >= -P.min_bounce_velocity) continue;
                    const V3 nn3 = cs[i].normal; double comp[3] = {nn3.x, nn3.y, nn3.z}; double vel[3] = {v.x, v.y, v.z};
                    int nz = 0, ax = -1; for (int k = 0; k < 3; k++) if (comp[k] != 0) { nz++; if (ax < 0) ax = k; }
                    bool exact = d0 == 0 && nz == 1 && std::fabs(comp[ax]) == 1 && vel[ax] == 0;
                    if (exact) {
                        double pl[3] = {cs[i].plane.x, cs[i].plane.y, cs[i].plane.z};
                        int pnz = 0, pax = -1; for (int k = 0; k < 3; k++) if (pl[k] != 0) { pnz++; if (pax < 0) pax = k; }
                        auto sgn = [](double x) { return (x > 0) - (x < 0); };
                        exact = pnz == 1 && pax == ax && sgn(pl[ax]) == sgn(comp[ax]);
                    }
                    if (!exact) any_fragile = true;
                }
                if (any_fragile) events.push_back(Event{1, 2, -1, p, V3{0, 0, 0}, 0});
            }
            cancel(v, w, cs, centered, false, s, r);
            V3 A = mul(mul(gravity, P.gravity), s.eff_active[5] ? .25 : 1.);
            if (cs.empty()) A = add(A, mul(mul(direction, P.air_acceleration), s.eff_active[5] ? 2. : 1.));
            {   // force contacts
                V3 sum{0, 0, 0}; int nfc = 0; double last_force = 0;
                for (auto& c : cs) if (c.mat.force != 0) { sum = add(sum, c.normal); nfc++; last_force = c.mat.force; }
                if (nfc) {
                    V3 n = unit(sum);
                    double impulse = last_force / (P.has_mass ? P.mass : 1.);
                    double dt = dot(n, v);
                    if (impulse > dt) A = add(A, dvs(mul(n, impulse - pymax(0., dt)), left));
                }
            }
            nf.clear();
            for (auto& c : cs) nf.push_back(-dot(c.normal, A));
            int best = -1;
            if (!nf.empty()) {
                int am = 0; for (int i = 1; i < (int)nf.size(); i++) if (nf[i] > nf[am]) am = i;
                double mx = nf[0]; for (int i = 1; i < (int)nf.size(); i++) mx = pymax(mx, nf[i]);
                if (mx > 0) best = am;
            }
            V3 angular{0, 0, 0};
            if (best >= 0) {
                const Contact& c = cs[best]; V3 n = c.normal; double normal_force = nf[best];
                double tolerance = 128 * std::numeric_limits<double>::epsilon() * pymax(1., std::fabs(normal_force));
                bool tie = false;
                for (int i = 0; i < (int)nf.size(); i++)
                    if (i != best && nf[i] > 0 && std::fabs(nf[i] - normal_force) <= tolerance && norm(sub(cs[i].normal, n)) > 1e-7) tie = true;
                if (tie) events.push_back(Event{1, 3, c.triangle, p, n, normal_force});
                s.best_normal = n; s.has_best_normal = true;
                events.push_back(Event{2, 0, c.triangle, n, V3{0, 0, 0}, 0});
                if (move.jump) {
                    double impulse = P.jump_impulse - pymax(0., f32(dot(v, n)));
                    if (impulse > 0) { v = add(v, mul(n, impulse)); events.push_back(Event{3, 0, c.triangle, n, V3{0, 0, 0}, impulse}); }
                }
            }
            for (auto& c : cs) {
                double force = -dot(c.normal, A);
                if (force > 0 && dot(v, c.normal) <= .0001) A = add(A, mul(c.normal, force));
            }
            if (best >= 0) {
                const Contact& c = cs[best]; V3 n = c.normal; double normal_force = nf[best];
                V3 at_contact = add(cross3(w, mul(neg(n), r)), v);
                double mag = norm(at_contact);
                bool slipping = false; V3 angular_friction{0, 0, 0}, linear_friction{0, 0, 0};
                if (mag > 1e-12) {
                    double mu = f32(P.kinetic_friction * c.mat.friction);
                    if (mu < 0) return E_RANDOM_FORCE;
                    double alpha = ((mu * 2.5) * normal_force) / r, accel = normal_force * mu;
                    double delta = (r * alpha + accel) * left;
                    slipping = mag >= delta;
                    if (mag < delta && delta > 0) { alpha *= mag / delta; accel *= mag / delta; }
                    V3 contact_dir = dvs(at_contact, mag);
                    angular_friction = mul(cross3(n, contact_dir), alpha);
                    linear_friction = mul(contact_dir, -accel);
                }
                if (!slipping) {
                    V3 addv = dvs(cross3(radius_up, A), r * r);
                    if (centered) {
                        control = sub(desired, add(mul(angular, left), w));
                        control = mul(control, pymin(1., P.braking_acceleration / pymax(norm(control), 1e-12)));
                    }
                    V3 linear_add = neg(cross3(control, mul(neg(n), r)));
                    double at_mag = norm(add(linear_add, cross3(addv, mul(neg(n), r))));
                    double stat = f32(P.static_friction * c.mat.friction);
                    double kin = f32(P.kinetic_friction * c.mat.friction);
                    if (stat * normal_force < f32(at_mag)) linear_add = mul(linear_add, (kin * normal_force) / pymax(f32(at_mag), 1e-12));
                    A = add(A, linear_add); angular = add(angular, addv);
                }
                A = add(A, linear_friction); angular = add(angular, angular_friction);
            }
            angular = add(angular, control);
            v = add(v, mul(A, left)); w = add(w, mul(angular, left));
            cancel(v, w, cs, centered, true, s, r);
            double native_speed = f32(std::sqrt((v.z * v.z + v.y * v.y) + v.x * v.x));
            bool stationary = native_speed < .001;
            Hit hit; bool hitting = !stationary && sweep_sphere(m, p, add(p, mul(v, left)), r, left, hit);
            if (hitting) {
                double elapsed = left * hit.t; p = hit.center;
                v = sub(v, mul(A, left - elapsed)); w = sub(w, mul(angular, left - elapsed));
                events.push_back(Event{4, 0, hit.triangle, hit.normal, hit.point, hit.t});
                left -= elapsed;
                if (elapsed < 1e-9) p = add(p, mul(hit.normal, 1e-8));
            } else {
                if (!stationary) p = add(p, mul(v, left));
                left = 0.;
            }
            if (!(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) || norm(v) > 1000) return E_DIVERGED;
            iterations++;
            if (iterations > 10) {
                if (left > 1e-10) events.push_back(Event{5, 0, -1, p, V3{0, 0, 0}, left});
                break;
            }
        }
        total -= chunk;
    }
    if (move.trigger1) { v = mul(v, .8); w = mul(w, .8); }
    for (int i = 0; i < 6; i++) {
        if (s.eff_active[i]) {
            long long rem = deadlines[i] - clock; s.eff_remaining[i] = rem > 0 ? rem : 0;
            if (clock > deadlines[i]) s.eff_active[i] = false;
        }
    }
    s.clock_ms = clock; for (int i = 0; i < 6; i++) s.eff_end[i] = deadlines[i]; s.has_eff_end = true;
    s.p = p; s.v = v; s.w = w;
    return E_OK;
}

#include "controllers.inc"

}  // namespace

// ------------------------------------------------------------------ C API
DLL int ms_abi() { return 13; }
DLL int ms_state_size() { return S_SIZE; }

DLL void* ms_create(int ntri, const double* tri, const double* source, const double* planes, const uint8_t* boundary,
                    const double* ab, const double* ac, const double* fcross, const double* nn, const double* lo, const double* hi,
                    const int* material_id, const long long* object_id, const int* polygon_id, int nmat, const double* mat_props) {
    Map* m = new Map();
    m->ntri = ntri;
    m->tri.assign(tri, tri + (size_t)ntri * 9); m->source.assign(source, source + (size_t)ntri * 9);
    m->planes.assign(planes, planes + (size_t)ntri * 4); m->boundary.assign(boundary, boundary + (size_t)ntri * 3);
    m->ab.assign(ab, ab + (size_t)ntri * 3); m->ac.assign(ac, ac + (size_t)ntri * 3); m->fcross.assign(fcross, fcross + (size_t)ntri * 3);
    m->nn.assign(nn, nn + ntri); m->lo.assign(lo, lo + (size_t)ntri * 3); m->hi.assign(hi, hi + (size_t)ntri * 3);
    m->material_id.assign(material_id, material_id + ntri); m->object_id.assign(object_id, object_id + ntri);
    m->polygon_id.assign(polygon_id, polygon_id + ntri);
    for (int i = 0; i < nmat; i++) m->materials.push_back(Material{mat_props[i * 3], mat_props[i * 3 + 1], mat_props[i * 3 + 2]});
    build_grid(*m, 2.0);
    return m;
}

DLL void ms_set_polygons(void* h, int npoly, const int* vcount, const double* vertices, const double* normals, const double* ds,
                         const double* edges, const uint8_t* edge_valid, const int* first_triangle) {
    Map* m = (Map*)h; m->polys.clear(); m->pverts.clear(); m->pedges.clear(); m->pvalid.clear();
    int off = 0;
    for (int i = 0; i < npoly; i++) {
        Poly p; p.nv = vcount[i]; p.vstart = off; p.first_triangle = first_triangle[i];
        p.object = (int)m->object_id[first_triangle[i]]; p.normal = load(normals + i * 3); p.d = ds[i];
        for (int j = 0; j < p.nv; j++) {
            m->pverts.push_back(load(vertices + (size_t)(off + j) * 3)); m->pedges.push_back(load(edges + (size_t)(off + j) * 3));
            m->pvalid.push_back(edge_valid[off + j]);
        }
        off += p.nv; m->polys.push_back(p);
    }
}

DLL void ms_add_object(void* h, long long oid, const float* inverse16, const float* scale3, int nhull, const float* hull_lo,
                       const float* hull_hi, const float* bbox_lo3, const float* bbox_hi3, const int* bin_start, const int* bin_count,
                       int nidx, const int* indices, const int* hp_off, const int* hp) {
    Map* m = (Map*)h; HullObject o;
    std::memcpy(o.inv, inverse16, sizeof(o.inv)); std::memcpy(o.scale, scale3, sizeof(o.scale));
    std::memcpy(o.bbox_lo, bbox_lo3, sizeof(o.bbox_lo)); std::memcpy(o.bbox_hi, bbox_hi3, sizeof(o.bbox_hi));
    o.nhull = nhull; o.hull_lo.assign(hull_lo, hull_lo + nhull * 3); o.hull_hi.assign(hull_hi, hull_hi + nhull * 3);
    std::memcpy(o.bin_start, bin_start, sizeof(o.bin_start)); std::memcpy(o.bin_count, bin_count, sizeof(o.bin_count));
    o.indices.assign(indices, indices + nidx); o.hp_off.assign(hp_off, hp_off + nhull + 1); o.hp.assign(hp, hp + hp_off[nhull]);
    m->objects[oid] = std::move(o);
}

DLL void ms_destroy(void* h) { delete (Map*)h; }

// One physics step. move = [x, y, yaw, jump, use, trigger1]. events_out rows of 11 doubles:
// [kind, sub, triangle, a.x, a.y, a.z, b.x, b.y, b.z, value, 0]. Returns an error code (0 = ok).
DLL int ms_step(void* h, const double* state_in, const double* move_in, int dt_ms, double* state_out,
                double* events_out, int max_events, int* n_events) {
    Map* m = (Map*)h; State s; unpack(state_in, s);
    Move mv{move_in[0], move_in[1], move_in[2], move_in[3] != 0, move_in[4] != 0, move_in[5] != 0};
    std::vector<Event> ev;
    int code = step(*m, s, mv, dt_ms, ev);
    pack(s, state_out);
    int n = 0;
    for (auto& e : ev) {
        if (n >= max_events) break;
        double* r = events_out + n * 11;
        r[0] = e.kind; r[1] = e.sub; r[2] = e.triangle; r[3] = e.a.x; r[4] = e.a.y; r[5] = e.a.z; r[6] = e.b.x; r[7] = e.b.y; r[8] = e.b.z;
        r[9] = e.value; r[10] = 0; n++;
    }
    *n_events = (int)ev.size();
    return code;
}

// Contacts at a position (validation): rows of [triangle, polygon, normal xyz, point xyz, distance]; returns the count.
DLL int ms_contacts(void* h, const double* p, double radius, double* out, int max_out) {
    Map* m = (Map*)h; std::vector<Contact> cs; std::vector<Event> amb;
    contacts(*m, load(p), radius, cs, &amb);
    int n = 0;
    for (auto& c : cs) {
        if (n >= max_out) break;
        double* r = out + n * 9;
        r[0] = c.triangle; r[1] = c.polygon; r[2] = c.normal.x; r[3] = c.normal.y; r[4] = c.normal.z;
        r[5] = c.point.x; r[6] = c.point.y; r[7] = c.point.z; r[8] = c.distance; n++;
    }
    return (int)cs.size();
}

// One 64 ms reply of the 16 ms bridge (bridge_small_step.predict_small_step): subframe 0 runs move0 after the reply's
// camera yaw (yaw = next_yaw = f32(value)) and an optional Blast velocity change; subframes 1-3 run move1. The clock of
// subframe k is clock0 + 16 k. rows_out receives the packed state after each completed subframe, amb_out[k] = 1 when
// subframe k produced an ambiguous-contact event. With stop_on_ambiguity the reply stops after such a subframe.
// Returns the number of completed subframes, plus (error code << 8) when a step failed.
DLL int ms_reply(void* h, const double* state_in, const double* move0, const double* move1, long long clock0,
                 int has_camera, double camera_yaw, int has_blast, const double* blast_dv, int stop_on_ambiguity,
                 double* rows_out, int* amb_out) {
    Map* m = (Map*)h; State s; unpack(state_in, s);
    std::vector<Event> ev;
    for (int k = 0; k < 4; k++) {
        s.clock_ms = clock0 + 16LL * k;
        const double* mv = k == 0 ? move0 : move1;
        if (k == 0) {
            if (has_blast) s.v = add(s.v, load(blast_dv));
            if (has_camera) { double y = f32(camera_yaw); s.yaw = y; s.next_yaw = y; s.has_next_yaw = true; }
        }
        Move move{mv[0], mv[1], mv[2], mv[3] != 0, mv[4] != 0, mv[5] != 0};
        ev.clear();
        int code = step(*m, s, move, 16, ev);
        if (code) return k | (code << 8);
        pack(s, rows_out + (size_t)k * S_SIZE);
        bool amb = false; for (auto& e : ev) if (e.kind == 1) { amb = true; break; }
        amb_out[k] = amb;
        if (stop_on_ambiguity && amb) return k + 1;
    }
    return 4;
}

// ------------------------------------------------------------------ walk grid / controllers / whole forecasts
DLL void* ms_terrain_create(void* map, float x0, float y0, double res, int KW, int H, int W, const float* walk_z, const float* wxs,
                            const float* wys, int has_graph, long long nnodes, const long long* indptr, long long nnz, const long long* indices,
                            const double* gdata, const long long* node_idx, const long long* nodes, const uint8_t* obstacle,
                            double clear_radius, double drop_edge, double step_up, double level_merge) {
    Terrain* t = new Terrain(); Map* m = (Map*)map; t->map = m;
    t->x0 = x0; t->y0 = y0; t->res = res; t->KW = KW; t->H = H; t->W = W;
    t->walk_z.assign(walk_z, walk_z + (size_t)KW * H * W); t->wxs.assign(wxs, wxs + W); t->wys.assign(wys, wys + H);
    t->has_graph = has_graph != 0;
    if (has_graph) {
        t->indptr.assign(indptr, indptr + nnodes + 1); t->indices.assign(indices, indices + nnz); t->gdata.assign(gdata, gdata + nnz);
        t->nodes.assign(nodes, nodes + nnodes * 3);
    }
    t->node_idx.assign(node_idx, node_idx + (size_t)KW * H * W);
    t->obstacle.assign(obstacle, obstacle + m->ntri); t->clear_radius = clear_radius;
    t->drop_edge = drop_edge; t->step_up = step_up; t->level_merge = level_merge;
    return t;
}
DLL void ms_terrain_destroy(void* t) { delete (Terrain*)t; }
DLL void ms_terrain_set_edge(void* th, const float* edge) {   // ABI 7: terrain.edge_dist for search.open_top
    Terrain* t = (Terrain*)th; t->edge.assign(edge, edge + (size_t)t->KW * t->H * t->W);
}
DLL void ms_terrain_set_route_tol(void* th, double tol, double ntol, double radius) {   // ABI 11: terrain.ROUTE_LEVEL_TOL_U, ROUTE_NEIGHBOR_TOL_U, ROUTE_RADIUS_U
    Terrain* t = (Terrain*)th; t->route_tol = tol; t->route_ntol = ntol; t->route_radius = radius;
}
DLL void ms_terrain_seed_point(void* th, const double* p, int clear) {
    Terrain* t = (Terrain*)th; V3 c = load(p); t->point_cache[std::string((const char*)&c, sizeof(V3))] = clear != 0;
}
DLL void ms_terrain_seed_segment(void* th, const double* key7, int clear) {
    Terrain* t = (Terrain*)th; t->segment_cache[std::string((const char*)key7, 7 * sizeof(double))] = clear != 0;
}

// helper checks (tools/native_controller_check.py)
DLL void ms_cell_of(void* th, double x, double y, int strong, int* ji) { int j, i; cell_of(*(Terrain*)th, x, y, strong != 0, strong != 0, j, i); ji[0] = j; ji[1] = i; }
DLL int ms_level_at(void* th, int j, int i, int has_z, double z) { return level_at(*(Terrain*)th, j, i, has_z != 0, z); }
DLL int ms_line_clear(void* th, double ax, double ay, double bx, double by, int has_az, double az, int has_bz, double bz, int sx, int sy) {
    return line_clear(*(Terrain*)th, ax, ay, bx, by, has_az != 0, az, has_bz != 0, bz, sx != 0, sy != 0);
}
DLL int ms_path_waypoint(void* th, const float* field, double mx, double my, const double* gem, int has_mz, double mz, double* out) {
    P3 o; bool r = path_waypoint(*(Terrain*)th, field, mx, my, P3{gem[0], gem[1], gem[2]}, has_mz != 0, mz, o);
    out[0] = o.x; out[1] = o.y; out[2] = o.z; return r;
}
DLL double ms_dist_at(void* th, const float* field, double x, double y, double gx, double gy, int has_z, double z, int strong) {
    return dist_at(*(Terrain*)th, field, x, y, gx, gy, has_z != 0, z, strong != 0);
}
DLL int ms_segment_clear(void* th, const double* a, const double* b) { return segment_clear(*(Terrain*)th, load(a), load(b)); }
DLL void ms_translated(const double* pending, int* hold, double* pow_yaw, double* move_out, double* camera) {
    Pending p; std::memcpy(p.a, pending, sizeof p.a); translated(p, *hold, *pow_yaw, move_out, *camera);
}
DLL double ms_py_round(double x, int nd) { return py_round(x, nd); }
DLL double ms_g6(double x) { return g6(x); }
DLL double ms_hypot(double a, double b) { return py_hypot(a, b); }

// seq spec layout (doubles): [route key x,y,z, has_wp, wp x,y,z, gem x,y,z, offset, scale, jump, use, turn_after, brake_after,
//   blast, direct, through, next key x,y,z, next has_wp, next wp x,y,z, ngem x,y,z, leg, has_arrive, arrive,
//   passed, has_prev, prev x,y,z] (37 values); written back after the forecast with the route / through state.
enum { Q_KEY = 0, Q_HAS_WP = 3, Q_WP = 4, Q_GEM = 7, Q_OFFSET = 10, Q_SCALE = 11, Q_JUMP = 12, Q_USE = 13, Q_TURN = 14, Q_BRAKE = 15,
       Q_BLAST = 16, Q_DIRECT = 17, Q_THROUGH = 18, Q_NKEY = 19, Q_NHAS_WP = 22, Q_NWP = 23, Q_NGEM = 26, Q_LEG = 29,
       Q_HAS_ARRIVE = 30, Q_ARRIVE = 31, Q_PASSED = 32, Q_HAS_PREV = 33, Q_PREV = 34,
       // ABI 5 (6 adds tail kind 3, landing.LandingContinuation): settle.SettleProposal / recharge.RechargeProposal (brake toward the gem for Q_BRAKE_PREFIX replies, then the
       // sequence at index - prefix) and phased_search.AimProposal (a use's aim minus Q_AIM)
       Q_BRAKE_PREFIX = 37, Q_HAS_AIM = 38, Q_AIM = 39, Q_HAS_MIN = 40, Q_MIN = 41,   // ABI 8: per-plan pickup minimum
       Q_HAS_BOX = 42, Q_BOX = 43, Q_FAST = 49, Q_FAST_MIN = 50,   // ABI 10: fast-family speed and floor
       Q_DIRECT_FROM = 51,   // ABI 12: direct_from + 1 (0 = never)
       Q_LEAD_K = 52, Q_LEAD_MAX = 53,   // ABI 13: SequenceProposal.lead_k, search.LEAD_AIM_MAX
       Q_LAND = 54, Q_LAND_G = 55, Q_LAND_AIR = 56, Q_LAND_DZ = 57, Q_LAND_MIN_VZ = 58, Q_LAND_MIN_T = 59, Q_LAND_AFTER = 60,
       Q_SIZE = 61 };   // ABI 13: SequenceProposal.land, search.LAND_*
enum { R_ACTION = 0, R_MOVE0 = 9, R_MOVE1 = 15, R_CAMERA = 21, R_IMPULSE = 22, R_PUSH = 23, R_DV = 26, R_METER = 29, R_SPECIAL = 30,
       R_DONE = 31, R_AMB = 32, R_BLAST_VALUE = 36, R_BLAST_SPECIAL = 37, R_SIZE = 38 };

void load_seq(SeqSpec& S, const double* seq, const float* field, const float* next_field) {
    S.route.field = field; S.route.key = P3{seq[Q_KEY], seq[Q_KEY + 1], seq[Q_KEY + 2]}; S.route.has_wp = seq[Q_HAS_WP] != 0;
    S.route.wp = P3{seq[Q_WP], seq[Q_WP + 1], seq[Q_WP + 2]}; S.gem = P3{seq[Q_GEM], seq[Q_GEM + 1], seq[Q_GEM + 2]};
    S.offset = seq[Q_OFFSET]; S.scale = seq[Q_SCALE]; S.jump = (int)seq[Q_JUMP]; S.use = (int)seq[Q_USE]; S.turn_after = (int)seq[Q_TURN];
    S.brake_after = (int)seq[Q_BRAKE]; S.blast = (int)seq[Q_BLAST]; S.direct = seq[Q_DIRECT] != 0; S.through = seq[Q_THROUGH] != 0;
    S.next_route.field = next_field; S.next_route.key = P3{seq[Q_NKEY], seq[Q_NKEY + 1], seq[Q_NKEY + 2]};
    S.next_route.has_wp = seq[Q_NHAS_WP] != 0; S.next_route.wp = P3{seq[Q_NWP], seq[Q_NWP + 1], seq[Q_NWP + 2]};
    S.ngem = P3{seq[Q_NGEM], seq[Q_NGEM + 1], seq[Q_NGEM + 2]}; S.leg = seq[Q_LEG]; S.has_arrive = seq[Q_HAS_ARRIVE] != 0;
    S.arrive = seq[Q_ARRIVE]; S.passed = seq[Q_PASSED] != 0; S.has_prev = seq[Q_HAS_PREV] != 0;
    S.prev = V3{seq[Q_PREV], seq[Q_PREV + 1], seq[Q_PREV + 2]};
    S.has_min = seq[Q_HAS_MIN] != 0; S.min_speed = seq[Q_MIN];
    S.has_box = seq[Q_HAS_BOX] != 0; for (int k = 0; k < 6; k++) S.box[k] = seq[Q_BOX + k];
    S.fast = seq[Q_FAST]; S.fast_min_speed = seq[Q_FAST_MIN];
    S.direct_from = (int)seq[Q_DIRECT_FROM] - 1;
    S.lead_k = seq[Q_LEAD_K]; S.lead_max = seq[Q_LEAD_MAX];
    S.land = seq[Q_LAND] != 0; S.land_g = seq[Q_LAND_G]; S.land_air = seq[Q_LAND_AIR]; S.land_dz = seq[Q_LAND_DZ];
    S.land_min_vz = seq[Q_LAND_MIN_VZ]; S.land_min_t = seq[Q_LAND_MIN_T]; S.land_after = (int)seq[Q_LAND_AFTER];
}
void save_seq(const SeqSpec& S, double* seq) {
    seq[Q_HAS_WP] = S.route.has_wp; seq[Q_WP] = S.route.wp.x; seq[Q_WP + 1] = S.route.wp.y; seq[Q_WP + 2] = S.route.wp.z;
    seq[Q_NHAS_WP] = S.next_route.has_wp; seq[Q_NWP] = S.next_route.wp.x; seq[Q_NWP + 1] = S.next_route.wp.y; seq[Q_NWP + 2] = S.next_route.wp.z;
    seq[Q_PASSED] = S.passed; seq[Q_HAS_PREV] = S.has_prev; seq[Q_PREV] = S.prev.x; seq[Q_PREV + 1] = S.prev.y; seq[Q_PREV + 2] = S.prev.z;
}

DLL void ms_seq_action(void* th, double* seq, const float* field, const float* next_field, const double* consts, const double* pv, int index, double* action_out) {
    Terrain* t = (Terrain*)th; SeqSpec S{};
    load_seq(S, seq, field, next_field);
    Consts C{consts[0], consts[1], consts[2], consts[3], consts[4], consts[5], consts[6] != 0, consts[7], consts[8], consts[9], consts[10],
             consts[11], consts[12]};   // ABI 7: 13 constants
    Action a = seq_call(*t, C, S, load(pv), load(pv + 3), index, pv[6]);   // pv: position, velocity, radius (ABI 7)
    std::memcpy(action_out, a.a, sizeof a.a);
    save_seq(S, seq);
}

static double qpc_now() {
    LARGE_INTEGER c, f; QueryPerformanceCounter(&c); QueryPerformanceFrequency(&f);
    return (double)c.QuadPart / (double)f.QuadPart;
}

// bridge_in: [pending native move x,y,yaw,jump,use,trigger1 (6), pending action (9), hold, pow_yaw, frame clock]
// status_out: [replies with a macro state, native rows written, failure reply (-1 none), failure code, stopped,
//              budget exceeded, physics steps done, final hold]; bridge_out: [pow_yaw]; blast_out: 13 values (layout of blast_in)
DLL int ms_forecast(void* h, void* th, const double* state_in, const double* bridge_in, const double* blast_in, const double* inv_in,
                    int count, int stop_on_ambiguity, int n_prefix, const double* prefix, int tail_kind, double* seq,
                    const float* field, const float* next_field, const double* consts, double budget_s, long long max_steps,
                    double* rows_out, double* reply_out, double* bridge_out, double* blast_out, int* status_out) {
    Map* m = (Map*)h; Terrain* t = (Terrain*)th;
    // budget_s = the planner's remaining time (Python's perf_counter has its own origin, so no absolute deadline here)
    const double stop_at = budget_s > 0 ? qpc_now() + budget_s : 0;
    State s; unpack(state_in, s); pack(s, rows_out);
    double move[6]; std::memcpy(move, bridge_in, sizeof move);
    Pending pending; std::memcpy(pending.a, bridge_in + 6, sizeof pending.a);
    int hold = (int)bridge_in[15]; double pow_yaw = bridge_in[16]; long long clock = (long long)bridge_in[17];
    BlastCtx b{blast_in[0] != 0, blast_in[1] != 0, blast_in[2] != 0, blast_in[3] != 0, blast_in[4] != 0, blast_in[5] != 0,
               blast_in[6] != 0, blast_in[7] != 0, blast_in[8], blast_in[9], blast_in[10], blast_in[11], blast_in[12]};
    InvCtx inv{inv_in[0] != 0, inv_in[1] != 0, inv_in[2] != 0, inv_in[3], (int)inv_in[4]};
    Consts C{consts[0], consts[1], consts[2], consts[3], consts[4], consts[5], consts[6] != 0, consts[7], consts[8], consts[9], consts[10],
             consts[11], consts[12]};   // ABI 7: 13 constants
    SeqSpec S{};
    if (tail_kind == 1) load_seq(S, seq, field, next_field);
    P3 brake_goal{0, 0, 0};
    if (tail_kind == 2 || tail_kind == 3) brake_goal = P3{seq[Q_GEM], seq[Q_GEM + 1], seq[Q_GEM + 2]};
    const int brake_prefix = tail_kind == 1 ? (int)seq[Q_BRAKE_PREFIX] : 0;
    const bool has_aim = tail_kind == 1 && seq[Q_HAS_AIM] != 0; const double aim = has_aim ? seq[Q_AIM] : 0.;
    if (tail_kind == 1 && brake_prefix > 0) brake_goal = P3{seq[Q_GEM], seq[Q_GEM + 1], seq[Q_GEM + 2]};
    int rows = 1, done_replies = 0, fail_reply = -1, fail_code = 0, stopped = 0, budget = 0; long long steps = 0;
    std::vector<Event> ev;
    for (int i = 0; i < count; i++) {
        if ((stop_at > 0 && qpc_now() >= stop_at) || (max_steps >= 0 && steps > max_steps)) { budget = 1; break; }
        double* R = reply_out + (size_t)i * R_SIZE;
        Action act;
        if (i < n_prefix) std::memcpy(act.a, prefix + (size_t)i * 9, sizeof act.a);
        else if (tail_kind == 1) {
            if (i < brake_prefix) act = proposal(C, brake_goal, 0., -1, -1, true, false, 0., 0., s.p, s.v);
            else {
                act = seq_call(*t, C, S, s.p, s.v, i - brake_prefix, s.radius);
                if (has_aim && act.a[6] != 0) act.a[7] -= aim;
            }
        }
        else if (tail_kind == 2) act = proposal(C, brake_goal, 0., -1, -1, true, false, 0., 0., s.p, s.v);
        else if (tail_kind == 3) act = proposal(C, brake_goal, 0., -1, -1, false, false, 0., 0., s.p, s.v);   // landing.LandingContinuation
        else { fail_reply = i; fail_code = 20; break; }
        std::memcpy(R + R_ACTION, act.a, sizeof act.a);
        double next_move[6], camera_yaw; translated(pending, hold, pow_yaw, next_move, camera_yaw);
        std::memcpy(R + R_MOVE0, move, sizeof move); std::memcpy(R + R_MOVE1, next_move, sizeof next_move); R[R_CAMERA] = camera_yaw;
        for (int k = 0; k < 4; k++) R[R_AMB + k] = 0;
        R[R_DONE] = 0; R[R_IMPULSE] = 0;
        int code = inventory_check(inv, s, pending);
        if (code) { fail_reply = i; fail_code = -code; break; }
        double push[3] = {0, 0, 0}, dv[3] = {0, 0, 0};
        R[R_METER] = b.value; R[R_SPECIAL] = b.special;
        int fired = blast_event(b, s, pending.a[8] != 0, push, dv);
        if (fired < 0) { fail_reply = i; fail_code = -fired; break; }
        R[R_IMPULSE] = fired; for (int q = 0; q < 3; q++) { R[R_PUSH + q] = push[q]; R[R_DV + q] = dv[q]; }
        int done = 0; bool amb_any = false;
        for (int k = 0; k < 4; k++) {
            s.clock_ms = clock + (long long)i * 64 + 16LL * k;
            const double* mv = k == 0 ? move : next_move;
            if (k == 0) {
                if (fired == 1) s.v = add(s.v, V3{dv[0], dv[1], dv[2]});
                double y = f32(camera_yaw); s.yaw = y; s.next_yaw = y; s.has_next_yaw = true;
            }
            Move mvv{mv[0], mv[1], mv[2], mv[3] != 0, mv[4] != 0, mv[5] != 0};
            ev.clear();
            V3 pickup_start = s.p;
            int sc = step(*m, s, mvv, 16, ev);
            if (sc) { fail_code = sc; break; }
            if (tail_kind == 1 && S.through && S.has_box && !S.passed) {
                V3 a = pickup_start, p = s.p; double r = s.radius + .2;
                S.passed = pymax(a.x,p.x)+r >= S.box[0] && pymax(a.y,p.y)+r >= S.box[1] && pymax(a.z,p.z)+r >= S.box[2]
                        && pymin(a.x,p.x)-r <= S.box[3] && pymin(a.y,p.y)-r <= S.box[4] && pymin(a.z,p.z)-r <= S.box[5];
            }
            steps++;
            int ac = blast_advance(b, k == 0 && fired == 1, 16);
            if (ac) { fail_code = -ac; break; }
            pack(s, rows_out + (size_t)rows * S_SIZE); rows++; done++;
            bool amb = false; for (auto& e : ev) if (e.kind == 1) { amb = true; break; }
            R[R_AMB + k] = amb; amb_any = amb_any || amb;
            if (stop_on_ambiguity && amb) break;
        }
        R[R_DONE] = done; R[R_BLAST_VALUE] = b.value; R[R_BLAST_SPECIAL] = b.special;
        if (fail_code) { fail_reply = i; break; }
        done_replies = i + 1;
        std::memcpy(move, next_move, sizeof move); std::memcpy(pending.a, act.a, sizeof act.a);
        if (stop_on_ambiguity && amb_any) { stopped = 1; break; }
    }
    status_out[0] = done_replies; status_out[1] = rows; status_out[2] = fail_reply; status_out[3] = fail_code;
    status_out[4] = stopped; status_out[5] = budget; status_out[6] = (int)steps; status_out[7] = hold;
    bridge_out[0] = pow_yaw;
    double bo[13] = {(double)b.present, (double)b.enabled, (double)b.frozen, (double)b.triple, (double)b.fireball, (double)b.cannon,
                     (double)b.special, (double)b.charging, b.value, b.required, b.special_power, b.power, b.charge_ms};
    std::memcpy(blast_out, bo, sizeof bo);
    if (tail_kind == 1) save_seq(S, seq);
    return 0;
}

// libm checks (tools): op 0 cos, 1 sin, 2 atan2, 3 sqrt, 4 pow, 5 fmod
DLL double ms_math(int op, double a, double b) {
    switch (op) { case 0: return std::cos(a); case 1: return std::sin(a); case 2: return std::atan2(a, b);
                  case 3: return std::sqrt(a); case 4: return std::pow(a, b); default: return std::fmod(a, b); }
}
DLL void ms_proposal(const double* goal, double offset, int jump_at, int use_at, int brake, int has_sd, double sd, double min_speed,
                     const double* p, const double* v, const double* consts, double* out) {
    Consts C{consts[0], consts[1], consts[2], consts[3], consts[4], consts[5], consts[6] != 0, consts[7], consts[8], consts[9], consts[10],
             consts[11], consts[12]};   // ABI 7: 13 constants
    Action a = proposal(C, P3{goal[0], goal[1], goal[2]}, offset, jump_at, use_at, brake != 0, has_sd != 0, sd, min_speed, load(p), load(v));
    std::memcpy(out, a.a, sizeof a.a);
}
DLL double ms_now() { return qpc_now(); }
