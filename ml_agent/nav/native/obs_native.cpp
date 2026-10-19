// Terrain queries of the learned model's observation in C++ (2026-10-19, search speed-up toward real-time play).
// Line-for-line ports of terrain_obs.TerrainMap (heights_at, floor_height, _march, edge_rays, gap_along) and
// nav/terrain.TerrainGrid (crop / _crop_one), bit-identical to the NumPy 2 originals:
//  * a Python float meeting a float32 array is cast to float32 first (NEP 50 weak scalars), so the arithmetic is float32;
//    a NumPy float64 scalar is strong and promotes the array to float64 (gap_along's heading, flagged by the caller);
//  * reductions along the levels axis accumulate sequentially from level 0 (numpy add.reduce over a non-contiguous axis);
//  * argmin / argmax return the first occurrence;
//  * np.argsort's tie order is not reproduced (the default kind is not stable for 4+ elements on this build): where a
//    crop sample has two levels at exactly the same distance from z with different values, the call reports it and the
//    caller recomputes that crop with NumPy.
// Built with the marble_sim flags (/O2 /fp:strict): no contraction, IEEE single and double rounding.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <smmintrin.h>

#define DLL extern "C" __declspec(dllexport)

namespace {
const float FINF = std::numeric_limits<float>::infinity();
const double DINF = std::numeric_limits<double>::infinity();
const float FNAN = std::numeric_limits<float>::quiet_NaN();
// inline classification (MSVC's std::isnan / std::isfinite can be CRT calls); same truth values for every input
inline bool fnan(float v) { return v != v; }
inline bool ffin(float v) { return std::fabs(v) <= std::numeric_limits<float>::max(); }
inline bool finf(float v) { return std::fabs(v) == std::numeric_limits<float>::infinity(); }
inline bool dnan(double v) { return v != v; }
inline bool dinf(double v) { return std::fabs(v) == std::numeric_limits<double>::infinity(); }

struct Grid {
    int K = 0, H = 0, W = 0;
    float x0 = 0, y0 = 0, res = 1;
    const float* heights = nullptr;      // (K, H, W), owned by Python (kept alive there)
    float follow_tol = .4f, dz_scale = 10.f, ray_range = 20.f;
    double follow_tol_d = .4;
    int ray_steps = 40; std::vector<float> ray_s;   // _RAY_S
    std::vector<float> ray_dirs;                    // _RAY_DIRS (16 x 2)
    int ray_headings = 16, gem_rays = 3;
    double ray_step = .5;
};

// np.rint (round half to even, independent of the MXCSR mode) then astype(int64); NaN / out-of-range give a value outside
// any grid (numpy's INT64_MIN for those also fails the bounds test)
inline long long rint_ll(float v) {
    if (!(std::fabs(v) < 4.0e9f)) return std::numeric_limits<long long>::min();
    const float r = _mm_cvtss_f32(_mm_round_ss(_mm_setzero_ps(), _mm_set_ss(v), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    return (long long)r;
}

// terrain_obs.heights_at for one float32 point: rint((x - x0) / res) in float32, NaN outside the grid
inline void heights_at1(const Grid& g, float x, float y, float* out) {
    const float fi = (x - g.x0) / g.res, fj = (y - g.y0) / g.res;
    const long long i = rint_ll(fi), j = rint_ll(fj);
    if (i >= 0 && i < g.W && j >= 0 && j < g.H) {
        const size_t plane = (size_t)g.H * g.W, off = (size_t)j * g.W + (size_t)i;
        for (int k = 0; k < g.K; k++) out[k] = g.heights[k * plane + off];
    } else {
        for (int k = 0; k < g.K; k++) out[k] = FNAN;
    }
}

inline float clip1(float v) { return v < -1.f ? -1.f : (v > 1.f ? 1.f : v); }

// terrain_obs.floor_height: the level nearest z (first on ties); z itself when no level
double floor_height(const Grid& g, double x, double y, double z) {
    float h[64]; heights_at1(g, (float)x, (float)y, h);
    const float zf = (float)z; int best = -1; float bd = 0;
    for (int k = 0; k < g.K; k++) {
        if (fnan(h[k])) continue;
        const float d = std::fabs(h[k] - zf);
        if (best < 0 || d < bd || fnan(bd)) { best = k; bd = d; }
    }
    return best < 0 ? z : (double)h[best];
}

// TerrainGrid._crop_one -> out (3, N): nearest-level relative height, present, second level. Returns 1 if a sample has a
// tie whose order matters (the caller recomputes with NumPy).
int crop_one(const Grid& g, double x, double y, double z, const float* offsets, int n, float* out) {
    const float xf = (float)x, yf = (float)y, zf = (float)z;
    float h[64], rel[64], dist[64];
    int amb = 0;
    for (int p = 0; p < n; p++) {
        heights_at1(g, offsets[2 * p] + xf, offsets[2 * p + 1] + yf, h);
        for (int k = 0; k < g.K; k++) { rel[k] = h[k] - zf; dist[k] = fnan(rel[k]) ? FINF : std::fabs(rel[k]); }
        // smallest distance (and the set tied with it), then the next one
        int a = 0; for (int k = 1; k < g.K; k++) if (dist[k] < dist[a]) a = k;
        int ties0 = 0; bool differ0 = false;
        for (int k = 0; k < g.K; k++) if (dist[k] == dist[a]) { ties0++; if (!(rel[k] == rel[a]) && !(fnan(rel[k]) && fnan(rel[a]))) differ0 = true; }
        if (differ0) amb = 1;
        const float first = rel[a];
        const bool present = ffin(first);
        out[p] = present ? clip1(first / g.dz_scale) : -1.f;
        out[n + p] = present ? 1.f : 0.f;
        float ch2 = 0.f;
        if (g.K > 1) {
            float second;
            if (ties0 >= 2) second = first;                 // the second sorted element shares the smallest distance
            else {
                int b = -1; for (int k = 0; k < g.K; k++) if (k != a && (b < 0 || dist[k] < dist[b])) b = k;
                for (int k = 0; k < g.K; k++) if (k != a && k != b && dist[k] == dist[b] && !(rel[k] == rel[b]) && !(fnan(rel[k]) && fnan(rel[b]))) amb = 1;
                second = rel[b];
            }
            ch2 = ffin(second) ? clip1(second / g.dz_scale) : 0.f;
        }
        out[2 * n + p] = ch2;
    }
    return amb;
}

// terrain_obs._march (fast path for rays without stacked levels in range, then the per-step loop for the rest)
void march(const Grid& g, float xf, float yf, float z0f, const float* dirs, const long long* max_steps, int R, float* edge_dist, float* edge_dz) {
    const int S = g.ray_steps, K = g.K;
    static thread_local std::vector<float> h; h.resize((size_t)K * R * S);
    float col[64];
    for (int r = 0; r < R; r++) for (int s = 0; s < S; s++) {
        heights_at1(g, xf + dirs[2 * r] * g.ray_s[s], yf + dirs[2 * r + 1] * g.ray_s[s], col);
        for (int k = 0; k < K; k++) h[((size_t)k * R + r) * S + s] = col[k];
    }
    static thread_local std::vector<float> lev, prev; static thread_local std::vector<int> count;
    lev.resize((size_t)R * S); prev.resize((size_t)R * S); count.resize((size_t)R * S);
    std::vector<char> active(R), simple(R);
    for (int r = 0; r < R; r++) {
        edge_dist[r] = FINF; edge_dz[r] = 0.f; active[r] = max_steps[r] > 0;
        bool stacked = false;
        for (int s = 0; s < S; s++) {
            int c = 0; float sum = 0.f;
            for (int k = 0; k < K; k++) {
                const float v = h[((size_t)k * R + r) * S + s]; const bool fin = !fnan(v);
                c += fin ? 1 : 0;
                sum = k == 0 ? (fin ? v : 0.f) : sum + (fin ? v : 0.f);
            }
            count[(size_t)r * S + s] = c;
            lev[(size_t)r * S + s] = c == 1 ? sum : FNAN;
            if (c > 1 && s < max_steps[r]) stacked = true;
        }
        simple[r] = !stacked;
    }
    for (int r = 0; r < R; r++) {
        if (!simple[r]) continue;
        int first = -1;
        for (int s = 0; s < S; s++) {
            const float pv = s == 0 ? z0f : lev[(size_t)r * S + s - 1]; prev[(size_t)r * S + s] = pv;
            const float lv = lev[(size_t)r * S + s];
            const float diff = fnan(lv) ? FINF : std::fabs(lv - pv);
            if (diff > g.follow_tol && s < max_steps[r]) { first = s; break; }
        }
        if (first >= 0) {
            edge_dist[r] = g.ray_s[first];
            const float level = lev[(size_t)r * S + first], base = prev[(size_t)r * S + first];
            if (!fnan(level)) {
                float d = level - base; if (fnan(d)) d = 0.f;
                else if (finf(d)) d = d > 0 ? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
                edge_dz[r] = clip1(d / g.dz_scale);
            } else edge_dz[r] = -1.f;
        }
        active[r] = 0;
    }
    std::vector<float> z_cur(R, z0f);
    for (int s = 0; s < S; s++) {
        bool any = false; for (int r = 0; r < R; r++) if (active[r]) { any = true; break; }
        if (!any) break;
        for (int r = 0; r < R; r++) {
            if (!active[r]) continue;
            int kb = 0; float best = 0, level = 0;
            for (int k = 0; k < K; k++) {
                const float c = h[((size_t)k * R + r) * S + s];
                const float d = fnan(c) ? FINF : std::fabs(c - z_cur[r]);
                if (k == 0 || d < best) { kb = k; best = d; level = c; }
            }
            (void)kb;
            const bool cont = best <= g.follow_tol;
            if (!cont) {
                edge_dist[r] = g.ray_s[s];
                if (ffin(best)) {
                    float d = level - z_cur[r]; if (fnan(d)) d = 0.f;
                    else if (finf(d)) d = d > 0 ? std::numeric_limits<float>::max() : -std::numeric_limits<float>::max();
                    edge_dz[r] = clip1(d / g.dz_scale);
                } else edge_dz[r] = -1.f;
                active[r] = 0;
            }
            if (active[r] && cont) z_cur[r] = level;
            active[r] = active[r] && (s + 1 < max_steps[r]);
        }
    }
}

// Python 3.9 math.hypot (vector_norm), as in nav/native/controllers.inc
inline double py_hypot(double a, double b) {
    double x0 = std::fabs(a), x1 = std::fabs(b), mx = 0.0;
    if (x0 > mx) mx = x0;
    if (x1 > mx) mx = x1;
    if (dinf(mx)) return mx;
    if (dnan(x0) || dnan(x1)) return std::numeric_limits<double>::quiet_NaN();
    if (mx == 0.0) return mx;
    double csum = 1.0, frac = 0.0, x, old;
    x = x0 / mx; x = x * x; old = csum; csum += x; frac += (old - csum) + x;
    x = x1 / mx; x = x * x; old = csum; csum += x; frac += (old - csum) + x;
    return mx * std::sqrt(csum - 1.0 + frac);
}
}  // namespace

DLL void* ob_create(int K, int H, int W, double x0, double y0, double res, const float* heights, double follow_tol, double dz_scale,
                    double ray_range, double ray_step, int ray_steps, const float* ray_s, int ray_headings, const float* ray_dirs, int gem_rays) {
    Grid* g = new Grid();
    g->K = K; g->H = H; g->W = W; g->x0 = (float)x0; g->y0 = (float)y0; g->res = (float)res; g->heights = heights;
    g->follow_tol = (float)follow_tol; g->follow_tol_d = follow_tol; g->dz_scale = (float)dz_scale; g->ray_range = (float)ray_range;
    g->ray_step = ray_step; g->ray_steps = ray_steps; g->ray_s.assign(ray_s, ray_s + ray_steps);
    g->ray_headings = ray_headings; g->ray_dirs.assign(ray_dirs, ray_dirs + 2 * ray_headings); g->gem_rays = gem_rays;
    return g;
}
DLL void ob_destroy(void* g) { delete (Grid*)g; }
DLL int ob_abi() { return 1; }

DLL double ob_floor_height(void* gh, double x, double y, double z) { return floor_height(*(Grid*)gh, x, y, z); }

// TerrainGrid.crop: (6, 32, 32) = fine crop then coarse crop. Returns 1 when a tie needs NumPy's sort order.
DLL int ob_crop(void* gh, double x, double y, double z, const float* fine, const float* coarse, int n, float* out) {
    const Grid& g = *(Grid*)gh;
    int amb = crop_one(g, x, y, z, fine, n, out);
    amb |= crop_one(g, x, y, z, coarse, n, out + 3 * n);
    return amb;
}

// TerrainMap.edge_rays(x, y, z, gem_rels): gem_rels (n_gem x 4: dx, dy, dz, present) -> out (EDGE_DIM float32)
DLL void ob_edge_rays(void* gh, double x, double y, double z, const double* gem_rels, int n_gem, float* out) {
    const Grid& g = *(Grid*)gh;
    const double z0 = floor_height(g, x, y, z);
    const int R = g.ray_headings + g.gem_rays;
    std::vector<float> dirs(2 * R), dist(R), dz(R), gem_d(g.gem_rays, 0.f); std::vector<long long> max_steps(R, g.ray_steps);
    for (int k = 0; k < g.ray_headings; k++) { dirs[2 * k] = g.ray_dirs[2 * k]; dirs[2 * k + 1] = g.ray_dirs[2 * k + 1]; }
    for (int i = 0; i < g.gem_rays; i++) {
        double dx = 0, dy = 0; bool present = false;
        if (i < n_gem) { dx = gem_rels[4 * i]; dy = gem_rels[4 * i + 1]; present = gem_rels[4 * i + 3] != 0; }
        const double d = py_hypot(dx, dy);
        const int r = g.ray_headings + i;
        if (present && d > 1e-3) {
            dirs[2 * r] = (float)(dx / d); dirs[2 * r + 1] = (float)(dy / d);
            gem_d[i] = (float)d;
            const long long m = (long long)std::ceil(d / g.ray_step);
            max_steps[r] = m < g.ray_steps ? m : g.ray_steps;
        } else {
            dirs[2 * r] = 1.f; dirs[2 * r + 1] = 0.f; max_steps[r] = 0;
        }
    }
    march(g, (float)x, (float)y, (float)z0, dirs.data(), max_steps.data(), R, dist.data(), dz.data());
    const int E = 2 * (g.ray_headings + g.gem_rays);
    for (int k = 0; k < E; k++) out[k] = 0.f;
    for (int k = 0; k < g.ray_headings; k++) {
        out[2 * k] = ffin(dist[k]) ? dist[k] / g.ray_range : 1.f;
        out[2 * k + 1] = dz[k];
    }
    for (int i = 0; i < g.gem_rays; i++) {
        const int j = g.ray_headings * 2 + i * 2;
        if (gem_d[i] > 0) {
            const float gd = dist[g.ray_headings + i];
            if (ffin(gd)) { const float q = gd / gem_d[i]; out[j] = q < 1.f ? q : 1.f; } else out[j] = 1.f;
            out[j + 1] = dz[g.ray_headings + i];
        }
    }
}

// TerrainMap.gap_along(x, y, z, ux, uy) with max_u = ray_range. sx / sy: ux / uy is a NumPy float64 (strong: the sample
// points are computed in float64 and then rounded to float32), else a Python float (weak: float32 arithmetic).
// out = (lip, gap, landing dz) as Python would return them.
DLL void ob_gap_along(void* gh, double x, double y, double z, double ux, double uy, int sx, int sy, double* out) {
    const Grid& g = *(Grid*)gh;
    const double z0 = floor_height(g, x, y, z);
    const int n = (int)std::nearbyint(g.ray_range / g.ray_step);
    std::vector<float> h((size_t)g.K * n);
    std::vector<float> ds(n);
    float col[64];
    for (int s = 0; s < n; s++) {
        ds[s] = (float)((double)(s + 1) * g.ray_step);
        const float px = sx ? (float)(x + ux * (double)ds[s]) : (float)x + (float)ux * ds[s];
        const float py = sy ? (float)(y + uy * (double)ds[s]) : (float)y + (float)uy * ds[s];
        heights_at1(g, px, py, col);
        for (int k = 0; k < g.K; k++) h[(size_t)k * n + s] = col[k];
    }
    double z_cur = z0, lip = DINF; int lip_i = -1;
    for (int s = 0; s < n; s++) {
        int best = -1; float bd = 0;
        const float zc = (float)z_cur;
        for (int k = 0; k < g.K; k++) {
            const float c = h[(size_t)k * n + s];
            const float d = std::fabs((ffin(c) ? c : FINF) - zc);
            if (best < 0 || d < bd) { best = k; bd = d; }
        }
        bool fin = false; for (int k = 0; k < g.K; k++) if (ffin(h[(size_t)k * n + s])) { fin = true; break; }
        if (fin) {
            const float c = h[(size_t)best * n + s];
            if (std::fabs((double)c - z_cur) <= g.follow_tol_d) { z_cur = (double)c; continue; }
        }
        lip = (double)ds[s]; lip_i = s;
        break;
    }
    if (lip_i < 0) { out[0] = DINF; out[1] = DINF; out[2] = 0.0; return; }
    for (int s = lip_i; s < n; s++) {
        int best = -1; float bd = 0, bz = 0;
        const float zc = (float)z_cur;
        for (int k = 0; k < g.K; k++) {
            const float c = h[(size_t)k * n + s];
            if (!ffin(c)) continue;
            const float d = std::fabs(c - zc);
            if (best < 0 || d < bd) { best = k; bd = d; bz = c; }
        }
        if (best >= 0) {
            out[0] = lip; out[1] = (double)(ds[s] - (float)lip); out[2] = (double)(bz - zc);
            return;
        }
    }
    out[0] = lip; out[1] = DINF; out[2] = 0.0;
}
