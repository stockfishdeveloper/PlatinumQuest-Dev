// Plan controllers, routing and the 64 ms bridge in C++ (2026-10-16), included by marble_sim.cpp inside its namespace.
// Ports, line for line: nav/routing.py (line_clear, path_waypoint, RoutePlanner.navigate / distance), the walk-grid
// helpers of nav/terrain.py (cell_of, level_at, _nearest_walkable, dist_at), the ground clearance of
// nav/marble_dynamics/geometry_route.py + ground_cache.py, gem_demo.Proposal, search.SequenceProposal (and the brake
// tail of BrakeContinuation), nav/joystick.action_to_joystick, bridge.reply / translated / quantize, blast.event /
// advance and inventory.check. NumPy 2 promotion is reproduced where it matters: a Python float meeting a float32
// operand is cast to float32 first (cell_of, level_at, the nearest-level search), round() of a NumPy float rounds
// half to even, Python's round(x, n) and format(x, '.6g') are correctly rounded decimal conversions.


// ------------------------------------------------------------------ Python scalar helpers
inline double py_round(double x, int nd) {
    if (!std::isfinite(x)) return x;
    char buf[400]; std::snprintf(buf, sizeof buf, "%.*f", nd, x); return std::strtod(buf, nullptr);
}
inline double g6(double x) { char buf[64]; std::snprintf(buf, sizeof buf, "%.6g", x); return std::strtod(buf, nullptr); }
inline double py_hypot(double a, double b) {          // Python 3.9 math.hypot (vector_norm)
    double x0 = std::fabs(a), x1 = std::fabs(b), mx = 0.0;
    if (x0 > mx) mx = x0;
    if (x1 > mx) mx = x1;
    if (std::isinf(mx)) return mx;
    if (std::isnan(x0) || std::isnan(x1)) return std::numeric_limits<double>::quiet_NaN();
    if (mx == 0.0) return mx;
    double csum = 1.0, frac = 0.0, x, old;
    x = x0 / mx; x = x * x; old = csum; csum += x; frac += (old - csum) + x;
    x = x1 / mx; x = x * x; old = csum; csum += x; frac += (old - csum) + x;
    return mx * std::sqrt(csum - 1.0 + frac);
}
inline double norm2(double a, double b) { return std::sqrt(a * a + b * b); }   // np.linalg.norm of a 2-vector
inline double clip01(double x) { return x < 0. ? 0. : (x > 1. ? 1. : x); }
// round(x, 6) of a NumPy float64: rint(x * 1e6) / 1e6 (not the correctly rounded decimal of a Python float)
inline double np_round6(double x) { return std::nearbyint(x * 1e6) / 1e6; }
inline double round6(double x, bool numpy) { return numpy ? np_round6(x) : py_round(x, 6); }

// ------------------------------------------------------------------ walk grid
struct Terrain {
    float x0 = 0, y0 = 0; double res = 1; int KW = 0, H = 0, W = 0;
    std::vector<float> walk_z, wxs, wys;
    std::vector<long long> indptr, indices, node_idx, nodes; std::vector<double> gdata; bool has_graph = false;
    double drop_edge = 2.0, step_up = 1.5, level_merge = .5;
    // ground clearance (GroundClearance + MemoGroundClearance): obstacle triangles = normal z <= .4
    std::vector<uint8_t> obstacle; double clear_radius = .189750001;
    std::unordered_map<std::string, bool> point_cache, segment_cache;
    Map* map = nullptr;
    std::vector<float> edge;   // walk grid distance to the nearest drop (terrain.edge_dist, KW x H x W); empty = none (ABI 7)
};
inline float wz(const Terrain& t, int k, int j, int i) { return t.walk_z[((size_t)k * t.H + j) * t.W + i]; }
inline float fz(const float* field, const Terrain& t, int k, int j, int i) { return field[((size_t)k * t.H + j) * t.W + i]; }

// terrain.cell_of: round((x - wxs[0]) / walk_res). With a plain Python float x NumPy 2 computes in float32 (the float is
// cast to the float32 origin's type first); with a NumPy float64 x it computes in float64. Callers pass which (strong =
// NumPy float64): a line toward a gem (np array) samples NumPy floats, a line toward a waypoint tuple plain floats.
inline double cell_coord(double x, float origin, double res, bool strong) {
    return strong ? (x - (double)origin) / res : (double)(((float)x - origin) / (float)res);
}
inline void cell_of(const Terrain& t, double x, double y, bool sx, bool sy, int& j, int& i) {
    i = (int)std::nearbyint(cell_coord(x, t.x0, t.res, sx)); j = (int)std::nearbyint(cell_coord(y, t.y0, t.res, sy));
}
inline bool in_grid(const Terrain& t, int j, int i) { return 0 <= j && j < t.H && 0 <= i && i < t.W; }

int level_at(const Terrain& t, int j, int i, bool has_z, double z) {
    if (!in_grid(t, j, i)) return -1;
    bool any = false;
    for (int k = 0; k < t.KW; k++) if (std::isfinite(wz(t, k, j, i))) { any = true; break; }
    if (!any) return -1;
    if (!has_z) return 0;
    const float zf = (float)z; int best = 0; float bd = 0;
    for (int k = 0; k < t.KW; k++) {
        float zk = wz(t, k, j, i); float d = std::fabs((std::isfinite(zk) ? zk : 1e9f) - zf);
        if (k == 0 || d < bd) { bd = d; best = k; }
    }
    return best;
}

bool nearest_walkable(const Terrain& t, int j, int i, int radius, bool has_z, double z, int& bk, int& bj, int& bi) {
    bool found = false; double b0 = 1e9, b1 = 1e9;
    for (int dj = -radius; dj <= radius; dj++) for (int di = -radius; di <= radius; di++) {
        int jj = j + dj, ii = i + di;
        if (!in_grid(t, jj, ii)) continue;
        int k = -1; float bd = 0;
        for (int kk = 0; kk < t.KW; kk++) {
            float zk = wz(t, kk, jj, ii);
            if (!std::isfinite(zk)) continue;
            if (!has_z) { k = kk; break; }
            float d = std::fabs(zk - (float)z);
            if (k < 0 || d < bd) { bd = d; k = kk; }
        }
        if (k < 0) continue;
        double key0 = (double)(dj * dj + di * di), key1 = has_z ? std::fabs((double)wz(t, k, jj, ii) - z) : 0.0;
        if (key0 < b0 || (key0 == b0 && key1 < b1)) { found = true; bk = k; bj = jj; bi = ii; b0 = key0; b1 = key1; }
    }
    return found;
}

double dist_at(const Terrain& t, const float* field, double x, double y, double gx, double gy, bool has_z, double z, bool strong) {
    int j, i; cell_of(t, x, y, strong, strong, j, i);
    int k = level_at(t, j, i, has_z, z);
    if (k >= 0 && std::isfinite(fz(field, t, k, j, i))) return (double)fz(field, t, k, j, i);
    int ck, cj, ci;
    if (nearest_walkable(t, j, i, 2, has_z, z, ck, cj, ci) && std::isfinite(fz(field, t, ck, cj, ci))) return (double)fz(field, t, ck, cj, ci) + t.res;
    return py_hypot(x - gx, y - gy) + 5.0;
}

// ------------------------------------------------------------------ ground clearance (geometry_route.GroundClearance)
// mesh.closest_points for one point and the obstacle triangles in range (einsum order x, z, y; sums sequential)
bool point_clear(Terrain& t, V3 c) {
    char key[96]; std::memcpy(key, &c, sizeof(V3));
    std::string k(key, sizeof(V3));
    auto it = t.point_cache.find(k);
    if (it != t.point_cache.end()) return it->second;
    Map& m = *t.map; const double radius = t.clear_radius - 1e-6;
    double lo[3] = {c.x - radius, c.y - radius, c.z - radius}, hi[3] = {c.x + radius, c.y + radius, c.z + radius};
    static thread_local std::vector<int> ids; query_aabb(m, lo, hi, ids);
    bool clear = true;
    for (int id : ids) {
        if (!t.obstacle[id]) continue;
        V3 a = load(&m.tri[id * 9]), b = load(&m.tri[id * 9 + 3]), cc = load(&m.tri[id * 9 + 6]);
        V3 ab = load(&m.ab[id * 3]), ac = load(&m.ac[id * 3]), n = load(&m.fcross[id * 3]); double nn = m.nn[id];
        double height = nn > 0 ? kdot(sub(c, a), n) / nn : 0.;
        V3 proj = sub(c, mul(n, height)); V3 rel = sub(proj, a);
        double u = nn > 0 ? kdot(cross3(rel, ac), n) / nn : 0., v = nn > 0 ? kdot(cross3(ab, rel), n) / nn : 0.;
        bool inside = nn > 0 && u >= 0 && v >= 0 && u + v <= 1;
        V3 closest;
        if (inside) closest = proj;
        else {
            V3 xs[3] = {a, b, cc}, ys[3] = {b, cc, a}; double bestd = 0; int bestk = 0; V3 opts[3];
            for (int q = 0; q < 3; q++) {
                V3 e = sub(ys[q], xs[q]); double ee = kdot(e, e);
                double s = clip01(ee > 0 ? kdot(sub(c, xs[q]), e) / ee : 0.);
                opts[q] = add(xs[q], mul(e, s));
                V3 d = sub(opts[q], c); double dd = (d.x * d.x + d.y * d.y) + d.z * d.z;
                if (q == 0 || dd < bestd) { bestd = dd; bestk = q; }
            }
            closest = opts[bestk];
        }
        V3 d = sub(closest, c);
        if (std::sqrt((d.x * d.x + d.y * d.y) + d.z * d.z) <= radius + 1e-7) { clear = false; break; }
    }
    if (t.point_cache.size() >= 200000) t.point_cache.clear();   // bounded memory (a pure predicate: dropping entries only costs time)
    t.point_cache[k] = clear;
    return clear;
}

bool obstacle_sweep(Terrain& t, V3 start, V3 end, double radius) {   // CompiledSweepMap.sweep_sphere on obstacles, no native dt
    Map& m = *t.map;
    V3 velocity = sub(end, start); double speed2 = dot(velocity, velocity);
    if (speed2 < 1e-18) return false;
    double lo[3] = {std::min(start.x, end.x) - radius, std::min(start.y, end.y) - radius, std::min(start.z, end.z) - radius};
    double hi[3] = {std::max(start.x, end.x) + radius, std::max(start.y, end.y) + radius, std::max(start.z, end.z) + radius};
    static thread_local std::vector<int> all, ids; query_aabb(m, lo, hi, all);
    ids.clear(); for (int id : all) if (t.obstacle[id]) ids.push_back(id);
    if (ids.empty()) return false;
    const bool single = ids.size() == 1; const double inf = std::numeric_limits<double>::infinity();
    const double approach_limit = -.001 * std::sqrt(speed2);
    for (size_t k = 0; k < ids.size(); k++) {
        const int id = ids[k]; V3 normal = load(&m.planes[id * 4]);
        const double approach = mv_row(normal, velocity, single), distance = mv_row(normal, start, single) + m.planes[id * 4 + 3];
        if (!(approach < approach_limit)) continue;
        double best = inf;
        {
            const double tt = (radius - distance) / approach;
            const V3 point = sub(add(start, mul(velocity, tt)), mul(normal, radius));
            const V3 rel = sub(point, load(&m.tri[id * 9])); const double nnv = m.nn[id];
            const double u = nnv > 0 ? kdot(cross3(rel, load(&m.ac[id * 3])), load(&m.fcross[id * 3])) / nnv : 0;
            const double v = nnv > 0 ? kdot(cross3(load(&m.ab[id * 3]), rel), load(&m.fcross[id * 3])) / nnv : 0;
            if (nnv > 0 && tt > 0 && tt < 1 && u >= -1e-7 && v >= -1e-7 && u + v <= 1 + 1e-7) best = tt;
        }
        for (int j = 0; j < 3; j++) {
            const V3 a = load(&m.source[id * 9 + j * 3]), b = load(&m.source[id * 9 + ((j + 1) % 3) * 3]);
            const V3 edge = sub(b, a), pv = sub(start, a);
            const double edge_dot = mv_row(edge, velocity, single), vertex_dot = mv_row(pv, velocity, single);
            const double length2 = kdot(edge, edge);
            const double ve = length2 > 0 ? edge_dot / length2 : 0, pe = length2 > 0 ? kdot(pv, edge) / length2 : 0;
            const V3 vr = sub(velocity, mul(edge, ve)), pr = sub(pv, mul(edge, pe));
            const double qa = kdot(vr, vr), qb = 2 * kdot(pr, vr), qc = kdot(pr, pr) - radius * radius;
            const double disc = qb * qb - 4 * qa * qc, root = std::sqrt(positive(disc));
            const double tt = qa > 1e-18 ? (-qb - root) / (2 * qa) : inf;
            const double along = pe + (std::isfinite(tt) ? tt : 0) * ve;
            if (m.boundary[id * 3 + j] && length2 > 0 && disc >= 0 && tt > 0 && tt < 1 && tt < best && along >= 0 && along <= 1) best = tt;
            const double vqb = 2 * vertex_dot, vqc = kdot(pv, pv) - radius * radius;
            const double vdisc = vqb * vqb - 4 * speed2 * vqc, vroot = std::sqrt(positive(vdisc));
            const double vt = (-vqb - vroot) / (2 * speed2);
            if (vdisc >= 0 && vt > 0 && vt < 1 && vt < best) best = vt;
        }
        if (std::isfinite(best)) return true;
    }
    return false;
}

bool segment_clear(Terrain& t, V3 start, V3 end) {          // MemoGroundClearance.segment over GroundClearance.segment
    double kv[7] = {t.clear_radius, start.x, start.y, start.z, end.x, end.y, end.z};
    bool finite = true; for (double v : kv) if (!std::isfinite(v)) finite = false;
    std::string key((const char*)kv, sizeof kv);
    if (finite) { auto it = t.segment_cache.find(key); if (it != t.segment_cache.end()) return it->second; }
    const double radius = t.clear_radius - 1e-6;
    bool clear;
    if (!point_clear(t, start) || !point_clear(t, end)) clear = false;
    else clear = !obstacle_sweep(t, start, end, radius) && !obstacle_sweep(t, end, start, radius);
    if (finite) { if (t.segment_cache.size() >= 200000) t.segment_cache.clear(); t.segment_cache[key] = clear; }
    return clear;
}

struct P3 { double x, y, z; };

bool line_clear(Terrain& t, double ax, double ay, double bx, double by, bool has_az, double az, bool has_bz, double bz, bool sx, bool sy) {
    const double step = .35;
    int n = std::max((int)std::ceil(py_hypot(bx - ax, by - ay) / step), 1);
    bool has_z = has_az; double z = az; bool have_last = false; int lk = -1, lj = -1, li = -1;
    static thread_local std::vector<P3> pts; pts.clear();
    for (int u = 0; u <= n; u++) {
        double x = ax + (bx - ax) * (double)u / (double)n, y = ay + (by - ay) * (double)u / (double)n;
        int j, i; cell_of(t, x, y, sx, sy, j, i);
        if (have_last && j == lj && i == li) continue;
        int k = level_at(t, j, i, has_z, z);
        if (k < 0) return false;
        double zz = (double)wz(t, k, j, i);
        if (has_z && !(-t.drop_edge <= zz - z && zz - z <= t.step_up)) return false;
        z = zz; has_z = true; have_last = true; lk = k; lj = j; li = i;
        pts.push_back({x, y, zz});
    }
    if (has_bz) {
        int dest = level_at(t, lj, li, true, bz);
        if (dest < 0 || std::fabs((double)wz(t, dest, lj, li) - z) > t.level_merge) return false;
    }
    if (pts.empty() || !(pts.back().x == bx && pts.back().y == by)) pts.push_back({bx, by, z});
    const double lift = t.clear_radius + .001;
    for (size_t q = 0; q + 1 < pts.size(); q++) {
        V3 a{pts[q].x + 0.0, pts[q].y + 0.0, pts[q].z + lift}, b{pts[q + 1].x + 0.0, pts[q + 1].y + 0.0, pts[q + 1].z + lift};
        if (!segment_clear(t, a, b)) return false;
    }
    return true;
}

bool path_waypoint(Terrain& t, const float* field, double mx, double my, P3 gem, bool has_mz, double mz, P3& out) {
    const double lookahead = 12.0;
    int j, i; cell_of(t, mx, my, false, false, j, i);
    int k = level_at(t, j, i, has_mz, mz);
    if (k < 0 || !std::isfinite(fz(field, t, k, j, i))) {
        int ck, cj, ci;
        if (!nearest_walkable(t, j, i, 3, has_mz, mz, ck, cj, ci) || !std::isfinite(fz(field, t, ck, cj, ci))) { out = gem; return false; }
        k = ck; j = cj; i = ci;
    }
    if (line_clear(t, mx, my, gem.x, gem.y, has_mz, mz, true, gem.z, true, true)) { out = gem; return false; }
    if (!t.has_graph) { out = gem; return false; }
    static thread_local std::vector<P3> path; path.clear();
    double travelled = 0.0;
    const int iters = (int)(lookahead * 3 / t.res) + 8;
    for (int it = 0; it < iters; it++) {
        long long node = t.node_idx[((size_t)k * t.H + j) * t.W + i];
        long long lo = t.indptr[node], hi = t.indptr[node + 1];
        if (hi == lo) break;
        long long q = -1; double best = 0; int bk = 0, bj = 0, bi = 0; float bv = 0;
        for (long long e = lo; e < hi; e++) {
            long long nd = t.indices[e]; int kk = (int)t.nodes[nd * 3], jj = (int)t.nodes[nd * 3 + 1], ii = (int)t.nodes[nd * 3 + 2];
            float val = fz(field, t, kk, jj, ii); double cost = t.gdata[e] + (double)val;
            if (q < 0 || cost < best) { q = e; best = cost; bk = kk; bj = jj; bi = ii; bv = val; }
        }
        if (!std::isfinite(best) || bv >= fz(field, t, k, j, i)) break;
        travelled += py_hypot((double)(bi - i), (double)(bj - j)) * t.res;
        k = bk; j = bj; i = bi;
        path.push_back({(double)t.wxs[i], (double)t.wys[j], (double)wz(t, k, j, i)});
        if (fz(field, t, k, j, i) <= 1e-5f || travelled >= lookahead * 3) break;
    }
    if (path.empty()) { out = gem; return false; }
    int pick = -1;
    for (int w = 0; w < (int)path.size(); w++) {
        if (py_hypot(path[w].x - mx, path[w].y - my) > lookahead) break;
        if (line_clear(t, mx, my, path[w].x, path[w].y, has_mz, mz, true, path[w].z, false, false)) pick = w;
    }
    P3 pk = pick >= 0 ? path[pick] : path[0];
    if (py_hypot(pk.x - gem.x, pk.y - gem.y) <= 1.5 && line_clear(t, pk.x, pk.y, gem.x, gem.y, true, pk.z, true, gem.z, true, true)) { out = gem; return false; }
    out = pk; return true;
}

struct Route {                         // RoutePlanner with its goal field already computed (same key)
    const float* field = nullptr; P3 key{0, 0, 0}; bool has_wp = false; P3 wp{0, 0, 0};
};
P3 navigate(Terrain& t, Route& r, P3 pos, P3 gem, bool& routed) {
    double x = pos.x, y = pos.y, z = pos.z;
    if (line_clear(t, x, y, gem.x, gem.y, true, z, true, gem.z, true, true)) { r.has_wp = false; routed = false; return gem; }
    if (r.has_wp && py_hypot(r.wp.x - x, r.wp.y - y) > 2.5 && line_clear(t, x, y, r.wp.x, r.wp.y, true, z, true, r.wp.z, false, false)) { routed = true; return r.wp; }
    P3 w; routed = path_waypoint(t, r.field, x, y, gem, true, z, w);
    r.has_wp = routed; if (routed) r.wp = w;
    return w;
}
double route_distance(Terrain& t, const Route& r, P3 pos, bool strong) {
    if (r.field == nullptr) return std::numeric_limits<double>::infinity();
    return dist_at(t, r.field, pos.x, pos.y, r.key.x, r.key.y, true, pos.z, strong);
}

// ------------------------------------------------------------------ controllers
struct Consts { double cap, brake_decel, throttle_gain, throttle_cap, pickup_min_speed, through_switch; bool route_speed_law;
                double open_top, open_near, open_far, travel_top, travel_near, travel_far; };   // ABI 7: search.OPEN_* / TRAVEL_*

struct Action { double a[9]; unsigned npmask = 0; };   // npmask bit k: a[k] is a NumPy float64 in the Python controller

// action_to_joystick (CAM_LOCK off) -> (fwd, back, left, right, jump, cam_yaw). tnp: the throttle is a NumPy float64
// (Proposal's min(cap, |accel| / gain) below the cap), which makes mx / my and their round(., 6) NumPy's.
unsigned joystick(double dx, double dy, double throttle, bool tnp, int jump, int brake, double vx, double vy, double out[6]) {
    if (brake > 0.5) {
        double sp = std::sqrt(vx * vx + vy * vy);
        if (sp > 0.1) { dx = -vx / sp; dy = -vy / sp; throttle = 1.0; tnp = false; }
    } else {
        double n = std::sqrt(dx * dx + dy * dy);
        if (n > 1e-6) { dx = dx / n; dy = dy / n; }
    }
    double cam, mx, my;
    if (brake > 0.5) { cam = 0.0; mx = dx * throttle; my = dy * throttle; }
    else { cam = 3.141592653589793 / 4.0 - std::atan2(dy, dx); double mm = std::sqrt(2.0) * throttle; mx = dx * mm; my = dy * mm; }
    double comps[4] = {my, -my, -mx, mx}; unsigned mask = 0;
    for (int k = 0; k < 4; k++) {
        bool np = tnp && !(0.0 > comps[k]);           // max(c, 0.0) returns c (its type) unless 0.0 > c
        double c = 0.0 > comps[k] ? 0.0 : comps[k];
        out[k] = round6(c, np); if (np) mask |= 1u << k;
    }
    out[4] = (double)jump; out[5] = py_round(cam, 6);
    return mask;
}

// gem_demo.Proposal(goal, offset, jump_at, use_at, brake, speed_distance, min_speed, top=None, coast=False)(state, 0)
Action proposal(const Consts& C, P3 goal, double offset, int jump_at, int use_at, bool brake, bool has_sd, double speed_distance,
                double min_speed, V3 p, V3 v, bool has_top = false, double top = 0.) {
    Action r;
    if (brake && norm2(v.x, v.y) <= .1) { double z[9] = {0, 0, 0, 0, 0, 0., 0, 0., 0.}; std::memcpy(r.a, z, sizeof z); return r; }
    double dx = goal.x - p.x, dy = goal.y - p.y, distance = norm2(dx, dy);
    double stop = has_sd ? pymax(speed_distance, distance) : distance;
    double target = pymin(has_top ? top : C.cap, pymax(min_speed, std::sqrt(pymax(0., (2 * C.brake_decel) * (stop - .25)))));
    double m = pymax(distance, 1e-9);
    double ax = (dx / m * target - v.x) * 2., ay = (dy / m * target - v.y) * 2.;
    if (offset != 0.) {
        double c = std::cos(offset), s = std::sin(offset);
        double rx = c * ax - s * ay, ry = s * ax + c * ay; ax = rx; ay = ry;
    }
    double magnitude = norm2(ax, ay);
    double throttle = 0.; bool tnp = false;
    if (magnitude > .05) {
        double x = magnitude / C.throttle_gain;
        if (x < C.throttle_cap) { throttle = x; tnp = true; } else throttle = C.throttle_cap;
    }
    double js[6]; r.npmask = joystick(ax, ay, throttle, tnp, jump_at == 0 ? 1 : 0, brake ? 1 : 0, v.x, v.y, js);
    r.a[0] = js[0]; r.a[1] = js[1]; r.a[2] = js[2]; r.a[3] = js[3]; r.a[4] = js[4]; r.a[5] = js[5];
    r.a[6] = use_at == 0 ? 1 : 0; r.a[7] = std::atan2(dx, dy); r.a[8] = 0.;
    return r;
}

struct SeqSpec {                       // search.SequenceProposal
    Route route; P3 gem; double offset, scale; int jump, use, turn_after, brake_after /* -1 none */, blast; bool direct;
    bool through; Route next_route; P3 ngem; double leg; bool has_arrive; double arrive;
    bool passed = false; bool has_prev = false; V3 prev{0, 0, 0};
    bool has_min = false; double min_speed = 0;   // ABI 8: search.SequenceProposal.min_speed (v62)
};

// search.open_top: state['position'] plain floats (weak cell_of), the level nearest z - radius (float32 compare)
bool open_top_at(const Terrain& t, const Consts& C, V3 p, double radius, double& top) {
    if (C.open_top == 0 || t.edge.empty()) return false;
    int j, i; cell_of(t, p.x, p.y, false, false, j, i);
    int k = level_at(t, j, i, true, p.z - radius);
    if (k < 0) return false;
    double d = (double)t.edge[((size_t)k * t.H + j) * t.W + i], base = 11.;
    if (d <= C.open_near) return false;
    top = base + (C.open_top - base) * pymin(1., (d - C.open_near) / (C.open_far - C.open_near));
    return true;
}
// search.leg_top: the travel ramp on a long leg, else search.open_top
bool leg_top(const Terrain& t, const Consts& C, V3 p, double radius, double distance, double& top) {
    if (C.travel_top != 0 && distance > C.travel_near) {
        top = C.cap + (C.travel_top - C.cap) * pymin(1., (distance - C.travel_near) / (C.travel_far - C.travel_near));
        return true;
    }
    return open_top_at(t, C, p, radius, top);
}

Action seq_call(Terrain& t, const Consts& C, SeqSpec& S, V3 p, V3 v, int index, double radius = .19) {
    if (S.through) {
        if (index == 0) { S.passed = false; S.has_prev = false; }
        V3 g{S.gem.x, S.gem.y, S.gem.z};
        if (!S.passed && S.has_prev) {
            V3 a = S.prev, ab = sub(p, a);
            double tt = clip01(dot(sub(g, a), ab) / pymax(dot(ab, ab), 1e-12));
            S.passed = norm(sub(add(a, mul(ab, tt)), g)) < C.through_switch;
        } else if (!S.passed) S.passed = norm(sub(p, g)) < C.through_switch;
        S.prev = p; S.has_prev = true;
        if (S.passed) {
            bool routed; P3 goal = navigate(t, S.next_route, P3{p.x, p.y, p.z}, S.ngem, routed);
            double remaining = route_distance(t, S.next_route, P3{p.x, p.y, p.z}, true);   // p = np.asarray(position)
            double distance = std::isfinite(remaining) ? remaining : norm2(S.ngem.x - p.x, S.ngem.y - p.y);
            double top = 0.; const bool has_top = leg_top(t, C, p, radius, distance, top);
            Action a = proposal(C, goal, 0., -1, -1, false, true, distance, C.pickup_min_speed, p, v, has_top, top);
            for (int q = 0; q < 4; q++) a.a[q] = round6(a.a[q] * S.scale, (a.npmask >> q) & 1u);
            a.a[8] = 0;
            return a;
        }
    }
    P3 goal; bool routed = false;
    if (S.direct) goal = S.gem; else goal = navigate(t, S.route, P3{p.x, p.y, p.z}, S.gem, routed);
    bool has_sd = false; double sd = 0;
    if (C.route_speed_law && !S.direct) {
        double remaining = route_distance(t, S.route, P3{p.x, p.y, p.z}, false);   // state['position'] (plain floats)
        sd = std::isfinite(remaining) ? remaining : norm2(S.gem.x - p.x, S.gem.y - p.y); has_sd = true;
    }
    bool braking = S.brake_after >= 0 && index >= S.brake_after;
    double min_speed = !braking ? (S.has_min ? S.min_speed : C.pickup_min_speed) : 0.;
    const double target_distance = has_sd ? sd : norm2(S.gem.x - p.x, S.gem.y - p.y);
    if (S.through) {
        double base = has_sd ? sd : norm2(S.gem.x - p.x, S.gem.y - p.y);
        if (S.has_arrive) { sd = base + S.arrive * S.arrive / (2 * C.brake_decel); min_speed = 0.; }
        else sd = base + S.leg;
        has_sd = true;
    }
    double top = 0.; const bool has_top = leg_top(t, C, p, radius, target_distance, top);
    Action a = proposal(C, goal, index < S.turn_after ? S.offset : 0., S.jump - index, S.use - index, braking, has_sd, sd, min_speed, p, v,
                        has_top, top);
    for (int q = 0; q < 4; q++) a.a[q] = round6(a.a[q] * S.scale, (a.npmask >> q) & 1u);
    a.a[8] = index == S.blast ? 1 : 0;
    return a;
}

// ------------------------------------------------------------------ bridge (bridge.translated / quantize)
inline double quantize(double value) {
    float v = (float)value;
    if (-1 <= v && v <= 1) { float q = (float)(v + 1.0f); q = (float)(q * 16.0f); int n = (int)q; return (double)(n - 16) / 16.0; }
    float c = v < -1.f ? -1.f : (v > 1.f ? 1.f : v);
    return (double)c;
}
struct Pending { double a[9]; };
// returns next native move [x, y, yaw, jump, use, trigger1] and the camera yaw
void translated(const Pending& act, int& hold, double& pow_yaw, double move_out[6], double& camera_yaw) {
    double fwd = act.a[0], back = act.a[1], left = act.a[2], right = act.a[3], jump = act.a[4], yaw = act.a[5], use = act.a[6], aim = act.a[7];
    if (use != 0) { hold = 12; pow_yaw = f32(py_round(aim, 6)); }
    if (hold) { yaw = pow_yaw; hold -= 1; }
    yaw = f32(g6(f32(py_round(yaw, 6))));
    double script_yaw = f32(g6(yaw));
    double c = f32(std::cos(script_yaw)), s = f32(std::sin(script_yaw));
    double wx = f32(f32(py_round(right, 6)) - f32(py_round(left, 6)));
    double wy = f32(f32(py_round(fwd, 6)) - f32(py_round(back, 6)));
    double cx = g6(f32(wx * c - wy * s)), cy = g6(f32(wx * s + wy * c));
    move_out[0] = quantize(cx); move_out[1] = quantize(cy); move_out[2] = 0.; move_out[3] = jump != 0; move_out[4] = use != 0; move_out[5] = 0.;
    camera_yaw = yaw;
}

// ------------------------------------------------------------------ Blast and inventory script steps (blast.py / inventory.py)
struct BlastCtx { bool present, enabled, frozen, triple, fireball, cannon, special, charging; double value, required, special_power, power, charge_ms; };
inline double script_arg(double x) { return f32(g6(f32(x))); }
// returns 0 = no impulse, 1 = impulse in dv/push, negative = error code (7 unavailable, 8 unverified special, 10 mass)
int blast_event(const BlastCtx& b, const State& s, bool requested, double push[3], double dv[3]) {
    if (!requested) return 0;
    if (!b.present) return -7;
    if (b.triple || b.fireball || b.cannon) return -8;
    bool available = b.enabled && !b.frozen && b.value >= b.required;
    if (!available) return 0;
    double amount = b.special ? f32(b.special_power) : f32(std::sqrt(pymax(0., script_arg(b.value))));
    double scalar = script_arg(f32(amount * -f32(b.power)));
    double g[3] = {script_arg(s.gravity.x), script_arg(s.gravity.y), script_arg(s.gravity.z)};
    for (int q = 0; q < 3; q++) push[q] = script_arg(f32(scalar * g[q]));
    double mass = f32(s.prm.has_mass ? s.prm.mass : 1.0);
    if (mass <= 0) return -10;
    for (int q = 0; q < 3; q++) dv[q] = f32(f32(push[q]) / mass);
    return 1;
}
int blast_advance(BlastCtx& b, bool fired, int dt_ms) {
    if (!b.present) return 0;
    if (fired) { b.value = 0.; b.special = false; }
    else if (b.charging) {
        if (b.charge_ms <= 0) return -11;
        b.value = pymin(1., pymax(0., f32(f32(b.value) + f32((double)dt_ms / b.charge_ms))));
    }
    return 0;
}
struct InvCtx { bool present, mega_active, permanent_mega; double normal_radius; int held_type; };
int inventory_check(const InvCtx& c, const State& s, const Pending& pending) {
    if (!c.present) return 0;
    if (c.mega_active && !c.permanent_mega) return -12;
    if (c.normal_radius > 0 && !c.mega_active && std::fabs(s.radius - c.normal_radius) > 1e-5) return -13;
    if (pending.a[6] != 0 && c.held_type == 6) return -14;
    return 0;
}
