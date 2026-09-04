// ---------------------------------------------------------------------------
// Null-geodesic ray tracer in Cartesian Kerr-Schild coordinates.
//
// Direct translation of `trace_kernel_mtl!` in src/metal.jl. The chart is
// g = eta + f l(x)l, regular at the poles AND at the horizon, so there is no
// 1/sin^2(theta) and no 1/(r - 2M) anywhere below and a camera can cross r = 2M
// without the integrator noticing.
//
// Rays are traced BACKWARD from the camera. Everything is float (the Julia
// kernel is Float32 throughout); the CPU reference in src/raytrace.jl remains
// the Float64 twin.
// ---------------------------------------------------------------------------
#include <metal_stdlib>
using namespace metal;

// Compile-time specialisation, the MSL analogue of Julia's `Val{...}`
// dispatch: each combination is a separate pipeline so the Schwarzschild
// kernel never pays Kerr's register budget and the no-gas kernel stays lean.
constant bool KERR [[function_constant(0)]];
constant bool VOL  [[function_constant(1)]];
// A second Schwarzschild hole, superposed. Its centre and mass ride in
// function constants so the geodesic RHS can read them at file scope without
// threading extra parameters through every RK stage. Referenced only inside
// `if (BINARY)`, which is dead-code-eliminated (and the constants left unset)
// for the single-hole pipelines.
constant bool  BINARY [[function_constant(3)]];
constant float BH2X   [[function_constant(4)]];
constant float BH2Y   [[function_constant(5)]];
constant float BH2Z   [[function_constant(6)]];
constant float BH2M   [[function_constant(7)]];
// Integrator: 4 = classical RK4, 2 = explicit midpoint, 45 = adaptive
// Cash-Karp RK45 capped by the radius-adaptive step, 46 = the same controller
// let off that leash (which is what shows why the leash is there).
constant int  ORDER [[function_constant(2)]];

// --- parameter block layouts ------------------------------------------------
// Indices are the Julia layouts minus one. See CAM_PARAMS_N in src/metal.jl.
//
// cam[0..2]   camera position (x, y, z)
// cam[3]      fov factor
// cam[4..7]   E_forward   (contravariant 4-vector, t x y z)
// cam[8..11]  E_right
// cam[12..15] E_up
// cam[16..19] observer 4-velocity u
// cam[20]     projection: 0 = pinhole, 1 = equidistant fisheye
// cam[21]     fisheye half-angle (radians)
//
// st[0] M          st[1] r_band     st[2] r_escape   st[3] relativistic flag
// st[4] hstep coef st[5] hstep cap  st[6] spin a     st[7] capture radius^2
//
// disc[0] inner  disc[1] outer  disc[2] falloff
// disc[3] LUT Tmin  disc[4] LUT Tmax  disc[5] LUT size
//
// vol[0] log s_in  vol[1] log s_out  vol[2] z_max
// vol[3] nr  vol[4] nphi  vol[5] nz
// vol[6] emission scale  vol[7] opacity scale  vol[8] gas arc-length stride
//
// star[0] strength   star[1] grid N     star[2] fill      star[3] sigma
// star[4] flux0      star[5..7] galactic normal            star[8] concentration
// star[9] Tmin       star[10] Tspan     star[11] LUT Tmin  star[12] LUT Tmax
// star[13] LUT size  star[14] seed      star[15] texture weight

// Floor-modulus. Julia's `mod` floors; MSL's `fmod` truncates, so a negative
// phi would index off the front of the sky without this.
static inline float modf_pos(float a, float b) {
    float m = fmod(a, b);
    return m < 0.0f ? m + b : m;
}
static inline int mod_pos(int a, int b) {
    int m = a % b;
    return m < 0 ? m + b : m;
}

// Hash -> [0,1). Bit-identical to `_sim_hash` in src/disc_sim.jl, so the
// procedural starfield renders the same sky as the Julia build.
static inline float sim_hash(int x, int y, int z) {
    uint h = (uint)x * 0x8da6b343u ^ (uint)y * 0xd8163841u ^ (uint)z * 0xcb1ab31fu;
    h ^= (h >> 13);
    h *= 0x9e3779b9u;
    h ^= (h >> 16);
    return (float)(h & 0x00ffffffu) * 5.9604645e-8f;
}

// ---------------------------------------------------------------------------
// Geodesic right-hand sides
// ---------------------------------------------------------------------------

// Schwarzschild. H = 1/2(-p_t^2 + |p|^2 - f l^2), f = 2M/r, l = -p_t + (x.p)/r.
// No trigonometry, no coordinate singularity. Returns (dx, dp).
// One hole's contribution to the Kerr-Schild Hamiltonian RHS, as the DEVIATION
// from flat space (so a superposition of holes is just the sum of these plus
// the single flat `+p` in dx). `xr` is the position relative to that hole.
static inline void ks_terms(float3 xr, float3 p, float p_t, float M,
                            thread float3 &ddx, thread float3 &ddp) {
    float r2 = dot(xr, xr);
    float inv_r = rsqrt(r2);
    float f = 2.0f * M * inv_r;
    float kap = dot(xr, p) * inv_r;
    float l = -p_t + kap;
    float c1 = f * l * inv_r;
    float c2 = f * l * (0.5f * l + kap) * inv_r * inv_r;
    ddx = -c1 * xr;
    ddp = c1 * p - c2 * xr;
}

static inline void ks_rhs(float3 x, float3 p, float p_t, float M,
                          thread float3 &dx, thread float3 &dp) {
    float3 ddx, ddp;
    ks_terms(x, p, p_t, M, ddx, ddp);
    dx = p + ddx;
    dp = ddp;
}

// Kerr, same chart and Hamiltonian convention; reduces to ks_rhs exactly at
// a = 0. r is an implicit function of position (the quartic
// r^4 - (rho^2 - a^2) r^2 - a^2 z^2 = 0), so the position derivatives route
// through dr/dx^i -- four derivative expressions instead of the nine partials
// a direct expansion of dL_j/dx^i would need.
static inline void kerr_rhs(float3 x, float3 p, float p_t, float M, float a,
                            thread float3 &dx, thread float3 &dp) {
    float a2 = a * a;
    float z2 = x.z * x.z;
    float w = dot(x, x) - a2;
    float r2 = 0.5f * (w + sqrt(w * w + 4.0f * a2 * z2));
    r2 = max(r2, 1.0e-12f);
    float r = sqrt(r2);
    float r3 = r2 * r;
    float invS = 1.0f / (r2 * r2 + a2 * z2);
    float R2A = r2 + a2;
    float iRA = 1.0f / R2A;
    float inv_r = 1.0f / r;

    float3 L = float3((r * x.x + a * x.y) * iRA,
                      (r * x.y - a * x.x) * iRA,
                      x.z * inv_r);
    float l = -p_t + dot(L, p);
    float f = 2.0f * M * r3 * invS;
    float fl = f * l;

    float3 dr = float3(x.x * r3 * invS, x.y * r3 * invS, x.z * r * R2A * invS);

    float dfdr = 2.0f * M * r2 * (3.0f * a2 * z2 - r2 * r2) * invS * invS;
    float dfz0 = -4.0f * M * r3 * a2 * x.z * invS * invS;

    float3 dl0 = float3((r * p.x - a * p.y) * iRA,
                        (a * p.x + r * p.y) * iRA,
                        p.z * inv_r);
    float dldr = (p.x * (x.x * R2A - 2.0f * r * (r * x.x + a * x.y)) +
                  p.y * (x.y * R2A - 2.0f * r * (r * x.y - a * x.x))) * iRA * iRA -
                 x.z * p.z * inv_r * inv_r;

    float hl2 = 0.5f * l * l;
    dx = p - fl * L;
    dp = hl2 * (dfdr * dr) + fl * (dldr * dr + dl0);
    dp.z += hl2 * dfz0;
}

static inline void rhs(float3 x, float3 p, float p_t, float M, float a,
                       thread float3 &dx, thread float3 &dp) {
    if (KERR) { kerr_rhs(x, p, p_t, M, a, dx, dp); return; }
    float3 ddx, ddp;
    ks_terms(x, p, p_t, M, ddx, ddp);
    dx = p + ddx;
    dp = ddp;
    if (BINARY) {
        float3 ddx2, ddp2;
        ks_terms(x - float3(BH2X, BH2Y, BH2Z), p, p_t, BH2M, ddx2, ddp2);
        dx += ddx2;
        dp += ddp2;
    }
}

// ---------------------------------------------------------------------------
// Samplers
// ---------------------------------------------------------------------------

// Bilinear sample of the equirectangular sky. `bg` is the Julia (3, W, H)
// array flattened column-major, so channel is the fastest axis.
static inline float3 sample_background(const device float *bg,
                                       float theta, float phi, int W, int H) {
    const float two_pi = 6.28318531f;
    float v = clamp(theta * (1.0f / 3.14159265f), 0.0f, 1.0f);
    float u = clamp(modf_pos(phi, two_pi) / two_pi, 0.0f, 1.0f);

    float xf = u * (float)(W - 1);
    float yf = v * (float)(H - 1);
    int x0 = clamp((int)floor(xf), 0, W - 1);
    int x1 = min(x0 + 1, W - 1);
    int y0 = clamp((int)floor(yf), 0, H - 1);
    int y1 = min(y0 + 1, H - 1);
    float fx = xf - (float)x0;
    float fy = yf - (float)y0;

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    int i00 = 3 * (x0 + W * y0), i10 = 3 * (x1 + W * y0);
    int i01 = 3 * (x0 + W * y1), i11 = 3 * (x1 + W * y1);
    return float3(bg[i00 + 0], bg[i00 + 1], bg[i00 + 2]) * w00 +
           float3(bg[i10 + 0], bg[i10 + 1], bg[i10 + 2]) * w10 +
           float3(bg[i01 + 0], bg[i01 + 1], bg[i01 + 2]) * w01 +
           float3(bg[i11 + 0], bg[i11 + 1], bg[i11 + 2]) * w11;
}

// Trilinear density lookup on the (nr, nphi, nz) log-r x phi x z grid.
// Periodic in phi, clamped in r and z, zero outside.
static inline float sample_volume(const device float *vol,
                                  const device float *vp,
                                  float s, float phi, float z) {
    float zmax = vp[2];
    if (s <= 0.0f || fabs(z) >= zmax) return 0.0f;
    float ls = log(s);
    float ls_in = vp[0], ls_out = vp[1];
    if (ls <= ls_in || ls >= ls_out) return 0.0f;

    int nr = (int)vp[3], nphi = (int)vp[4], nz = (int)vp[5];
    const float two_pi = 6.28318531f;

    // Rotate the frozen gas pattern prograde by sampling at phi - Omega*t.
    // Fully Keplerian shear (Omega ~ 1/s^1.5, inner ~15x the outer rate) is
    // physically right but winds a *snapshot* texture into tight concentric
    // streaks without bound -- fine for a short film, wrong for a viewer that
    // holds on one frame. So rotate mostly coherently (at a reference radius)
    // with only a fraction DIFF of the true local differential: the inner disc
    // still visibly leads, but the shear -- and thus the winding -- is bounded
    // to a slow drift. The relativistic beaming in disc_emission is unaffected;
    // it reads the real Keplerian velocity independently.
    const float DIFF = 0.4f;         // 0 = solid body, 1 = full Keplerian
    const float S_REF = 7.0f;        // coherent-rotation reference radius
    float t_disc = vp[9];
    float phi_s = phi;
    if (t_disc != 0.0f) {
        float kep     = 1.0f / (pow(s, 1.5f) + vp[10]);
        float kep_ref = 1.0f / (pow(S_REF, 1.5f) + vp[10]);
        float omega = t_disc * (DIFF * kep + (1.0f - DIFF) * kep_ref);
        phi_s = phi - omega;
    }

    float fr = (ls - ls_in) / (ls_out - ls_in) * (float)(nr - 1);
    float fp = modf_pos(phi_s, two_pi) / two_pi * (float)nphi;
    float fz = (z + zmax) / (2.0f * zmax) * (float)(nz - 1);

    int i0 = clamp((int)floor(fr), 0, nr - 2);
    int j0 = (int)floor(fp);
    int k0 = clamp((int)floor(fz), 0, nz - 2);
    float tr = fr - (float)i0;
    float tp = fp - (float)j0;
    float tz = fz - (float)k0;

    int ja = mod_pos(j0, nphi);
    int jb = mod_pos(j0 + 1, nphi);
    int s1 = nr, s2 = nr * nphi;

    float c00 = mix(vol[i0 + s1 * ja + s2 * k0],       vol[i0 + 1 + s1 * ja + s2 * k0], tr);
    float c10 = mix(vol[i0 + s1 * jb + s2 * k0],       vol[i0 + 1 + s1 * jb + s2 * k0], tr);
    float c01 = mix(vol[i0 + s1 * ja + s2 * (k0 + 1)], vol[i0 + 1 + s1 * ja + s2 * (k0 + 1)], tr);
    float c11 = mix(vol[i0 + s1 * jb + s2 * (k0 + 1)], vol[i0 + 1 + s1 * jb + s2 * (k0 + 1)], tr);
    return mix(mix(c00, c10, tp), mix(c01, c11, tp), tz);
}

// Procedural point stars on an equal-area (phi/2pi, (cos t + 1)/2) grid, so
// density is uniform on the sky with no pole pile-up and the 3x3 cell scan
// needs no latitude correction. Evaluated in the SOURCE sky, so lensing
// magnifies and brightens star images near the critical curve for free.
static inline float3 starfield(float3 d, const device float *sp,
                               const device float *lut) {
    float N = sp[1];
    float fill = sp[2];
    float sigma = sp[3];
    float flux0 = sp[4];
    float3 gn = float3(sp[5], sp[6], sp[7]);
    float gconc = sp[8];
    float tmin = sp[9], tspan = sp[10];
    float lut_tmin = sp[11], lut_tmax = sp[12], lut_size = sp[13];
    int seed = (int)sp[14];

    const float two_pi = 6.28318531f;
    float u = atan2(d.y, d.x) / two_pi + 0.5f;
    float v = (d.z + 1.0f) * 0.5f;
    int Ni = (int)N;
    int i0 = (int)floor(u * N);
    int j0 = (int)floor(v * N);

    float inv2s2 = 1.0f / (2.0f * sigma * sigma);
    float cut = 25.0f * sigma * sigma;   // 5 sigma
    float3 acc = float3(0.0f);

    for (int dj = -1; dj <= 1; ++dj) {
        int jj = j0 + dj;
        if (jj < 0 || jj >= Ni) continue;
        for (int di = -1; di <= 1; ++di) {
            int ii = mod_pos(i0 + di, Ni);

            float su = sim_hash(ii, jj, seed + 1);
            float sv = sim_hash(ii, jj, seed + 2);
            float phis = (((float)ii + su) / N - 0.5f) * two_pi;
            float sz = 2.0f * ((float)jj + sv) / N - 1.0f;
            float sr = sqrt(max(1.0f - sz * sz, 0.0f));
            float3 sdir = float3(sr * cos(phis), sr * sin(phis), sz);

            float3 e = d - sdir;
            float d2 = dot(e, e);
            if (d2 > cut) continue;

            // Occupancy thinned away from the galactic plane, so the field
            // has a Milky Way concentration rather than uniform noise.
            float occ = fill;
            if (gconc > 0.0f) occ *= exp(-fabs(dot(sdir, gn)) * gconc);
            if (sim_hash(ii, jj, seed) >= occ) continue;

            // Counts grow as 10^(0.6m), flux falls as 10^(-0.4m); composed,
            // that is a flux drawn as xi^(-2/3) for uniform xi.
            float xi = max(sim_hash(ii, jj, seed + 3), 1.0e-4f);
            float flux = flux0 * exp(-0.6666667f * log(xi));
            float w = flux * exp(-d2 * inv2s2);

            float T = tmin + tspan * sim_hash(ii, jj, seed + 4);
            float frac = (clamp(T, lut_tmin, lut_tmax) - lut_tmin) /
                         max(lut_tmax - lut_tmin, 1.0e-6f);
            int li = clamp((int)(frac * (lut_size - 1.0f) + 0.5f), 0, (int)lut_size - 1);
            acc += w * float3(lut[3 * li + 0], lut[3 * li + 1], lut[3 * li + 2]);
        }
    }
    return acc;
}

// Doppler- and gravitationally-shifted Planck emission for gas orbiting at
// the local Keplerian rate. Shared by the volumetric march and the thin-plane
// crossing so the two disc models cannot drift apart.
static inline float3 disc_emission(float s_cyl, float r, float3 vdir,
                                   float2 xy, float M, float scam,
                                   const device float *dp,
                                   const device float *bb_lut,
                                   thread float &out_inten) {
    float R = s_cyl / (2.0f * M);
    float T_emit = exp(10.034259f - 0.375f * log(max(R * R, 1.0e-6f)));
    float v_mag = clamp(0.70710678f / sqrt(max(R - 1.0f, 0.1f)), 0.0f, 0.999f);
    float vlen = max(length(vdir), 1.0e-20f);
    float vdotn = v_mag * (-xy.y * vdir.x + xy.x * vdir.y) / (s_cyl * vlen);
    float gam = 1.0f / sqrt(1.0f - clamp(v_mag * v_mag, 0.0f, 0.99f));
    float Rs = r / (2.0f * M);
    float opzg = 1.0f / sqrt(max(1.0f - 1.0f / max(Rs, 1.0f), 0.01f));
    float opz = max(gam * (1.0f + vdotn) * opzg, 0.1f);
    float T_obs = T_emit * scam / opz;
    out_inten = 100.0f / (exp(29622.4f / max(T_obs, 1.0f)) - 1.0f);

    float lut_tmin = dp[3], lut_tmax = dp[4], lut_size = dp[5];
    float frac = (clamp(T_obs, lut_tmin, lut_tmax) - lut_tmin) / (lut_tmax - lut_tmin);
    int li = clamp((int)(frac * (lut_size - 1.0f) + 0.5f), 0, (int)lut_size - 1);
    return float3(bb_lut[3 * li + 0], bb_lut[3 * li + 1], bb_lut[3 * li + 2]);
}

// ---------------------------------------------------------------------------
// The kernel: one thread per pixel
// ---------------------------------------------------------------------------
kernel void trace(device float          *out       [[buffer(0)]],
                  const device float    *bg        [[buffer(1)]],
                  const device float    *bb_lut    [[buffer(2)]],
                  const device float    *star_lut  [[buffer(3)]],
                  const device float    *vol       [[buffer(4)]],
                  const device float    *vp        [[buffer(5)]],
                  const device float    *sp        [[buffer(6)]],
                  const device float    *cam       [[buffer(7)]],
                  const device float    *st        [[buffer(8)]],
                  const device float    *dp        [[buffer(9)]],
                  constant uint4        &dims      [[buffer(10)]],  // W H bgW bgH
                  constant uint2        &limits    [[buffer(11)]],  // nmax, unused
                  constant float2       &steps     [[buffer(12)]],  // dt, unused
                  uint gid [[thread_position_in_grid]])
{
    int W = (int)dims.x, H = (int)dims.y;
    int bgW = (int)dims.z, bgH = (int)dims.w;
    if (gid >= (uint)(W * H)) return;
    int i0 = (int)gid % W;
    int j0 = (int)gid / W;

    int nmax = (int)limits.x;
    float dt = steps.x;
    float tol = steps.y;

    float M        = st[0];
    float r_escape = st[2];
    float spin_a   = st[6];
    float rkill2   = st[7];

    float3 c   = float3(cam[0], cam[1], cam[2]);
    float  fov = cam[3];
    float4 Ef  = float4(cam[4],  cam[5],  cam[6],  cam[7]);
    float4 Er  = float4(cam[8],  cam[9],  cam[10], cam[11]);
    float4 Eu  = float4(cam[12], cam[13], cam[14], cam[15]);
    float4 ut  = float4(cam[16], cam[17], cam[18], cam[19]);

    // Sensor coordinate. Pixel centres (the Julia kernel's jitter is 0.5 for
    // the single-sample preview path this replaces).
    float half_h = (float)H * 0.5f;
    float u = ((float)i0 + 0.5f - (float)W * 0.5f) / half_h;
    float v = ((float)j0 + 0.5f - (float)H * 0.5f) / half_h;

    // Pixel direction as unit coefficients on the camera tetrad axes.
    float cr, cu, cf;
    if (cam[20] > 0.5f) {
        // Equidistant fisheye: pixel radius proportional to view angle, so
        // fields wider than 180 degrees render cleanly (a rectilinear pinhole
        // cannot reach 180 at any focal length).
        float rho = sqrt(u * u + v * v);
        float th = rho * cam[21];
        float sth = sin(th);
        float inv_rho = rho > 1.0e-8f ? 1.0f / rho : 0.0f;
        cr = sth * u * inv_rho;
        cu = sth * v * inv_rho;
        cf = cos(th);
    } else {
        float dxl = u * fov, dyl = v * fov;
        float nu = sqrt(dxl * dxl + dyl * dyl + 1.0f);
        cr = dxl / nu; cu = dyl / nu; cf = 1.0f / nu;
    }

    // Received photon p = w(u + n); trace q = n - u, i.e. backward in time,
    // which is what lets a ray legally exit the horizon when the camera is
    // inside it. Then lower the index: p_mu = eta_mu_nu q^nu + f l_mu (l.q).
    // In Kerr the metric is g = eta + f l(x)l with the KS null covector l_mu
    // and KS scalar f; using the Schwarzschild f = 2M/r and l = xhat here
    // mis-initialises every ray's energy and momentum off-equator.
    float3 x = c;
    float r = length(x);
    float4 q = cf * Ef + cr * Er + cu * Eu - ut;
    float p_t, lq;
    float3 p;
    if (KERR) {
        float a2 = spin_a * spin_a;
        float w0 = r * r - a2;
        float rk2 = 0.5f * (w0 + sqrt(w0 * w0 + 4.0f * a2 * x.z * x.z));
        float rk = sqrt(max(rk2, 1.0e-12f));
        float iRA = 1.0f / (rk2 + a2);
        float3 L = float3((rk * x.x + spin_a * x.y) * iRA,
                          (rk * x.y - spin_a * x.x) * iRA,
                          x.z / rk);
        float fk = 2.0f * M * rk2 * rk / (rk2 * rk2 + a2 * x.z * x.z);
        lq = q.x + dot(L, q.yzw);
        float flq = fk * lq;
        p_t = -q.x + flq;
        p = q.yzw + flq * L;
    } else {
        float f = 2.0f * M / r;
        lq = q.x + dot(x, q.yzw) / r;
        p_t = -q.x + f * lq;
        p = q.yzw + (f * lq / r) * x;
        if (BINARY) {
            // Sum the second hole's index-lowering. The cross term with hole 1
            // is O(f1 f2) and negligible at the camera, where both f are tiny.
            float3 xr = x - float3(BH2X, BH2Y, BH2Z);
            float rb = length(xr);
            float f2 = 2.0f * BH2M / rb;
            float lq2 = q.x + dot(xr, q.yzw) / rb;
            p_t += f2 * lq2;
            p += (f2 * lq2 / rb) * xr;
        }
    }

    // Relativistic shading: each ray is normalised to unit frequency in the
    // camera tetrad and p_t is conserved, so the camera/infinity shift factor
    // is just g = 1/|p_t|. Exactly 1 when the option is off.
    float scam = st[3] > 0.5f ? 1.0f / clamp(fabs(p_t), 0.05f, 20.0f) : 1.0f;

    float disc_inner = dp[0], disc_outer = dp[1], disc_falloff = dp[2];
    bool disc_enabled = disc_inner > 0.0f && disc_outer > disc_inner;
    bool disc_plane = disc_enabled && !VOL;

    float vol_zmax = vp[2], vol_emis = vp[6], vol_opac = vp[7];
    // The gas is marched on its OWN arc-length stride rather than every Nth
    // integration step. Riding the integrator's schedule was fine while that
    // schedule was fixed, but an adaptive controller bounds the ODE's local
    // truncation error and knows nothing about the gas -- so it resamples the
    // volume at wildly uneven arc lengths and the alpha compositing goes wrong
    // even though the geodesic is more accurate than before.
    float ds_gas = max(vp[8], 1.0e-3f);
    float vol_s_out = exp(vp[1]);
    float vol_rb2 = vol_s_out * vol_s_out + vol_zmax * vol_zmax;

    // Accumulated emission and remaining transmittance. Rays are NOT
    // terminated at a disc crossing, so lensed secondary and higher-order
    // images composite correctly on top of the primary.
    float3 acc = float3(0.0f);
    float alpha = 1.0f;

    bool hit_horizon = false;
    // Camera outside the horizon: no legal ray is ever below 2M (a dip is
    // horizon-ridge overshoot). Camera inside: rays exit through the band and
    // the floor is only a near-singularity safety net.
    float r_floor = r > 2.05f * M ? 2.0f * M : 0.3f * M;
    float r_prev = -1.0f;
    // RK45 state: the step size carried between iterations.
    float h_carry = -1.0f;
    // Arc length travelled since the last gas sample.
    float s_accum = 0.0f;

    for (int stepi = 1; stepi <= nmax; ++stepi) {
        float r2 = dot(x, x);
        r = sqrt(r2);

        if (KERR) {
            // rho != r in Kerr-Schild, so the capture test is written in KS r.
            // The threshold is the prograde photon orbit, not the horizon:
            // nothing reaching infinity dips below it, and testing the horizon
            // alone let near-critical rays bounce off the coordinate ridge and
            // escape as phantom sky inside the shadow.
            float aw = r2 - spin_a * spin_a;
            float rk2 = 0.5f * (aw + sqrt(aw * aw + 4.0f * spin_a * spin_a * x.z * x.z));
            if (rk2 < rkill2) { hit_horizon = true; break; }
        } else if (r < 3.2f * M) {
            // Exact criterion: an escaping null geodesic never has a turning
            // point below the photon sphere, so a ray moving inward below
            // ~2.95M can never legally return -- it is the shadow.
            if (r < r_floor ||
                (r < 2.95f * M && r_prev > 0.0f && r < r_prev - 1.0e-4f * M)) {
                hit_horizon = true; break;
            }
            r_prev = r;
        }
        // Second hole's shadow: a plain radius test inside ~2.95 M2.
        if (BINARY && length(x - float3(BH2X, BH2Y, BH2Z)) < 2.95f * BH2M) {
            hit_horizon = true; break;
        }
        if (r > r_escape) break;

        // Radius-adaptive affine step: curvature goes as M/r^3, so scaling h
        // with r holds the per-step bending error uniform while collapsing the
        // nearly-flat travel legs. Capped at 2x inside the gas bounding
        // sphere, which contains the strong field where the shadow-kill tests
        // need small steps.
        float hcap = (VOL && r2 < vol_rb2) ? 2.0f : st[5];
        // Step adapts to the NEAREST hole so hole 2's strong field is resolved.
        float rstep = r;
        if (BINARY) rstep = min(rstep, length(x - float3(BH2X, BH2Y, BH2Z)));
        float h = dt * min(max(st[4] * rstep / M, 1.0f), hcap);
        if (KERR) {
            // Capture radius and horizon converge as a -> M (1.074M against
            // 1.063M at a = 0.998) and a step floored at dt cannot resolve
            // that gap. Shrink as the ray closes on capture; far away it is a
            // no-op.
            float aw = r2 - spin_a * spin_a;
            float rk2 = 0.5f * (aw + sqrt(aw * aw + 4.0f * spin_a * spin_a * x.z * x.z));
            float fr = clamp((rk2 - rkill2) / max(rkill2, 1.0e-6f), 0.0f, 1.0f);
            h *= 0.12f + 0.88f * fr;
        }

        float3 xp = x, pp = p;

        float3 k1x, k1p;
        rhs(x, p, p_t, M, spin_a, k1x, k1p);

        // `h_used` is the step actually taken. For the fixed-order paths that
        // is the radius-adaptive h above; for RK45 the error controller may
        // shrink it, and the gas sampler needs whichever was used.
        float h_used = h;

        if (ORDER == 45 || ORDER == 46) {
            // Adaptive Cash-Karp RK45: six stages, an embedded fourth-order
            // solution for the error estimate, per-lane step control.
            //
            // The step is capped at the radius-adaptive h rather than allowed
            // to run free. That cap is not about accuracy -- it is what the
            // shadow-kill tests need: they are sampled ONCE per step, so a
            // long step near the photon sphere jumps the band the test looks
            // at and a captured ray escapes as phantom sky.
            float hmax = (ORDER == 46) ? 50.0f * dt : h;
            // The gas does NOT constrain the step. It needs samples every
            // ds_gas of arc length, which is not the same as steps every
            // ds_gas -- a long step is sub-sampled below instead. Bounding the
            // step here was measurably the wrong call: it cost the whole
            // adaptive speedup and bought nothing, because the sampling itself
            // is cheap (a 16x stride sweep moves the frame by ~10%).
            float hh = min(h_carry > 0.0f ? h_carry : h, hmax);
            float3 nx = x, np = p;
            bool accepted = false;

            for (int att = 0; att < 4 && !accepted; ++att) {
                float3 k2x, k2p, k3x, k3p, k4x, k4p, k5x, k5p, k6x, k6p;
                rhs(x + hh*(0.2f*k1x),
                    p + hh*(0.2f*k1p), p_t, M, spin_a, k2x, k2p);
                rhs(x + hh*(0.075f*k1x + 0.225f*k2x),
                    p + hh*(0.075f*k1p + 0.225f*k2p), p_t, M, spin_a, k3x, k3p);
                rhs(x + hh*(0.3f*k1x - 0.9f*k2x + 1.2f*k3x),
                    p + hh*(0.3f*k1p - 0.9f*k2p + 1.2f*k3p), p_t, M, spin_a, k4x, k4p);
                rhs(x + hh*(-0.2037037037f*k1x + 2.5f*k2x - 2.5925925926f*k3x + 1.2962962963f*k4x),
                    p + hh*(-0.2037037037f*k1p + 2.5f*k2p - 2.5925925926f*k3p + 1.2962962963f*k4p),
                    p_t, M, spin_a, k5x, k5p);
                rhs(x + hh*(0.0294958043f*k1x + 0.341796875f*k2x + 0.0415943148f*k3x
                            + 0.4003454071f*k4x + 0.061767578125f*k5x),
                    p + hh*(0.0294958043f*k1p + 0.341796875f*k2p + 0.0415943148f*k3p
                            + 0.4003454071f*k4p + 0.061767578125f*k5p),
                    p_t, M, spin_a, k6x, k6p);

                // Fifth order, and the embedded fourth order it is compared to.
                float3 d5x = 0.0978835979f*k1x + 0.4025764895f*k3x + 0.2104377104f*k4x
                           + 0.2891022021f*k6x;
                float3 d5p = 0.0978835979f*k1p + 0.4025764895f*k3p + 0.2104377104f*k4p
                           + 0.2891022021f*k6p;
                float3 d4x = 0.1021773726f*k1x + 0.3839079034f*k3x + 0.2445927372f*k4x
                           + 0.01932198661f*k5x + 0.25f*k6x;
                float3 d4p = 0.1021773726f*k1p + 0.3839079034f*k3p + 0.2445927372f*k4p
                           + 0.01932198661f*k5p + 0.25f*k6p;

                nx = x + hh * d5x;
                np = p + hh * d5p;

                float3 ex = hh * (d5x - d4x);
                float3 ep = hh * (d5p - d4p);
                // Mixed absolute/relative scaling, so a ray far from the hole
                // is not held to the same absolute error as one at periapsis.
                float sc = tol * (1.0f + max(length(x), length(p)));
                float err = max(length(ex), length(ep)) / max(sc, 1.0e-30f);

                if (err <= 1.0f || hh <= 1.0e-4f * dt) {
                    accepted = true;
                    h_used = hh;
                    // Grow for the next step, but never past the kill-test cap.
                    float g = err > 1.0e-12f ? 0.9f * pow(err, -0.2f) : 4.0f;
                    h_carry = min(hh * clamp(g, 0.2f, 4.0f), hmax);
                } else {
                    hh = max(hh * max(0.9f * pow(err, -0.25f), 0.2f), 1.0e-4f * dt);
                }
                // A rejected attempt is work the GPU did and threw away;
                // the `att` bound is what keeps that bounded per step.
            }
            x = nx;
            p = np;
        } else if (ORDER == 2) {
            float3 k2x, k2p;
            rhs(x + 0.5f * h * k1x, p + 0.5f * h * k1p, p_t, M, spin_a, k2x, k2p);
            x += h * k2x;
            p += h * k2p;
        } else {
            // Classical RK4. k1 is already in hand (the gas sampler needs the
            // photon's coordinate velocity), so this costs three more
            // evaluations.
            float3 k2x, k2p, k3x, k3p, k4x, k4p;
            rhs(x + 0.5f * h * k1x, p + 0.5f * h * k1p, p_t, M, spin_a, k2x, k2p);
            rhs(x + 0.5f * h * k2x, p + 0.5f * h * k2p, p_t, M, spin_a, k3x, k3p);
            rhs(x + h * k3x,        p + h * k3p,        p_t, M, spin_a, k4x, k4p);
            x += (h / 6.0f) * (k1x + 2.0f * k2x + 2.0f * k3x + k4x);
            p += (h / 6.0f) * (k1p + 2.0f * k2p + 2.0f * k3p + k4p);
        }

        // Volumetric gas: sampled every vol_mstep-th step with vol_mstep x the
        // path weight -- the gas structure is far coarser than the integration
        // step, so this is free detail. Geodesics are unaffected. Evaluated at
        // the PRE-step position, as in the Julia kernel.
        if (VOL && alpha > 0.003f && r2 < vol_rb2) {
            // Emit one gas sample per ds_gas of arc length, SUB-SAMPLING the
            // step when it is long. The geodesic over a single step is very
            // nearly straight -- far straighter than the gas is smooth -- so
            // linear interpolation between the step's endpoints places the
            // samples well, and the emission quadrature ends up independent of
            // whatever schedule the integrator chose.
            // Integrate along the CHORD the step actually produced, not
            // along the tangent at its start: with long steps the two differ,
            // and using the start tangent both mis-measures the optical depth
            // and Doppler-shifts every sub-sample as if the photon were still
            // travelling in its initial direction.
            float3 dseg = x - xp;
            float seg = max(length(dseg), 1.0e-20f);
            s_accum += seg;
            // Bounded: one enormous vacuum step must not spin here.
            for (int sub = 0; sub < 32 && s_accum >= ds_gas; ++sub) {
                s_accum -= ds_gas;
                // Distance back from the step's end, as a fraction of it.
                float t = clamp(1.0f - s_accum / seg, 0.0f, 1.0f);
                float3 xs = mix(xp, x, t);
                float rs2 = dot(xs, xs);
                if (rs2 >= vol_rb2 || fabs(xs.z) >= vol_zmax) continue;
                float s_cyl = sqrt(xs.x * xs.x + xs.y * xs.y);
                if (s_cyl <= 1.0e-6f) continue;
                float rho = sample_volume(vol, vp, s_cyl, atan2(xs.y, xs.x), xs.z);
                if (rho <= 1.0e-4f) continue;
                float inten;
                float3 col = disc_emission(s_cyl, sqrt(rs2), dseg, xs.xy, M, scam,
                                           dp, bb_lut, inten);
                float tau = vol_opac * rho * ds_gas;
                float a = 1.0f - exp(-tau);
                acc += alpha * a * inten * vol_emis * col;
                alpha *= (1.0f - a);
                if (alpha < 0.003f) break;   // transmittance exhausted
            }
            if (alpha < 0.003f) break;
        }

        // A non-finite ray can never satisfy the exit tests and would reach
        // the sky sampler as NaN, trapping the thread in the loop.
        if (!isfinite(x.x) || !isfinite(x.z) || !isfinite(p.x)) {
            hit_horizon = true; break;
        }

        // Thin-plane disc: equatorial crossing located by linear interpolation
        // across the step.
        if (disc_plane && alpha > 0.001f && xp.z * x.z < 0.0f) {
            float cfr = xp.z / (xp.z - x.z);
            float3 xh = xp + cfr * (x - xp);
            float s = length(xh.xy);
            if (disc_inner < s && s < disc_outer) {
                float3 ph = pp + cfr * (p - pp);
                float fh = 2.0f * M / s;
                float kh = dot(xh.xy, ph.xy) / s;
                float lh = -p_t + kh;
                float c1h = fh * lh / s;
                float3 vh = float3(ph.x - c1h * xh.x, ph.y - c1h * xh.y, ph.z);

                float inten;
                float3 col = disc_emission(s, s, vh, xh.xy, M, scam, dp, bb_lut, inten);

                float R = s / (2.0f * M);
                float R_in = disc_inner / (2.0f * M);
                float R_out = disc_outer / (2.0f * M);
                float T_emit = exp(10.034259f - 0.375f * log(R * R));
                float iscotaper = clamp((R * R - R_in * R_in) * 0.3f, 0.0f, 1.0f);
                float outertaper = clamp(T_emit / 1000.0f, 0.0f, 1.0f);
                float density = clamp((R_out - R) / (R_out - R_in), 0.0f, 1.0f);
                float dpow = density <= 1.0e-6f ? (disc_falloff > 0.0f ? 0.0f : 1.0f)
                                                : exp(disc_falloff * log(density));
                float opacity = iscotaper * outertaper * dpow;

                acc += alpha * opacity * inten * col;
                alpha *= (1.0f - opacity);
            }
        }
    }

    float3 rgb;
    // Rays that ran out of steps while still deep in the strong field are
    // near-critical or horizon-hugging: black, like the captured ones.
    float rf = max(length(x), 1.0e-6f);
    if (hit_horizon || rf < 4.0f * M) {
        rgb = acc;
    } else {
        // Sample the sky by the ASYMPTOTIC MOMENTUM direction, not the escape
        // position: position sampling parallax-shifts stars by up to
        // b/r_escape radians for disc-grazing rays. The photon's coordinate
        // velocity is the integrator's own dx/dlambda, correct for Kerr too.
        float3 vf, unused;
        rhs(x, p, p_t, M, spin_a, vf, unused);
        float vl = max(length(vf), 1.0e-20f);
        float3 d = vf / vl;
        float3 sky = sample_background(bg, acos(clamp(d.z, -1.0f, 1.0f)),
                                       atan2(d.y, d.x), bgW, bgH);
        if (sp[0] > 0.0f) {
            sky *= sp[15];                       // texture weight (crossfade)
            sky += sp[0] * starfield(d, sp, star_lut);
        }
        if (scam != 1.0f) {
            // A ~5800 K star observed at T = g * 5800 K: per-channel Planck
            // ratios at 610/550/465 nm. The g^4 bolometric beaming falls out
            // of the same formula.
            sky.r *= 57.4f  / (exp(4.067f / scam) - 1.0f);
            sky.g *= 90.2f  / (exp(4.513f / scam) - 1.0f);
            sky.b *= 206.5f / (exp(5.335f / scam) - 1.0f);
        }
        rgb = acc + alpha * sky;
    }

    int o = 3 * (i0 + W * j0);
    out[o + 0] = rgb.r;
    out[o + 1] = rgb.g;
    out[o + 2] = rgb.b;
}
