//! Accretion disc: blackbody colour, and the volumetric density grid.
//!
//! Port of src/blackbody.jl and the host-side bake in src/disc_volume.jl. The
//! grid is built once at startup and uploaded; the GPU only ever samples it.

use std::thread;

// ---------------------------------------------------------------------------
// Blackbody colour
// ---------------------------------------------------------------------------

fn cie_gauss(l: f64, mu: f64, s1: f64, s2: f64) -> f64 {
    let s = if l < mu { s1 } else { s2 };
    (-0.5 * ((l - mu) / s).powi(2)).exp()
}

/// Linear sRGB of a blackbody at `t` kelvin, normalised to peak 1. Planck's
/// law integrated against the CIE colour matching functions (the multi-lobe
/// Gaussian fit), then XYZ -> sRGB.
pub fn blackbody_rgb(t: f64) -> [f64; 3] {
    let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
    let mut l = 380.0;
    while l <= 780.0 {
        let lm: f64 = l * 1e-9;
        let b = 1.0 / (lm.powi(5) * ((0.014388 / (lm * t)).exp() - 1.0));
        let xb = 1.056 * cie_gauss(l, 599.8, 37.9, 31.0)
               + 0.362 * cie_gauss(l, 442.0, 16.0, 26.7)
               - 0.065 * cie_gauss(l, 501.1, 20.4, 26.2);
        let yb = 0.821 * cie_gauss(l, 568.8, 46.9, 40.5)
               + 0.286 * cie_gauss(l, 530.9, 16.3, 31.1);
        let zb = 1.217 * cie_gauss(l, 437.0, 11.8, 36.0)
               + 0.681 * cie_gauss(l, 459.0, 26.0, 13.8);
        x += b * xb; y += b * yb; z += b * zb;
        l += 5.0;
    }
    let r = (3.2406 * x - 1.5372 * y - 0.4986 * z).max(0.0);
    let g = (-0.9689 * x + 1.8758 * y + 0.0415 * z).max(0.0);
    let b = (0.0557 * x - 0.2040 * y + 1.0570 * z).max(0.0);
    let m = r.max(g).max(b).max(1e-10);
    [r / m, g / m, b / m]
}

/// A white-balanced blackbody colour table. `wb_temperature` decides which
/// temperature renders neutral.
pub struct Blackbody {
    pub table_min: f64,
    pub table_max: f64,
    pub table_size: usize,
    /// Flattened (3, table_size), channel-fastest, matching the GPU layout.
    pub table: Vec<f32>,
}

impl Blackbody {
    pub fn new(wb_temperature: f64, table_size: usize) -> Self {
        let (table_min, table_max) = (500.0, 30000.0);
        let wb = blackbody_rgb(wb_temperature);
        let gains = [1.0 / wb[0], 1.0 / wb[1], 1.0 / wb[2]];
        let mut table = vec![0.0f32; 3 * table_size];
        for i in 0..table_size {
            let t = table_min + (table_max - table_min) * i as f64 / (table_size - 1) as f64;
            let c = blackbody_rgb(t.max(1.0));
            for k in 0..3 {
                table[3 * i + k] = (c[k] * gains[k]) as f32;
            }
        }
        Self { table_min, table_max, table_size, table }
    }
}

/// The star colour table is deliberately NOT the disc's. Sharing one made star
/// colour a function of the disc's white balance -- a 10000 K disc white point
/// put every star below it, and the whole field came out uniformly gold.
pub const STAR_WB_TEMPERATURE: f64 = 6500.0;

// ---------------------------------------------------------------------------
// Disc geometry
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct AccretionDisc {
    pub inner_radius: f64,
    pub outer_radius: f64,
    pub density_falloff: f64,
}

/// Radial brightness/density profile, matching the thin disc's opacity tapers
/// so the two disc models cannot drift apart.
fn radial_profile(s: f64, disc: &AccretionDisc, m: f64) -> f64 {
    let r = s / (2.0 * m);
    let r_in = disc.inner_radius / (2.0 * m);
    let r_out = disc.outer_radius / (2.0 * m);
    let iscotaper = ((r * r - r_in * r_in) * 0.3).clamp(0.0, 1.0);
    let t_emit = (10.034259 - 0.375 * (r * r).ln()).exp();
    let outertaper = (t_emit / 1000.0).clamp(0.0, 1.0);
    let density = ((r_out - r) / (r_out - r_in)).clamp(0.0, 1.0);
    iscotaper * outertaper * density.powf(disc.density_falloff)
}

// ---------------------------------------------------------------------------
// Value-noise fBm on a hashed lattice (build time only)
// ---------------------------------------------------------------------------

const LAT: usize = 64;

/// Deterministic lattice, so the same binary always bakes the same gas.
fn noise_lattice(seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    let mut v = vec![0.0f32; LAT * LAT * LAT];
    for x in v.iter_mut() {
        // splitmix64
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        *x = ((z >> 40) as f32) / 16777216.0;
    }
    v
}

#[inline]
fn lat(l: &[f32], i: i64, j: i64, k: i64) -> f64 {
    let m = |a: i64| a.rem_euclid(LAT as i64) as usize;
    l[m(i) + LAT * m(j) + LAT * LAT * m(k)] as f64
}

fn value_noise(l: &[f32], x: f64, y: f64, z: f64) -> f64 {
    let (ix, iy, iz) = (x.floor() as i64, y.floor() as i64, z.floor() as i64);
    let (mut fx, mut fy, mut fz) = (x - ix as f64, y - iy as f64, z - iz as f64);
    fx = fx * fx * (3.0 - 2.0 * fx);
    fy = fy * fy * (3.0 - 2.0 * fy);
    fz = fz * fz * (3.0 - 2.0 * fz);
    let c = |di, dj, dk| lat(l, ix + di, iy + dj, iz + dk);
    let m = |a: f64, b: f64, t: f64| a + t * (b - a);
    let c00 = m(c(0, 0, 0), c(1, 0, 0), fx);
    let c10 = m(c(0, 1, 0), c(1, 1, 0), fx);
    let c01 = m(c(0, 0, 1), c(1, 0, 1), fx);
    let c11 = m(c(0, 1, 1), c(1, 1, 1), fx);
    m(m(c00, c10, fy), m(c01, c11, fy), fz)
}

/// Octaves of value noise, each `lacunarity` finer and `gain` weaker,
/// normalised by the sum of amplitudes.
///
/// `gain` sets how much texture lives at small scales. Amplitude falls as
/// f^-H with H = -ln(gain)/ln(lacunarity), so the default 0.5 at lacunarity
/// 2.1 gives H ~ 0.93 -- very smooth, with the coarsest octave alone carrying
/// 53% of the signal. That is why adding octaves does nothing visible.
fn fbm(l: &[f32], x: f64, y: f64, z: f64, octaves: u32, gain: f64, lacunarity: f64) -> f64 {
    let (mut amp, mut freq, mut total, mut norm) = (0.5, 1.0, 0.0, 0.0);
    for _ in 0..octaves {
        total += amp * value_noise(l, x * freq, y * freq, z * freq);
        norm += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    total / norm
}

/// Gain that lifts Worley F2-F1 onto the same range as 1-F1.
const VEIN_SCALE: f64 = 2.5;

/// Cellular noise. `ridge` returns F2-F1, whose zero set is a thin web with a
/// kink in the gradient across it -- the sharpest structure a procedural field
/// offers, and the only kind that survives being averaged along a ray.
/// Otherwise 1-F1, the classic billow of round lobes.
fn worley(l: &[f32], x: f64, y: f64, z: f64, ridge: bool) -> f64 {
    let (ix, iy, iz) = (x.floor() as i64, y.floor() as i64, z.floor() as i64);
    let (mut f1, mut f2) = (f64::INFINITY, f64::INFINITY);
    for dk in -1..=1i64 {
        for dj in -1..=1i64 {
            for di in -1..=1i64 {
                let (cx, cy, cz) = (ix + di, iy + dj, iz + dk);
                let px = cx as f64 + lat(l, cx, cy, cz);
                let py = cy as f64 + lat(l, cx + 17, cy + 31, cz + 7);
                let pz = cz as f64 + lat(l, cx + 43, cy + 11, cz + 29);
                let d = (px - x).powi(2) + (py - y).powi(2) + (pz - z).powi(2);
                if d < f1 { f2 = f1; f1 = d; } else if d < f2 { f2 = d; }
            }
        }
    }
    let (f1, f2) = (f1.sqrt(), f2.sqrt());
    if ridge { (VEIN_SCALE * (f2 - f1)).clamp(0.0, 1.0) } else { (1.0 - f1).clamp(0.0, 1.0) }
}

fn worley_fbm(l: &[f32], x: f64, y: f64, z: f64, octaves: u32, ridge: bool) -> f64 {
    let (mut amp, mut freq, mut total, mut norm) = (0.5, 1.0, 0.0, 0.0);
    for _ in 0..octaves {
        total += amp * worley(l, x * freq, y * freq, z * freq, ridge);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    total / norm
}

/// Filament coverage in [0,1]: the base fBm shape, optionally carved by a
/// Worley detail field with the standard cloud remap
/// `(base - erosion*detail) / (1 - erosion*detail)`. Where the base is high
/// the remap barely moves it; where it is marginal the detail cuts straight
/// through to zero, giving a near-binary edge a ray either passes or does not.
#[allow(clippy::too_many_arguments)]
fn shape_coverage(l: &[f32], ls: f64, u: f64, zh: f64, octaves: u32, gain: f64,
                  lac: f64, erosion: f64, escale: f64, eoct: u32, eridge: bool) -> f64 {
    let n = fbm(l, 10.0 * ls, 6.0 * u, zh, octaves, gain, lac);
    if erosion <= 0.0 { return n; }
    let w = worley_fbm(l, escale * 10.0 * ls, escale * 6.0 * u, escale * zh, eoct, eridge);
    ((n - erosion * w) / (1.0 - erosion * w)).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// The grid
// ---------------------------------------------------------------------------

pub struct DiscVolume {
    /// Flattened (nr, nphi, nz) column-major: r fastest, matching the GPU.
    pub density: Vec<f32>,
    pub nr: usize,
    pub nphi: usize,
    pub nz: usize,
    pub log_s_in: f32,
    pub log_s_out: f32,
    pub z_max: f32,
    pub emission_scale: f32,
    pub opacity_scale: f32,
}

pub struct VolumeSettings {
    pub nr: usize,
    pub nphi: usize,
    pub nz: usize,
    pub scale_height: f64,
    pub turbulence: f64,
    pub spiral_twist: f64,
    pub octaves: u32,
    pub gain: f64,
    pub lacunarity: f64,
    pub erosion: f64,
    pub erosion_scale: f64,
    pub erosion_octaves: u32,
    /// `true` carves with Worley F2-F1 (veins), `false` with 1-F1 (billows).
    pub erosion_vein: bool,
    pub emission_scale: f64,
    pub opacity_scale: f64,
    pub seed: u64,
}

impl Default for VolumeSettings {
    fn default() -> Self {
        Self {
            nr: 192, nphi: 256, nz: 48,
            scale_height: 0.08, turbulence: 0.8, spiral_twist: 4.0,
            octaves: 4, gain: 0.5, lacunarity: 2.1,
            erosion: 0.0, erosion_scale: 4.0, erosion_octaves: 3, erosion_vein: false,
            emission_scale: 0.8, opacity_scale: 1.2,
            seed: 0x5A17_C0DE_1234_5678,
        }
    }
}

/// Azimuthal band over which the noise wrap is crossfaded. `u` is not periodic
/// in phi (the argument jumps several lattice units across the 0<->2pi wrap),
/// which prints a filament seam along the phi = 0 half-plane without this.
const WRAP_BLEND: f64 = 0.31415926535; // 18 degrees

impl DiscVolume {
    pub fn bake(disc: &AccretionDisc, m: f64, cfg: &VolumeSettings) -> Self {
        let (nr, nphi, nz) = (cfg.nr, cfg.nphi, cfg.nz);
        let z_max = 3.0 * cfg.scale_height * disc.outer_radius;
        let (log_in, log_out) = (disc.inner_radius.ln(), disc.outer_radius.ln());
        let lattice = noise_lattice(cfg.seed);
        let ero = cfg.erosion.clamp(0.0, 0.95);

        let mut density = vec![0.0f32; nr * nphi * nz];
        let nthreads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let slab = nr * nphi;

        thread::scope(|scope| {
            let lattice = &lattice;
            for (t, chunk) in density.chunks_mut(slab * ((nz + nthreads - 1) / nthreads))
                                     .enumerate()
            {
                let k_base = t * ((nz + nthreads - 1) / nthreads);
                scope.spawn(move || {
                    for (kk, plane) in chunk.chunks_mut(slab).enumerate() {
                        let k = k_base + kk;
                        let z = -z_max + (k as f64) / (nz - 1) as f64 * 2.0 * z_max;
                        for j in 0..nphi {
                            let phi = (j as f64) / nphi as f64 * std::f64::consts::TAU;
                            for i in 0..nr {
                                let s = (log_in + (i as f64) / (nr - 1) as f64
                                         * (log_out - log_in)).exp();
                                let h = cfg.scale_height * s;
                                let rho0 = radial_profile(s, disc, m);
                                let slabz = (-z * z / (2.0 * h * h)).exp();

                                // Sheared noise coordinates: radial detail fine,
                                // azimuthal stretched into arcs, spiral twist
                                // trailing with radius.
                                let u = phi + cfg.spiral_twist * (s / disc.inner_radius).ln();
                                let mut n = shape_coverage(lattice, s.ln(), u, 2.0 * z / h,
                                    cfg.octaves, cfg.gain, cfg.lacunarity, ero,
                                    cfg.erosion_scale, cfg.erosion_octaves, cfg.erosion_vein);
                                if phi > std::f64::consts::TAU - WRAP_BLEND {
                                    let mut w = (phi - (std::f64::consts::TAU - WRAP_BLEND))
                                                / WRAP_BLEND;
                                    w = w * w * (3.0 - 2.0 * w);
                                    let n2 = shape_coverage(lattice, s.ln(),
                                        u - std::f64::consts::TAU, 2.0 * z / h,
                                        cfg.octaves, cfg.gain, cfg.lacunarity, ero,
                                        cfg.erosion_scale, cfg.erosion_octaves,
                                        cfg.erosion_vein);
                                    n = (1.0 - w) * n + w * n2;
                                }
                                // fBm has a small linear variance (~+-0.12), so
                                // exponentiate to get filament-scale contrast:
                                // turbulence 1 is roughly 20x between wisp and gap.
                                let turb = (6.0 * cfg.turbulence * (n - 0.5)).exp();
                                plane[i + nr * j] = (rho0 * slabz * turb) as f32;
                            }
                        }
                    }
                });
            }
        });

        let peak = density.iter().cloned().fold(0.0f32, f32::max);
        if peak > 0.0 {
            for d in density.iter_mut() { *d /= peak; }
        }

        Self {
            density, nr, nphi, nz,
            log_s_in: log_in as f32,
            log_s_out: log_out as f32,
            z_max: z_max as f32,
            emission_scale: cfg.emission_scale as f32,
            opacity_scale: cfg.opacity_scale as f32,
        }
    }
}
