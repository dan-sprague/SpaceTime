//! Arcade palettes: perceptually-ordered ramps sampled to a handful of tones.
//! Port of `RAMP_ANCHORS` / `arcade_palette` in src/native_shell.jl.

pub const PALETTE_MAX: usize = 32;

#[derive(Clone, Copy, PartialEq)]
pub enum Ramp { Magma, Inferno, Plasma }

/// Decile anchors of each ramp.
///
/// `Magma` is the calmest -- near-black through purple and rose to cream, with
/// no acid yellow -- and is the default for that reason. `Inferno` is the same
/// shape run warmer. `Plasma` is the most saturated: blue-violet through
/// magenta to an electric yellow that takes over any frame it lands in.
fn anchors(r: Ramp) -> &'static [[f32; 3]; 11] {
    match r {
        Ramp::Magma => &[
            [0.001, 0.000, 0.014], [0.079, 0.054, 0.212], [0.232, 0.060, 0.438],
            [0.390, 0.100, 0.502], [0.550, 0.161, 0.506], [0.716, 0.215, 0.475],
            [0.869, 0.288, 0.409], [0.968, 0.440, 0.360], [0.995, 0.624, 0.427],
            [0.997, 0.813, 0.584], [0.987, 0.991, 0.750]],
        Ramp::Inferno => &[
            [0.001, 0.000, 0.014], [0.087, 0.045, 0.225], [0.258, 0.039, 0.406],
            [0.416, 0.090, 0.433], [0.578, 0.148, 0.404], [0.736, 0.216, 0.330],
            [0.865, 0.317, 0.226], [0.955, 0.469, 0.100], [0.988, 0.645, 0.040],
            [0.964, 0.844, 0.273], [0.988, 0.998, 0.645]],
        Ramp::Plasma => &[
            [0.050, 0.030, 0.528], [0.255, 0.014, 0.615], [0.418, 0.001, 0.658],
            [0.563, 0.052, 0.642], [0.693, 0.165, 0.565], [0.798, 0.280, 0.470],
            [0.881, 0.393, 0.383], [0.949, 0.518, 0.296], [0.987, 0.652, 0.211],
            [0.988, 0.816, 0.145], [0.940, 0.975, 0.131]],
    }
}

/// A hand-built pixel-art palette: three hues, three shades each, plus a true
/// black at entry 0 for empty sky. Ten tones total.
///
/// The pipeline maps scene LUMINANCE onto this ramp, so the three hues land on
/// the three brightness zones -- shadows read **indigo**, midtones **magenta**,
/// highlights **amber** -- each stepped dark -> light. The entries are ordered
/// strictly by luminance (that is what the luminance index requires); the hues
/// are chosen to stay legible while respecting that order, which is why the
/// blue family stays dark and the amber family carries the highlights.
///
/// Flat (3, 10) channel-fastest, matching `arcade_palette`.
pub fn arcade_triad() -> Vec<f32> {
    // entry 0 is black; then indigo x3, magenta x3, amber x3 -- monotone in
    // luminance so the LUT steps through them cleanly.
    const TONES: [[f32; 3]; 10] = [
        [0.000, 0.000, 0.000], // 0  void
        [0.05, 0.04, 0.13],    // 1  indigo   dark
        [0.10, 0.10, 0.30],    // 2  indigo   mid
        [0.15, 0.15, 0.40],    // 3  indigo   light
        [0.42, 0.12, 0.40],    // 4  magenta  dark
        [0.62, 0.18, 0.50],    // 5  magenta  mid
        [0.78, 0.22, 0.52],    // 6  magenta  light
        [0.85, 0.45, 0.30],    // 7  amber    dark (ember)
        [0.97, 0.65, 0.30],    // 8  amber    mid (orange)
        [1.00, 0.90, 0.65],    // 9  amber    light (gold)
    ];
    let mut pal = vec![0.0f32; 3 * TONES.len()];
    for (k, t) in TONES.iter().enumerate() {
        pal[3 * k..3 * k + 3].copy_from_slice(t);
    }
    pal
}

/// `n` colours as a flat (3, n) channel-fastest array. With `black`, entry 0
/// is pure black and the remaining n-1 span `lo..hi` of the ramp -- so empty
/// sky bottoms out at true black rather than the ramp's darkest tone.
///
/// `lo` is the useful dial: raising it drops the muddiest low end, lowering
/// `hi` drops the blown-out top, which is where these ramps are hardest to
/// look at.
pub fn arcade_palette(n: usize, ramp: Ramp, black: bool, lo: f64, hi: f64) -> Vec<f32> {
    assert!(n >= 2 && n <= PALETTE_MAX, "palette needs 2..={PALETTE_MAX} entries");
    let a = anchors(ramp);
    let mut pal = vec![0.0f32; 3 * n];
    let k0 = if black { 1 } else { 0 };
    let m = n - k0;
    for k in k0..n {
        let t = if m == 0 { hi } else { lo + (hi - lo) * (k - k0) as f64 / m as f64 };
        let u = t.clamp(0.0, 1.0) * (a.len() - 1) as f64;
        let i = (u.floor() as usize).min(a.len() - 2);
        let f = (u - i as f64) as f32;
        for c in 0..3 {
            pal[3 * k + c] = a[i][c] + f * (a[i + 1][c] - a[i][c]);
        }
    }
    pal
}
