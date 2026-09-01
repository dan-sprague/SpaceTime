"""
    Volumetric accretion disc: a density grid marched by the renderers.

`DiscVolume` holds gas density on a cylindrical grid (log-spaced radius s,
periodic azimuth φ, uniform height z). The renderers integrate
emission/absorption through it along (curved) rays; emission is shaded with
the same Doppler/blackbody machinery as the thin disc, using Keplerian
orbital velocity at the sample's cylindrical radius.

The grid is the hand-off point for particle hydrodynamics: today it is
filled procedurally (Gaussian slab + sheared fractal noise); an SPH
simulation can later deposit particle mass into the same grid and the
renderers won't know the difference.
"""

"""
Grid resolution and noise depth the shipped look was tuned at: 192×256×48
cells over 4 fBm octaves, composed and graded at 360-line proxy renders.
[`volume_resolution`](@ref) scales from here.
"""
const VOLUME_BASE_DIMS = (192, 256, 48)
const VOLUME_BASE_OCTAVES = 4
const VOLUME_REF_HEIGHT = 360

"""
    volume_resolution(target_height; ref_height=360, base=VOLUME_BASE_DIMS,
                      base_octaves=4, max_cells=64_000_000)
        -> (nr, nphi, nz, octaves)

Grid dimensions and fBm depth for rendering at `target_height` lines.

**Measured result: this does not make the gas look more detailed.** Keep the
default grid unless you have a specific reason not to; this exists so the
experiment does not have to be repeated.

The reasoning that motivates it is sound as far as it goes — the gas grid is
independent of output resolution, so at 2160 lines an azimuthal cell at the
disc's outer edge spans ~0.49M against a pixel footprint near 0.0085M, one
cell covering tens of pixels. Scaling cells alone would only resolve the same
four octaves more smoothly, so octaves are scaled too (the finest sits at
9.26× the base frequency, already near Nyquist across 192 radial cells).

What that misses is why the extra structure never reaches the image:

- fBm halves its amplitude every octave, so a fifth octave carries about **3%**
  of the signal. Depth is nearly invisible by construction.
- Emission is alpha-composited along each ray, and a line integral is a
  low-pass filter. Every pixel already averages over many cells along its path,
  so fine 3-D structure is smeared out unless it is coherent along the line of
  sight — and isotropic fBm is not.

Going from 2.4M to 63.7M cells (255 MB, 11 s of bake) and 4 to 5 octaves
changed high-frequency image energy by **0.98×**, i.e. not at all; adding a 5×
finer ray march on top of that gave 0.95×. The gas look is limited by the
noise *spectrum*, not by resolution. The lever that does work is the amplitude
falloff in `_fbm` (`amp *= 0.5`), which costs nothing to change.

Cells grow linearly with resolution until `max_cells` (64M ≈ 256 MB at
Float32), then all three axes are scaled back together to preserve the aspect
of the base grid. Octaves grow as log2 of the linear scale.
"""
function volume_resolution(target_height::Real;
                           ref_height::Real=VOLUME_REF_HEIGHT,
                           base::NTuple{3,Int}=VOLUME_BASE_DIMS,
                           base_octaves::Int=VOLUME_BASE_OCTAVES,
                           max_cells::Int=64_000_000)
    s = max(Float64(target_height) / Float64(ref_height), 1.0)
    nr, nphi, nz = ceil.(Int, base .* s)
    cells = nr * nphi * nz
    if cells > max_cells
        # Shrink all axes by a common factor so the grid keeps its shape.
        f = (max_cells / cells)^(1 / 3)
        nr = max(base[1], floor(Int, nr * f))
        nphi = max(base[2], floor(Int, nphi * f))
        nz = max(base[3], floor(Int, nz * f))
    end
    octaves = base_octaves + max(0, floor(Int, log2(s)))
    return (nr, nphi, nz, octaves)
end

"""
    DiscVolume(disc::AccretionDisc; M=1.0, nr=192, nphi=256, nz=48,
               target_height=nothing, scale_height=0.08, turbulence=0.8,
               spiral_twist=4.0, noise_octaves=4, emission_scale=0.8,
               opacity_scale=1.2, rng=Random.default_rng())

Build a volumetric disc for `disc`'s annulus around a black hole of mass `M`.

- `target_height`: output frame height this grid will be rendered at, sizing
  `nr`/`nphi`/`nz`/`noise_octaves` via [`volume_resolution`](@ref) and
  overriding those four arguments. **Measured not to improve the look** — see
  `volume_resolution` for why (the fBm amplitude spectrum and the line
  integral along each ray, not the grid, are what limit gas detail). Off by
  default; the shipped look is 192×256×48 at 4 octaves.
- `scale_height`: H(s) = scale_height · s (flared slab); the grid spans
  ±3 scale heights at the outer edge.
- `turbulence`: relative amplitude of the fractal density modulation.
- `noise_gain` / `noise_lacunarity`: the fBm spectrum. **Leave these alone.**
  Grid resolution, octave count and gain were all measured and none of them
  makes the gas look more detailed — raising the gain makes it measurably
  *worse*, because the octaves are near-independent and the normalised sum has
  variance Σaᵢ²/(Σaᵢ)², so spreading weight across octaves flattens the field
  (0.38σ² at gain 0.5, 0.29σ² at 0.7). See [`_fbm`](@ref).
- `erosion`: strength in [0, 0.95] of the Worley carve, and **the dial that
  actually adds gas detail**. Zero (the default, and the shipped look)
  reproduces the plain fBm exactly. The reason spectral tuning fails is that
  emission is alpha-composited along each ray, and a line integral is a √N
  averaging operator that suppresses octave *k* by about lacunarity^(−k/2);
  surviving it would need amplitude *rising* with frequency. Erosion sidesteps
  the whole argument by not being a modulation: the detail field is subtracted
  from the base as a threshold, producing near-binary edges, and a ray either
  passes through a hole or it does not. See [`_shape_coverage`](@ref).
  Because erosion only removes gas, the disc gets dimmer as it rises — expect
  to raise `opacity_scale`/`emission_scale` alongside it.
- `erosion_scale` / `erosion_octaves`: frequency multiplier and depth of the
  carving field, relative to the base shape noise. The default 4× puts the
  coarsest carve just below the finest shape octave.
- `erosion_mode`: `:vein` carves with Worley F2−F1, whose zero set is a thin
  web, leaving bright filaments split by dark lanes; `:billow` carves with
  1−F1, punching round lobes. See [`_worley`](@ref).
- `spiral_twist`: azimuthal shear applied to the noise coordinates, smearing
  blobs into trailing spiral filaments.
- `emission_scale` / `opacity_scale`: brightness and optical-depth knobs used
  by the renderers (stored here so all paths agree).
"""
struct DiscVolume
    density::Array{Float32,3}     # (nr, nphi, nz), peak-normalised
    log_s_in::Float32
    log_s_out::Float32
    z_max::Float32
    scale_height::Float32
    emission_scale::Float32
    opacity_scale::Float32
end

# Width of the azimuthal crossfade that closes the noise seam at the ϕ=0↔2π
# wrap: a couple of base-frequency filament arcs, so the blend hides among
# the turbulence it is stitching.
const WRAP_BLEND = deg2rad(18.0)

function DiscVolume(disc::AccretionDisc; M::Real=1.0, nr::Int=192,
                    nphi::Int=256, nz::Int=48,
                    target_height::Union{Real,Nothing}=nothing,
                    scale_height::Real=0.08,
                    turbulence::Real=0.8, spiral_twist::Real=4.0,
                    noise_octaves::Int=4, noise_gain::Real=0.5,
                    noise_lacunarity::Real=2.1,
                    erosion::Real=0.0, erosion_scale::Real=4.0,
                    erosion_octaves::Int=3, erosion_mode::Symbol=:billow,
                    emission_scale::Real=0.8, opacity_scale::Real=1.2,
                    rng::Random.AbstractRNG=Random.default_rng())
    if target_height !== nothing
        nr, nphi, nz, noise_octaves = volume_resolution(target_height)
    end
    erosion_mode in (:vein, :billow) ||
        throw(ArgumentError("erosion_mode must be :vein or :billow, got $erosion_mode"))
    eridge = erosion_mode === :vein
    # Keep the remap's denominator away from zero: at erosion 1 a detail value
    # of 1 would divide by 0 and erase the cell regardless of its density.
    ero = clamp(Float64(erosion), 0.0, 0.95)
    s_in = disc.inner_radius
    s_out = disc.outer_radius
    z_max = 3.0 * scale_height * s_out
    density = Array{Float32,3}(undef, nr, nphi, nz)
    lattice = _noise_lattice(rng)

    log_in, log_out = log(s_in), log(s_out)
    Threads.@threads for k in 1:nz
        z = -z_max + (k - 1) / (nz - 1) * 2z_max
        for j in 1:nphi
            ϕ = (j - 1) / nphi * 2π
            for i in 1:nr
                s = exp(log_in + (i - 1) / (nr - 1) * (log_out - log_in))
                H = scale_height * s
                ρ0 = _disc_radial_profile(s, disc, M)
                slab = exp(-z^2 / (2H^2))
                # Sheared noise coordinates: radial detail fine, azimuthal
                # stretched into arcs, spiral twist trails with radius.
                u = ϕ + spiral_twist * log(s / s_in)
                n = _shape_coverage(lattice, log(s), u, 2.0 * z / H,
                                    noise_octaves, noise_gain,
                                    noise_lacunarity, ero, erosion_scale,
                                    erosion_octaves, eridge)
                # `u` is not periodic in ϕ (the noise argument jumps 6·2π
                # lattice units across the 0↔2π wrap), which printed a
                # filament seam along the ϕ=0 half-plane. Crossfade the last
                # WRAP_BLEND of azimuth onto the ϕ−2π branch so the wrap
                # column rejoins ϕ=0; only this sector's filaments change.
                if ϕ > 2π - WRAP_BLEND
                    w = (ϕ - (2π - WRAP_BLEND)) / WRAP_BLEND
                    w = w * w * (3.0 - 2.0 * w)
                    n2 = _shape_coverage(lattice, log(s), u - 2π,
                                         2.0 * z / H, noise_octaves,
                                         noise_gain, noise_lacunarity, ero,
                                         erosion_scale, erosion_octaves,
                                         eridge)
                    n = (1.0 - w) * n + w * n2
                end
                # Log-normal modulation: fBm has a small linear variance
                # (~±0.12), so exponentiate to get filament-scale density
                # contrast (turbulence=1 → roughly 20× between wisp and gap).
                turb = exp(6.0 * turbulence * (n - 0.5))
                density[i, j, k] = Float32(ρ0 * slab * turb)
            end
        end
    end
    peak = maximum(density)
    peak > 0 && (density ./= peak)
    return DiscVolume(density, Float32(log_in), Float32(log_out),
                      Float32(z_max), Float32(scale_height),
                      Float32(emission_scale), Float32(opacity_scale))
end

"""Radial brightness/density profile matching the thin disc's opacity tapers."""
function _disc_radial_profile(s, disc::AccretionDisc, M)
    R = s / (2M)
    R_in = disc.inner_radius / (2M)
    R_out = disc.outer_radius / (2M)
    iscotaper = clamp((R^2 - R_in^2) * 0.3, 0.0, 1.0)
    T_emit = exp(10.034259 - 0.375 * log(R^2))
    outertaper = clamp(T_emit / 1000.0, 0.0, 1.0)
    density = clamp((R_out - R) / (R_out - R_in), 0.0, 1.0)
    return iscotaper * outertaper * density^disc.density_falloff
end

# ---------------------------------------------------------------------------
# Value-noise fBm on a hashed lattice (CPU-only; used at grid build time)
# ---------------------------------------------------------------------------

_noise_lattice(rng) = rand(rng, Float32, 64, 64, 64)

@inline _lat(lattice, i, j, k) =
    @inbounds lattice[mod(i, 64) + 1, mod(j, 64) + 1, mod(k, 64) + 1]

function _value_noise(lattice, x, y, z)
    ix, iy, iz = floor(Int, x), floor(Int, y), floor(Int, z)
    fx, fy, fz = x - ix, y - iy, z - iz
    # smoothstep
    fx = fx * fx * (3 - 2fx); fy = fy * fy * (3 - 2fy); fz = fz * fz * (3 - 2fz)
    c000 = _lat(lattice, ix, iy, iz);     c100 = _lat(lattice, ix+1, iy, iz)
    c010 = _lat(lattice, ix, iy+1, iz);   c110 = _lat(lattice, ix+1, iy+1, iz)
    c001 = _lat(lattice, ix, iy, iz+1);   c101 = _lat(lattice, ix+1, iy, iz+1)
    c011 = _lat(lattice, ix, iy+1, iz+1); c111 = _lat(lattice, ix+1, iy+1, iz+1)
    c00 = c000 + fx * (c100 - c000); c10 = c010 + fx * (c110 - c010)
    c01 = c001 + fx * (c101 - c001); c11 = c011 + fx * (c111 - c011)
    c0 = c00 + fy * (c10 - c00); c1 = c01 + fy * (c11 - c01)
    return c0 + fz * (c1 - c0)
end

"""
    _fbm(lattice, x, y, z, octaves; gain=0.5, lacunarity=2.1)

Fractional Brownian motion: octaves of value noise, each `lacunarity`× finer
and `gain`× weaker than the last, normalised by the sum of amplitudes.

`gain` sets how much of the texture lives at small scales, and it is the dial
that controls how detailed the gas looks. Amplitude falls as f^(−H) with
H = −ln(gain)/ln(lacunarity), so the default 0.5 at lacunarity 2.1 gives
**H ≈ 0.93** — very close to the smooth extreme, with the coarsest octave
alone carrying 53% of the signal and the fourth only 7%. That is why adding
octaves or grid cells does nothing visible: a fifth octave is 3% of the
result (see [`volume_resolution`](@ref)).

Kolmogorov turbulence is H = 1/3, which at this lacunarity would be a gain of
2.1^(−1/3) ≈ 0.78. Values around 0.65–0.7 (H ≈ 0.58–0.48) put real energy in
the fine octaves while staying smoother than fully developed turbulence.

Raising `gain` *lowers* the variance of the result, which is not obvious: the
octaves are near-independent and the sum is normalised, so spreading weight
across more comparable terms averages them. Variance goes as
Σaᵢ²/(Σaᵢ)² — 0.38σ² at gain 0.5, 0.29σ² at 0.7 — and since density is
`exp(6·turbulence·(n−0.5))`, finer gas also comes out flatter. Raise
`turbulence` alongside `gain` to hold the wisp-to-gap contrast.
"""
function _fbm(lattice, x, y, z, octaves; gain::Real=0.5, lacunarity::Real=2.1)
    amp = 0.5
    freq = 1.0
    total = 0.0
    norm = 0.0
    for _ in 1:octaves
        total += amp * _value_noise(lattice, x * freq, y * freq, z * freq)
        norm += amp
        amp *= gain
        freq *= lacunarity
    end
    return total / norm
end

"""Gain that lifts Worley F2−F1 onto the same range as 1−F1; see [`_worley`](@ref)."""
const VEIN_SCALE = 2.5

"""
    _worley(lattice, x, y, z; ridge=true)

Cellular (Worley) noise: distance to scattered feature points, one per unit
lattice cell, searched over the 3×3×3 neighbourhood. Returns a value in [0, 1].

This exists because [`_value_noise`](@ref) is smooth trilinear interpolation
and **cannot produce a sharp feature at any amplitude** — every level set is a
gentle gradient, and a gentle gradient is exactly what a line integral erases.
Worley has creases: `ridge=true` returns F2−F1, which is zero along the
equidistant surfaces between neighbouring points and rises into the cell
interiors, so its zero set is a thin web with a kink in the gradient across it.
That crease is the sharpest structure available from a procedural field, and it
is what survives being averaged along a ray. `ridge=false` returns 1−F1, the
classic billow: round lobes centred on the feature points.

Feature-point coordinates are drawn from the same hashed lattice as the value
noise at three decorrelated offsets, so the whole volume stays reproducible
from one `rng`.
"""
function _worley(lattice, x, y, z; ridge::Bool=true)
    ix, iy, iz = floor(Int, x), floor(Int, y), floor(Int, z)
    f1 = Inf; f2 = Inf
    for dk in -1:1, dj in -1:1, di in -1:1
        cx, cy, cz = ix + di, iy + dj, iz + dk
        px = cx + _lat(lattice, cx, cy, cz)
        py = cy + _lat(lattice, cx + 17, cy + 31, cz + 7)
        pz = cz + _lat(lattice, cx + 43, cy + 11, cz + 29)
        d = (px - x)^2 + (py - y)^2 + (pz - z)^2
        if d < f1
            f2 = f1; f1 = d
        elseif d < f2
            f2 = d
        end
    end
    f1 = sqrt(f1); f2 = sqrt(f2)
    # Range-match the two modes. Raw F2−F1 has mean ≈0.08 against 1−F1's ≈0.48,
    # and the erosion remap is a *threshold* against the base shape noise
    # (mean 0.50, sd 0.11): a detail field that never climbs into that range
    # carves nothing and degrades into an affine contrast gain. VEIN_SCALE
    # lifts F2−F1 onto the same footing so `erosion` means the same thing in
    # both modes.
    return ridge ? clamp(VEIN_SCALE * (f2 - f1), 0.0, 1.0) :
                   clamp(1.0 - f1, 0.0, 1.0)
end

"""
    _worley_fbm(lattice, x, y, z, octaves; gain=0.5, lacunarity=2.0, ridge=true)

Octaves of [`_worley`](@ref), normalised to [0, 1]. Used as the *detail* field
that erodes the base shape — see the `erosion` argument of [`DiscVolume`](@ref).
The gain falloff matters far less here than in `_fbm`, because erosion is a
threshold operation: what reaches the image is where the field crosses the base
density, not how much amplitude it carries.
"""
function _worley_fbm(lattice, x, y, z, octaves; gain::Real=0.5,
                     lacunarity::Real=2.0, ridge::Bool=true)
    amp = 0.5
    freq = 1.0
    total = 0.0
    norm = 0.0
    for _ in 1:octaves
        total += amp * _worley(lattice, x * freq, y * freq, z * freq; ridge=ridge)
        norm += amp
        amp *= gain
        freq *= lacunarity
    end
    return total / norm
end

"""
    _shape_coverage(lattice, ls, u, zh, octaves, gain, lac,
                    erosion, escale, eoct, eridge) -> Float64

Filament coverage in [0, 1] at one grid cell: the base fBm shape, optionally
carved by a Worley detail field.

The carve is the standard cloud remap — the detail noise becomes the new
*minimum* of the range:

    coverage = remap(base, erosion·detail, 1, 0, 1)
             = (base − erosion·detail) / (1 − erosion·detail)

Where the base is high (a filament core) the remap barely moves it; where the
base is marginal the detail cuts straight through to zero. The result is a
near-binary edge instead of a smooth gradient, and a ray either passes through
a hole or it does not.

`_fbm` returns a normalised weighted average of lattice values in [0, 1], so
the base is already a coverage field and `erosion = 0` reproduces it exactly.

**The two fields have to be range-matched or this does nothing.** The remap is
a threshold, so it only bites where `erosion·detail` reaches into the base's
distribution — and the base fBm is a narrow Gaussian, mean 0.50, sd 0.11, with
a 1st percentile of 0.247, not a coverage field with mass near zero. Carving it
with raw F2−F1 (mean 0.08) drives `erosion·detail` to ≈0.11, which clears the
base on **0.17%** of cells; the remap then degenerates into an affine rescale
of the base, i.e. a broadband contrast gain with no change in structure — which
is exactly what it measured as. [`VEIN_SCALE`](@ref) and a default of `:billow`
(mean 0.48) keep the detail on the same footing as the base, so `erosion` near
0.9 actually reaches the base's bulk and cuts holes.
"""
@inline function _shape_coverage(lattice, ls, u, zh, octaves, gain, lac,
                                 erosion, escale, eoct, eridge)
    n = _fbm(lattice, 10.0 * ls, 6.0 * u, zh, octaves;
             gain=gain, lacunarity=lac)
    erosion <= 0.0 && return n
    w = _worley_fbm(lattice, escale * 10.0 * ls, escale * 6.0 * u,
                    escale * zh, eoct; ridge=eridge)
    ew = erosion * w
    return clamp((n - ew) / (1.0 - ew), 0.0, 1.0)
end

"""
    octave_weights(octaves; gain=0.5, lacunarity=2.1) -> Vector{Float64}

Fraction of the fBm signal each octave contributes. Useful for choosing
`noise_octaves`: past the point where an octave is worth a couple of percent,
adding depth (and the grid resolution to carry it) buys nothing.
"""
function octave_weights(octaves::Int; gain::Real=0.5, lacunarity::Real=2.1)
    a = [0.5 * gain^(k - 1) for k in 1:octaves]
    return a ./ sum(a)
end

# ---------------------------------------------------------------------------
# CPU sampling (used by the final renderer; the Metal kernel has its own twin)
# ---------------------------------------------------------------------------

"""
    sample_disc_volume(vol::DiscVolume, s, ϕ, z) -> Float64

Trilinear density lookup at cylindrical radius `s`, azimuth `ϕ`, height `z`.
Zero outside the grid.
"""
function sample_disc_volume(vol::DiscVolume, s, ϕ, z)
    (s <= 0 || abs(z) >= vol.z_max) && return 0.0
    ls = log(s)
    (ls <= vol.log_s_in || ls >= vol.log_s_out) && return 0.0
    nr, nphi, nz = size(vol.density)
    fr = (ls - vol.log_s_in) / (vol.log_s_out - vol.log_s_in) * (nr - 1)
    fp = mod(ϕ, 2π) / 2π * nphi
    fz = (z + vol.z_max) / (2vol.z_max) * (nz - 1)
    i0 = clamp(floor(Int, fr), 0, nr - 2); tr = fr - i0
    j0 = floor(Int, fp);                   tp = fp - j0
    k0 = clamp(floor(Int, fz), 0, nz - 2); tz = fz - k0
    j0a = mod(j0, nphi) + 1
    j1a = mod(j0 + 1, nphi) + 1
    d = vol.density
    @inbounds begin
        c00 = d[i0+1, j0a, k0+1] + tr * (d[i0+2, j0a, k0+1] - d[i0+1, j0a, k0+1])
        c10 = d[i0+1, j1a, k0+1] + tr * (d[i0+2, j1a, k0+1] - d[i0+1, j1a, k0+1])
        c01 = d[i0+1, j0a, k0+2] + tr * (d[i0+2, j0a, k0+2] - d[i0+1, j0a, k0+2])
        c11 = d[i0+1, j1a, k0+2] + tr * (d[i0+2, j1a, k0+2] - d[i0+1, j1a, k0+2])
    end
    c0 = c00 + tp * (c10 - c00)
    c1 = c01 + tp * (c11 - c01)
    return Float64(c0 + tz * (c1 - c0))
end

# ---------------------------------------------------------------------------
# Slow disc flares: subtle brightness dynamics on the static grid
# ---------------------------------------------------------------------------

"""
One flare event: a Gaussian patch in (log s, φ) that rises over `rise`
seconds of footage, decays over `decay` seconds, and drifts at its radius's
Keplerian rate. `amp` is the peak fractional density gain (the brightness
bump is at most that, since opacity responds sublinearly).
"""
struct DiscFlare
    t0::Float64        # onset, footage seconds
    rise::Float64      # attack, seconds
    decay::Float64     # exponential decay constant, seconds
    amp::Float64       # peak density gain − 1
    s0::Float64        # patch radius (M)
    φ0::Float64        # patch azimuth at t = 0
    σls::Float64       # radial width in log s
    σφ::Float64        # azimuthal width (rad)
end

"""
    DiscFlares(vol; duration=30.0, nflares=8, M_per_s=4.0,
               amp=(0.5, 1.1), rise=(2.0, 4.0), decay=(4.0, 8.0),
               rng=Xoshiro(11))

Seeded schedule of slow, subtle flares for a video of `duration` footage
seconds. Events are stretched into arcs (σφ ≫ σls, matching the sheared
filaments) and stagger across the timeline. `M_per_s` maps footage seconds to
coordinate M-time so the drift matches the shot's own time-lapse rate.

Flares live in the **outer** disc (roughly 12-19M for a 3-20M annulus), and
`amp` is a density gain of order 1 rather than a few percent, because
brightness saturates as `a = 1 − exp(−τ)`: the inner disc is optically thick,
so density there buys almost no light (measured: +300% density at 10M moves
under 5% of pixels by 0.02), while the thin outer gas responds nearly
linearly. Emission also falls with radius, so an outer flare reads as a
gentle swell of the extended glow — not a hotspot.

The intended use is the **static** gas grid: call
[`apply_flares!`](@ref) once per frame before rendering. The live fluid sim
owns `ctx.vol_gpu` and would overwrite the modulation — don't combine them.
"""
struct DiscFlares
    events::Vector{DiscFlare}
    M_per_s::Float64
    gain::Matrix{Float32}        # (nr, nphi) scratch
    scratch::Array{Float32,3}    # modulated density upload buffer
end

function DiscFlares(vol::DiscVolume;
                    duration::Real=30.0, nflares::Int=8, M_per_s::Real=4.0,
                    amp::NTuple{2,Real}=(0.5, 1.1),
                    rise::NTuple{2,Real}=(2.0, 4.0),
                    decay::NTuple{2,Real}=(4.0, 8.0),
                    rng::Random.AbstractRNG=Random.Xoshiro(11))
    nr, nphi, _ = size(vol.density)
    lsin, lsout = Float64(vol.log_s_in), Float64(vol.log_s_out)
    lerp(a, b, u) = a + (b - a) * u
    events = [DiscFlare(rand(rng) * duration,
                        lerp(rise..., rand(rng)),
                        lerp(decay..., rand(rng)),
                        lerp(amp..., rand(rng)),
                        # Outer band of the log-radius range: optically thin
                        # enough that density reads as brightness, and slow
                        # enough (a couple of deg/s) not to whip around.
                        exp(lerp(lsin, lsout, 0.75 + 0.22 * rand(rng))),
                        rand(rng) * 2π,
                        0.14 + 0.08 * rand(rng),
                        0.40 + 0.30 * rand(rng)) for _ in 1:nflares]
    return DiscFlares(events, Float64(M_per_s),
                      Matrix{Float32}(undef, nr, nphi),
                      similar(vol.density))
end

"""
    apply_flares!(ctx::MetalPreviewContext, vol::DiscVolume,
                  fl::DiscFlares, t::Real)

Upload `vol`'s density modulated by the flare gain field at footage time `t`
(seconds) into `ctx.vol_gpu`. Deterministic in `t`, so approval frames match
the final run. ~ms of CPU work and a ~9 MB upload per call.

`ctx` is a `MetalPreviewContext`; it is untyped here because this file is
included before the Metal renderer defines that type.
"""
function apply_flares!(ctx, vol::DiscVolume, fl::DiscFlares, t::Real)
    g = fl.gain
    nr, nphi = size(g)
    lsin, lsout = Float64(vol.log_s_in), Float64(vol.log_s_out)
    fill!(g, 1.0f0)
    for ev in fl.events
        τ = t - ev.t0
        env = τ <= 0.0 ? 0.0 :
              τ < ev.rise ? (u = τ / ev.rise; u * u * (3.0 - 2.0 * u)) :
              exp(-(τ - ev.rise) / ev.decay)
        env < 1.0e-3 && continue
        a = ev.amp * env
        # Keplerian co-rotation at the shot's time-lapse rate (M = 1).
        φc = ev.φ0 + sqrt(1.0 / ev.s0^3) * fl.M_per_s * t
        ls0 = log(ev.s0)
        for j in 1:nphi
            dφ = rem((j - 1) / nphi * 2π - φc, 2π, RoundNearest)
            wφ = a * exp(-dφ^2 / (2.0 * ev.σφ^2))
            wφ < 1.0e-3 && continue
            for i in 1:nr
                dls = lsin + (i - 1) / (nr - 1) * (lsout - lsin) - ls0
                g[i, j] += Float32(wφ * exp(-dls^2 / (2.0 * ev.σls^2)))
            end
        end
    end
    d = vol.density
    sc = fl.scratch
    Threads.@threads for k in axes(d, 3)
        @inbounds for j in axes(d, 2), i in axes(d, 1)
            sc[i, j, k] = d[i, j, k] * g[i, j]
        end
    end
    copyto!(ctx.vol_gpu, sc)
    return nothing
end

# ---------------------------------------------------------------------------
# Emission/absorption accumulation for the CPU (adaptive) renderer
# ---------------------------------------------------------------------------

"""
    _accumulate_volume_sample!(meta, st, disc, vol, x, y, z, px, py, pz, p_t, ds)

Add one path sample of Doppler-shaded volume emission to `meta` and attenuate
its transmittance. Position and momentum are Cartesian Kerr–Schild. Mirrors
the Metal kernel's in-loop shading.
"""
function _accumulate_volume_sample!(meta, st::Schwarzschild,
                                    disc::AccretionDisc, vol::DiscVolume,
                                    x, y, z, px, py, pz, p_t, ds)
    M = st.M
    s_cyl = sqrt(x^2 + y^2)
    s_cyl <= 1e-12 && return
    ϕ = atan(y, x)
    ρ = sample_disc_volume(vol, s_cyl, ϕ, z)
    ρ <= 1e-4 && return

    r = sqrt(x^2 + y^2 + z^2)
    # Photon coordinate velocity dx/dλ: the spatial part of the KS RHS.
    f = 2M / r
    κ = (x * px + y * py + z * pz) / r
    c1 = f * (-p_t + κ) / r
    sp, cp = y / s_cyl, x / s_cyl
    px = px - c1 * x
    py = py - c1 * y
    pz = pz - c1 * z
    plen = sqrt(px^2 + py^2 + pz^2)
    plen <= 0 && return

    # Keplerian flow at the cylindrical radius; redshift factors as in the
    # thin-disc path.
    R = s_cyl / (2M)
    T_emit = exp(10.034259 - 0.375 * log(max(R^2, 1e-6)))
    v_mag = clamp(0.70710678 / sqrt(max(R - 1.0, 0.1)), 0.0, 0.999)
    vdotn = v_mag * (-sp * px + cp * py) / plen
    γ = 1.0 / sqrt(1.0 - clamp(v_mag^2, 0.0, 0.99))
    R_sph = r / (2M)
    opz_grav = 1.0 / sqrt(max(1.0 - 1.0 / max(R_sph, 1.0), 0.01))
    opz = max(γ * (1.0 + vdotn) * opz_grav, 0.1)
    T_obs = T_emit * meta.gcam / opz
    inten = 100.0 / (exp(29622.4 / max(T_obs, 1.0)) - 1.0)
    c = wb_blackbody_color_fast(T_obs, disc.blackbody)

    τ = vol.opacity_scale * ρ * ds
    a = 1.0 - exp(-τ)
    w = meta.alpha * a * inten * vol.emission_scale
    meta.acc_color += RGBf(w * c[1], w * c[2], w * c[3])
    meta.alpha *= (1.0 - a)
    return
end

"""
    make_volume_cb(vol::DiscVolume, disc::AccretionDisc)

`DiscreteCallback` that fires after every accepted integrator step and
accumulates volume emission along the step (sub-sampled so the integrator's
large weak-field steps don't skip through the slab). Terminates rays whose
transmittance is exhausted.
"""
function make_volume_cb(vol::DiscVolume, disc::AccretionDisc)
    condition(u, t, integrator) = true
    zmax = vol.z_max
    function affect!(integrator)
        meta = integrator.p[2]
        if meta.alpha < 0.003
            terminate!(integrator)
            return
        end
        st = integrator.p[1]
        u1 = integrator.uprev
        u2 = integrator.u
        x1, y1, z1 = u1[2], u1[3], u1[4]
        x2, y2, z2 = u2[2], u2[3], u2[4]

        # Cheap reject: both endpoints on the same side, outside the slab.
        if abs(z1) > zmax && abs(z2) > zmax && sign(z1) == sign(z2)
            return
        end

        ds_tot = sqrt((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
        ds_tot <= 0 && return
        nsub = clamp(ceil(Int, ds_tot / 0.25), 1, 24)
        h = ds_tot / nsub
        p_t = u1[5]                                  # conserved
        for m in 1:nsub
            f = (m - 0.5) / nsub
            _accumulate_volume_sample!(meta, st, disc, vol,
                                       x1 + f * (x2 - x1),
                                       y1 + f * (y2 - y1),
                                       z1 + f * (z2 - z1),
                                       u1[6] + f * (u2[6] - u1[6]),
                                       u1[7] + f * (u2[7] - u1[7]),
                                       u1[8] + f * (u2[8] - u1[8]),
                                       p_t, h)
        end
        return
    end
    return DiscreteCallback(condition, affect!, save_positions=(false, false))
end
