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
    DiscVolume(disc::AccretionDisc; M=1.0, nr=192, nphi=256, nz=48,
               scale_height=0.08, turbulence=0.8, spiral_twist=4.0,
               noise_octaves=4, emission_scale=0.8, opacity_scale=1.2,
               rng=Random.default_rng())

Build a volumetric disc for `disc`'s annulus around a black hole of mass `M`.

- `scale_height`: H(s) = scale_height · s (flared slab); the grid spans
  ±3 scale heights at the outer edge.
- `turbulence`: relative amplitude of the fractal density modulation.
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

function DiscVolume(disc::AccretionDisc; M::Real=1.0, nr::Int=192,
                    nphi::Int=256, nz::Int=48, scale_height::Real=0.08,
                    turbulence::Real=0.8, spiral_twist::Real=4.0,
                    noise_octaves::Int=4,
                    emission_scale::Real=0.8, opacity_scale::Real=1.2,
                    rng::Random.AbstractRNG=Random.default_rng())
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
                n = _fbm(lattice, 10.0 * log(s), 6.0 * u, 2.0 * z / H,
                         noise_octaves)
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

function _fbm(lattice, x, y, z, octaves)
    amp = 0.5
    freq = 1.0
    total = 0.0
    norm = 0.0
    for _ in 1:octaves
        total += amp * _value_noise(lattice, x * freq, y * freq, z * freq)
        norm += amp
        amp *= 0.5
        freq *= 2.1
    end
    return total / norm
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
