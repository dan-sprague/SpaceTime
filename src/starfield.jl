# -----------------------------------------------------------------------------
# Procedural starfield for the CPU renderer.
#
# A straight port of `starfield_mtl` (src/metal.jl), sharing its `_sim_hash`, so
# the two renderers draw the *same* stars from the same seed — the field is a
# pure function of ray direction, and nothing about it needed a GPU.
#
# Why this exists rather than a bigger sky map: the 4096x2048 equirectangular
# map gives 11.4 px per degree, and a 33mm lens at 2160 lines wants 67, so the
# map is magnified ~5.9x and every star arrives as a soft 6-pixel blob. Baking a
# sharp map instead would need ~24000x12000 — 288 Mpixel, 864 MB at 8-bit — and
# would still be wrong at any other output size. Evaluating the field per ray
# costs no storage and is correct at every resolution, because the PSF is sized
# from the render height.
# -----------------------------------------------------------------------------

"""
    Starfield(; strength=1.0, texture_weight=0.0, height=1080, fov_factor=0.55,
                density=384, fill=0.5, flux=0.011, psf_pixels=0.5,
                galactic=(0,0,1), concentration=3.0, temp_min=3000,
                temp_max=16000, seed=12345, wb_temperature=STAR_WB_TEMPERATURE)

Procedural sky for the CPU renderers, matching [`set_starfield!`](@ref)
argument for argument so a shot configured one way renders the same on either
device.

Stars live on an `density × density` lattice of directions; each cell holds at
most one, placed by a hash of its indices, so the field is deterministic,
seamless and free of storage. `fill` is the occupancy, thinned away from the
galactic plane by `concentration` so the result has a Milky Way rather than
being uniform noise. Brightness runs as `ξ^(-2/3)` up from `flux`, and colour
comes from a blackbody LUT with its **own** white point, never the disc's.

`psf_pixels` is the Gaussian width as a multiple of the pixel's angular
footprint, computed from `height` and `fov_factor` — pass the *render* height.
This is what keeps stars about a pixel wide at every resolution instead of
inheriting a texture's magnification.

`texture_weight` scales the sky image the stars are drawn over: 1.0 keeps it at
full strength and adds stars on top, 0.0 replaces it entirely. The map is good
at the diffuse glow, which is low-frequency and magnifies without loss, and bad
at point sources, which is what this replaces.
"""
struct Starfield
    strength::Float64
    texture_weight::Float64
    density::Float64
    fill::Float64
    sigma::Float64            # PSF width, radians
    flux::Float64
    galactic::SVector{3,Float64}
    concentration::Float64
    temp_min::Float64
    temp_span::Float64
    seed::Int32
    lut::Matrix{Float32}
    lut_tmin::Float64
    lut_tmax::Float64
end

function Starfield(; strength::Real=1.0, texture_weight::Real=0.0,
                   height::Integer=1080, fov_factor::Real=0.55,
                   density::Real=384, fill::Real=0.5, flux::Real=0.011,
                   psf_pixels::Real=0.5,
                   galactic::NTuple{3,Real}=(0.0, 0.0, 1.0),
                   concentration::Real=3.0, temp_min::Real=3000,
                   temp_max::Real=16000, seed::Integer=12345,
                   wb_temperature::Real=STAR_WB_TEMPERATURE,
                   table_size::Int=1024)
    gn = sqrt(sum(abs2, galactic))
    gn > 0 || throw(ArgumentError("galactic normal must be non-zero"))
    # Identical to the kernel: the pixel's angular footprint is 2*fov_factor
    # across `height` rows.
    σ = psf_pixels * 2 * fov_factor / height
    lut = _star_lut_cpu(wb_temperature; table_size=table_size)
    bb = Blackbody(; wb_temperature=wb_temperature, table_size=table_size)
    return Starfield(strength, texture_weight, density, fill, σ, flux,
                     SVector{3,Float64}(galactic ./ gn), concentration,
                     temp_min, temp_max - temp_min, Int32(seed),
                     lut, bb.table_min, bb.table_max)
end

"""
    starfield_color(sf::Starfield, d::SVector{3,Float64}) -> RGBf

Star light arriving along unit direction `d`. Scans the ray's own lattice cell
and its eight neighbours, which is enough because the PSF is far narrower than
a cell.
"""
function starfield_color(sf::Starfield, d::SVector{3,Float64})
    sf.strength <= 0 && return RGBf(0, 0, 0)
    N = sf.density
    Ni = unsafe_trunc(Int32, N)
    two_pi = 2π
    u = atan(d[2], d[1]) / two_pi + 0.5
    v = (d[3] + 1.0) * 0.5
    i0 = unsafe_trunc(Int32, floor(u * N))
    j0 = unsafe_trunc(Int32, floor(v * N))

    inv2σ2 = 1.0 / (2 * sf.sigma^2)
    cut = 25.0 * sf.sigma^2          # 5σ; beyond it the Gaussian is negligible
    nlut = size(sf.lut, 2)
    acc_r = acc_g = acc_b = 0.0

    for dj in Int32(-1):Int32(1)
        jj = j0 + dj
        (jj < Int32(0) || jj >= Ni) && continue
        for di in Int32(-1):Int32(1)
            ii = Int32(mod(i0 + di, Ni))

            su = _sim_hash(ii, jj, sf.seed + Int32(1))
            sv = _sim_hash(ii, jj, sf.seed + Int32(2))
            φs = ((Float64(ii) + su) / N - 0.5) * two_pi
            sz = 2.0 * (Float64(jj) + sv) / N - 1.0
            sr = sqrt(max(1.0 - sz * sz, 0.0))
            sx = sr * cos(φs)
            sy = sr * sin(φs)

            # Chord² ≈ angle² at these scales (σ is ~1e-4 rad).
            ex = d[1] - sx; ey = d[2] - sy; ez = d[3] - sz
            d2 = ex * ex + ey * ey + ez * ez
            d2 > cut && continue

            occ = sf.fill
            if sf.concentration > 0
                sb = abs(sx * sf.galactic[1] + sy * sf.galactic[2] +
                         sz * sf.galactic[3])
                occ *= exp(-sb * sf.concentration)
            end
            _sim_hash(ii, jj, sf.seed) < occ || continue

            ξ = max(_sim_hash(ii, jj, sf.seed + Int32(3)), 1.0f-4)
            flux = sf.flux * exp(-(2 / 3) * log(Float64(ξ)))     # ξ^(−2/3)
            w = flux * exp(-d2 * inv2σ2)

            ht = _sim_hash(ii, jj, sf.seed + Int32(4))
            T = sf.temp_min + sf.temp_span * ht
            frac = (clamp(T, sf.lut_tmin, sf.lut_tmax) - sf.lut_tmin) /
                   max(sf.lut_tmax - sf.lut_tmin, 1.0e-6)
            li = clamp(round(Int, frac * (nlut - 1)) + 1, 1, nlut)
            acc_r += w * sf.lut[1, li]
            acc_g += w * sf.lut[2, li]
            acc_b += w * sf.lut[3, li]
        end
    end
    return RGBf(acc_r * sf.strength, acc_g * sf.strength, acc_b * sf.strength)
end

"""
    sky_color(background, sf, θ, ϕ, d) -> RGBf

The sky an escaping ray sees: the map scaled by `texture_weight`, plus the
procedural stars. With `sf === nothing` this is the map alone, which is the
historical behaviour.
"""
@inline function sky_color(background, sf::Union{Starfield,Nothing},
                           θ::Float64, ϕ::Float64, d::SVector{3,Float64})
    tex = RGBf(sample_background(background, θ, ϕ))
    sf === nothing && return tex
    s = starfield_color(sf, d)
    w = Float32(sf.texture_weight)
    return RGBf(tex.r * w + s.r, tex.g * w + s.g, tex.b * w + s.b)
end
