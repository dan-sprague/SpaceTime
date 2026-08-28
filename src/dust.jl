# ==============================================================================
# Interstellar dust — extinction, scattering, and thermal emission.
# ==============================================================================

"""
    InterstellarDust(; density=0.0, Rv=3.1, albedo=0.6, g_param=0.5,
                       inner_radius=3.0, outer_radius=100.0, dust_mass=1.0)

Parameters for interstellar dust along the line of sight and around the
black hole.

- `density`: column density scaling (0 = no dust).  ~1 gives Milky Way-like
  extinction over ~1 kpc scales, so use small values (e.g. 1e-4 to 1e-2) for
  the scales around a black hole.
- `Rv`: total-to-selective extinction ratio.  3.1 = diffuse ISM, 4–5 = dense
  clouds.
- `albedo`: grain albedo (fraction of extinction due to scattering).  ~0.6
  for ISM grains in the visible.
- `g_param`: Henyey-Greenstein asymmetry parameter for forward scattering.
  0 = isotropic, ~0.5–0.7 for ISM grains.
- `inner_radius`, `outer_radius`: radial extent of the dust distribution
  (in units of M, the black-hole mass parameter).  Dust inside `inner_radius`
  is destroyed by the accretion disc.
- `dust_mass`: total dust mass scaling, controls IR emission brightness.
"""
struct InterstellarDust
    density::Float64
    Rv::Float64
    albedo::Float64
    g_param::Float64
    inner_radius::Float64
    outer_radius::Float64
    dust_mass::Float64
end

InterstellarDust(; density=0.0, Rv=3.1, albedo=0.6, g_param=0.5,
                 inner_radius=3.0, outer_radius=100.0, dust_mass=1.0) =
    InterstellarDust(density, Rv, albedo, g_param, inner_radius, outer_radius, dust_mass)

# ------------------------------------------------------------------------------
# Cardelli / Clayton / Mathis (1989) extinction curve.
#
# Returns A_λ / A_V for wavelength λ in nm.  Valid for 0.1 μm ≤ λ ≤ 3.3 μm.
# The curve is parameterised by R_V = A_V / E(B-V).
# ------------------------------------------------------------------------------

function _ccm_extinction(λ_nm::Float64, Rv::Float64)
    x = 1000.0 / λ_nm   # reciprocal microns

    if x < 0.3 || x > 10.0
        return 0.0       # outside validity range — transparent
    end

    if x <= 1.1
        # Infrared: 0.3 ≤ x ≤ 1.1
        a = 0.574 * x^1.61
        b = -0.527 * x^1.61
    elseif x <= 3.3
        # Optical / NIR: 1.1 ≤ x ≤ 3.3
        y = x - 1.82
        a = 1.0 + 0.17699*y - 0.50447*y^2 - 0.02427*y^3 +
            0.72085*y^4 + 0.01979*y^5 - 0.77530*y^6 + 0.32999*y^7
        b = 0.0 + 1.41338*y + 2.28305*y^2 + 1.07233*y^3 -
            5.38434*y^4 - 0.62251*y^5 + 5.30260*y^6 - 2.09002*y^7
    elseif x <= 8.0
        # UV: 3.3 ≤ x ≤ 8.0
        F_a = x > 5.9 ? 0.0 : (x > 5.9 ? 0.0 : 0.0)
        if x >= 5.9
            Fa = 0.0
            Fb = 0.0
        else
            Fa = -0.04473*(x - 5.9)^2 - 0.009779*(x - 5.9)^3
            Fb =  0.21300*(x - 5.9)^2 + 0.120700*(x - 5.9)^3
        end
        a = 1.752 - 0.316*x - 0.104 / ((x - 4.67)^2 + 0.341) + Fa
        b = -3.090 + 1.825*x + 1.206 / ((x - 4.62)^2 + 0.263) + Fb
    else
        # Far UV: 8.0 < x ≤ 10.0
        y = x - 8.0
        a = -1.073 - 0.628*y + 0.137*y^2 - 0.070*y^3
        b = 13.670 + 4.257*y - 0.420*y^2 + 0.374*y^3
    end

    return a + b / Rv
end

# Reference wavelengths for RGB channels (nm).
const λ_R = 620.0
const λ_G = 530.0
const λ_B = 460.0

"""
    dust_extinction_rgb(dust::InterstellarDust)

Return `(A_R, A_G, A_B)` — the extinction per unit optical depth for the
three RGB channels, relative to A_V = 1.
"""
function dust_extinction_rgb(dust::InterstellarDust)
    Rv = dust.Rv
    A_R = _ccm_extinction(λ_R, Rv)
    A_G = _ccm_extinction(λ_G, Rv)
    A_B = _ccm_extinction(λ_B, Rv)
    return A_R, A_G, A_B
end

"""
    apply_dust_extinction!(color::RGBf, path_length::Float64, dust::InterstellarDust)

Attenuate `color` in-place for interstellar dust over `path_length` (in world
units, i.e. multiples of M).
"""
function apply_dust_extinction!(color::RGBf, path_length::Float64,
                                dust::InterstellarDust)
    dust.density <= 0.0 && return color
    A_R, A_G, A_B = dust_extinction_rgb(dust)
    τ = dust.density * path_length
    factor_R = exp(-A_R * τ)
    factor_G = exp(-A_G * τ)
    factor_B = exp(-A_B * τ)
    return RGBf(color.r * factor_R, color.g * factor_G, color.b * factor_B)
end

"""
    apply_dust_extinction(color, path_length, dust)

Non-mutating version. Accepts any `RGB` element type (ray colors promote to
`RGB{Float64}` when the background blend widens them) and returns `RGBf`.
"""
function apply_dust_extinction(color::RGB, path_length::Float64,
                               dust::InterstellarDust)
    dust.density <= 0.0 && return RGBf(color)
    A_R, A_G, A_B = dust_extinction_rgb(dust)
    τ = dust.density * path_length
    return RGBf(color.r * exp(-A_R * τ),
                color.g * exp(-A_G * τ),
                color.b * exp(-A_B * τ))
end

# ------------------------------------------------------------------------------
# Dust thermal emission — IR glow around the black hole.
#
# Dust grains heated by the accretion disc radiation reach an equilibrium
# temperature T_d ∝ r^(-1/2) (optically thin).  We model the emission as a
# simple radial profile added to the image in post-processing.
# ------------------------------------------------------------------------------

"""
    dust_glow_profile(r::Float64, M::Float64, dust::InterstellarDust, disc::AccretionDisc)

Return the relative brightness (0–1) of dust thermal emission at radius `r`
from a black hole of mass `M`.
"""
function dust_glow_profile(r::Float64, M::Float64, dust::InterstellarDust,
                           disc::AccretionDisc)
    if r < dust.inner_radius || r > dust.outer_radius
        return 0.0
    end
    R = r / (2M)
    R_inner = disc.inner_radius / (2M)
    # Temperature: T ∝ r^(-1/2) for optically thin dust.
    # Emission: B_λ(T) ~ T in Rayleigh-Jeans (IR), but we approximate brightness
    # as ∝ T^4 * r^(-1) (dilution) → ∝ r^(-3).
    # Simpler: just a smooth falloff.
    tau = max((R - dust.inner_radius/(2M)) / (dust.outer_radius/(2M) - dust.inner_radius/(2M)), 0.0)
    # Gaussian-like profile peaking near inner edge and falling off.
    sigma_inner = 0.5
    profile = exp(-0.5 * ((R - R_inner) / sigma_inner)^2) * (R_inner / R)^2
    return clamp(dust.dust_mass * profile, 0.0, 1.0)
end

"""
    apply_dust_glow!(image, cam, spacetime, dust, disc, blackbody)

Add a radial dust-emission glow to `image` in-place.  Uses the dust temperature
profile and the blackbody colour table.
"""
function apply_dust_glow!(image::Matrix{RGBf}, cam::AbstractCamera,
                          spacetime::Schwarzschild, dust::InterstellarDust,
                          disc::AccretionDisc)
    dust.density <= 0.0 && dust.dust_mass <= 0.0 && return image
    M = spacetime.M
    w, h = size(image)

    for j in 1:h, i in 1:w
        u, v = sensor_coordinate(i, j, w, h)
        # Trace a simple ray to estimate the impact parameter at the equatorial plane.
        # This is an approximation: we find the minimum r along the pinhole ray.
        origin, direction = get_ray(Camera(cam.pos, cam.pos + cam.fwd,
                                           cam.up_local, cam.fov_factor), u, v)
        # For a Schwarzschild black hole, the impact parameter b determines
        # the closest approach.  We approximate by projecting onto the equatorial plane.
        b = norm(cross(origin, direction))
        r_eff = max(b, dust.inner_radius)  # rough closest-approach estimate

        glow = dust_glow_profile(r_eff, M, dust, disc)
        if glow > 0.0
            # Dust temperature: T_d ≈ 1500 * (r / r_inner)^(-1/2) K (typical)
            R = r_eff / (2M)
            R_inner = disc.inner_radius / (2M)
            T_dust = 1500.0 * sqrt(R_inner / max(R, R_inner))
            c = wb_blackbody_color_fast(T_dust, disc.blackbody)
            image[i, j] = RGBf(
                image[i, j].r + glow * c[1] * 0.3,
                image[i, j].g + glow * c[2] * 0.3,
                image[i, j].b + glow * c[3] * 0.3
            )
        end
    end
    return image
end

# ------------------------------------------------------------------------------
# Henyey-Greenstein scattering phase function.
# ------------------------------------------------------------------------------

"""
    henyey_greenstein(cos_θ, g)

Henyey-Greenstein phase function for scattering angle θ.
`g` is the asymmetry parameter: g > 0 = forward-peaked, g = 0 = isotropic.
Normalised so that ∫ p(cos_θ) dΩ = 1.
"""
function henyey_greenstein(cos_θ::Float64, g::Float64)
    g2 = g * g
    denom = 1.0 + g2 - 2.0 * g * cos_θ
    return (1.0 - g2) / (4π * denom^1.5)
end

# ------------------------------------------------------------------------------
# Combined dust post-processing.
# ------------------------------------------------------------------------------

"""
    apply_dust_post!(image, cam, spacetime, dust, disc)

Apply dust glow (scattered + thermal) to the image as a post-process step.
This is a simplified approximation that doesn't require volume ray-marching.
"""
function apply_dust_post!(image::Matrix{RGBf}, cam::AbstractCamera,
                          spacetime::Schwarzschild, dust::InterstellarDust,
                          disc::AccretionDisc)
    apply_dust_glow!(image, cam, spacetime, dust, disc)
    return image
end
