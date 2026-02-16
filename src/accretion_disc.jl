"""
    AccretionDisc(inner_radius, outer_radius, blackbody)
Represents an accretion disc around a black hole, defined by its inner and outer radii and a blackbody emission profile. The `inner_radius` and `outer_radius` parameters specify the radial extent of the disc, while the `blackbody` parameter is an instance of the `Blackbody` struct that defines the temperature profile and emission characteristics of the disc.
"""
struct AccretionDisc
    inner_radius::Float64
    outer_radius::Float64
    blackbody::Blackbody
    density_falloff::Float64
end

AccretionDisc(; inner_radius=3.0, outer_radius=20.0, blackbody=Blackbody(), density_falloff=0.0) =
    AccretionDisc(inner_radius, outer_radius, blackbody, density_falloff)

"""
    get_disc_color_doppler(r, pos, p_cartesian, bh::AbstractSpacetime, disc::AccretionDisc)
Calculates the observed color of the accretion disc at a given radius `r`, position `pos`, and photon momentum `p_cartesian`, taking into account Doppler and gravitational redshift effects. The function uses the properties of the black hole spacetime `bh` and the accretion disc `disc` to compute the local temperature and intensity of the emitted radiation, and returns the resulting color as an RGB value along with the observed temperature.
"""
function get_disc_color_doppler(r, pos, p_cartesian, bh::AbstractSpacetime, disc::AccretionDisc)
    M = bh.M
    R = r / (2M)
    Rsqr = R^2

    T_emit = exp(10.034259 - 0.375 * log(Rsqr))

    v_mag = 0.70710678 * max(R - 1.0, 0.1)^(-0.5)
    v_mag = clamp(v_mag, 0.0, 0.999)

    n_pos = pos / norm(pos)
    disc_v = v_mag * cross(SVector(0.0, 0.0, 1.0), n_pos)

    v_sqr = clamp(dot(disc_v, disc_v), 0.0, 0.99)
    gamma = 1.0 / sqrt(1.0 - v_sqr)
    n_photon = p_cartesian / norm(p_cartesian)
    opz_doppler = gamma * (1.0 + dot(disc_v, n_photon))

    opz_grav = 1.0 / sqrt(max(1.0 - 1.0 / max(R, 1.0), 0.01))

    T_obs = T_emit / clamp(opz_doppler * opz_grav, 0.1, Inf)

    intensity_val = 1.0 / (exp(29622.4 / max(T_obs, 1.0)) - 1.0)

    final_intensity = intensity_val * 100.0

    c = wb_blackbody_color_fast(T_obs, disc.blackbody)
    color = RGBf(final_intensity * c[1], final_intensity * c[2], final_intensity * c[3])
    return color, T_obs
end
