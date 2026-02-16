function get_disc_color_doppler(r, pos, p_cartesian, bh::BlackHole)
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

    c = wb_blackbody_color_fast(T_obs)
    color = RGBf(final_intensity * c[1], final_intensity * c[2], final_intensity * c[3])
    return color, T_obs
end