"""
    Kerr–Schild camera tetrads and the fixed-step CPU preview renderer

The camera frame used by every renderer in the package (`ks_camera_tetrad`:
static observer, radial free-faller inside r = 2.5M, optionally Lorentz-boosted),
the per-ray initial conditions (`ks_init_photon`), and a fast fixed-step
preview renderer. The preview deliberately trades accuracy for speed: a custom
RK4 integrator (no DifferentialEquations.jl overhead), no Doppler-shifted disc
events, pinhole optics only. It is the CPU twin of the Metal kernel in
`metal.jl`, and the reference the GPU path is checked against.
"""

"""
    PreviewSettings(width, height, dt, nmax, r_escape_factor)

Settings for the fast fixed-step preview renderer.

- `width`, `height`: preview resolution in pixels.
- `dt`: fixed integration step.
- `nmax`: maximum number of steps per ray.
- `r_escape_factor`: escape radius is `r_escape_factor * norm(cam.pos)`.

Defaults are chosen for ~10–30 fps interactive preview on a modern CPU.
"""
struct PreviewSettings
    width::Int
    height::Int
    dt::Float64
    nmax::Int
    r_escape_factor::Float64
end

PreviewSettings(; width=160, height=120, dt=0.1, nmax=1000, r_escape_factor=2.0) =
    PreviewSettings(width, height, dt, nmax, r_escape_factor)

"""
    ks_rhs(μ, p_t, M)

Geodesic RHS in Cartesian Kerr–Schild coordinates for the preview renderer —
the CPU twin of `ks_rhs_mtl` (see that docstring for the derivation). State
is `(x, y, z, px, py, pz)`; the conserved `p_t` is carried separately.
Regular at both the poles and the horizon.
"""
function ks_rhs(μ::SVector{6,T}, p_t::T, M::T) where T
    x, y, z, px, py, pz = μ
    r2 = x * x + y * y + z * z
    inv_r = one(T) / sqrt(r2)
    f = 2M * inv_r
    κ = (x * px + y * py + z * pz) * inv_r
    ℓ = -p_t + κ
    c1 = f * ℓ * inv_r
    c2 = f * ℓ * (T(0.5) * ℓ + κ) * inv_r * inv_r
    return SVector{6,T}(px - c1 * x, py - c1 * y, pz - c1 * z,
                        c1 * px - c2 * x, c1 * py - c2 * y, c1 * pz - c2 * z)
end

"""
    rk4_step_preview(μ, p_t, M, dt)

One fixed-step RK4 update for the Kerr–Schild preview integrator.
"""
function rk4_step_preview(μ::SVector{6,T}, p_t::T, M::T, dt::T) where T
    half_dt = T(0.5) * dt
    k1 = ks_rhs(μ, p_t, M)
    k2 = ks_rhs(μ + half_dt * k1, p_t, M)
    k3 = ks_rhs(μ + half_dt * k2, p_t, M)
    k4 = ks_rhs(μ + dt * k3, p_t, M)
    return μ + (dt / T(6)) * (k1 + 2k2 + 2k3 + k4)
end

# The camera observer freezes into a static frame outside this radius and is
# a radial free-faller (dropped from rest here) inside it, so the view is
# unchanged in the exterior and remains physical through the horizon, where
# no static observers exist.
const _KS_FREEZE_R = 2.5      # in units of M
const _KS_FREEZE_E2 = 1.0 - 2.0 / _KS_FREEZE_R   # conserved E² of that faller

"""
    ks_camera_tetrad(pos, fwd, right, up, M) -> (u, Ef, Er, Eu)

Orthonormal camera tetrad in Cartesian Kerr–Schild coordinates, as four
contravariant 4-vectors `(t, x, y, z)`. The observer `u` is static for
`r ≥ 2.5M` and a radial free-faller dropped from rest at `2.5M` inside
(regular through the horizon; static frames don't exist there). `Ef/Er/Eu`
are the camera's forward/right/up axes, Gram–Schmidt orthonormalised under
the KS metric with forward first, so the look direction is exact.

With spin (`a ≠ 0`) the metric is the Kerr one in Kerr–Schild form and the
observer is static (u ∝ ∂_t) where f ≤ 0.72, blending smoothly to the
KS-congruence faller (u_μ ∝ −dt_μ, regular through the ergosphere and the
horizon) by f = 0.88 — so the camera may ride anywhere outside r₊.
"""
function ks_camera_tetrad(pos::SVector{3,Float64}, fwd::SVector{3,Float64},
                          right::SVector{3,Float64}, up::SVector{3,Float64},
                          M::Float64;
                          beta::SVector{3,Float64}=SVector(0.0, 0.0, 0.0),
                          a::Float64=0.0)
    local l⃗::SVector{3,Float64}, f::Float64, u::SVector{4,Float64}
    if a == 0.0
        r = norm(pos)
        x̂ = pos / r
        f = 2M / r
        # Radial-geodesic observer: E² = max(E_freeze², 1−f) gives dr/dτ = 0
        # exactly for r ≥ 2.5M and the freeze-radius faller inside.
        E = sqrt(max(_KS_FREEZE_E2, 1.0 - f))
        v = -sqrt(max(E^2 - (1.0 - f), 0.0))
        w = (1.0 - E * (E - v)) / (E - v)       # covariant u_i = w x̂_i
        lu = E + w                               # l^μ u_μ
        u = SVector(E + f * lu, ((w - f * lu) * x̂)...)
        l⃗ = x̂
    else
        # Kerr–Schild null direction and scalar, same convention as
        # `kerr_rhs_mtl`: l_μ = (1, (rx+ay)/(r²+a²), (ry−ax)/(r²+a²), z/r),
        # f = 2Mr³/(r⁴ + a²z²), with r the KS radius from the implicit
        # quartic. Observer: static (u ∝ ∂_t) where f ≤ 0.72, blending to
        # the ZAMO by f = 0.88. The ZAMO — u_μ ∝ −∇t_BL, zero angular
        # momentum, corotating at ω, no radial fall — is the natural
        # "hovering" frame inside the ergosphere and exists down to r₊
        # (where its lapse → 0 and the sky blueshift diverges, which is
        # physically what hovering at the horizon costs). The KS-slicing
        # Eulerian observer was rejected here: it falls at the escape speed,
        # and the Doppler redshift of the whole outside universe beats the
        # gravitational blueshift — a camera deep in renders near-black. In
        # KS coordinates t_BL = t_KS − A(r) with A'(r) = 2Mr/Δ, so
        # u_μ ∝ −(dt_μ − A' ∂r/∂xⁱ dxⁱ), normalised with g⁻¹ = η − f l⊗l.
        x, y, z = pos
        a2 = a * a
        wq = x * x + y * y + z * z - a2
        rk2 = 0.5 * (wq + sqrt(wq * wq + 4.0 * a2 * z * z))
        rk = sqrt(max(rk2, 1.0e-12))
        iRA = 1.0 / (rk2 + a2)
        l⃗ = SVector((rk * x + a * y) * iRA, (rk * y - a * x) * iRA, z / rk)
        Σq = rk2 * rk2 + a2 * z * z
        f = 2.0 * M * rk2 * rk / Σq
        Δ = max(rk2 - 2.0 * M * rk + a2, 1.0e-6)
        k = 2.0 * M * rk / Δ                        # A'(r)
        ∇r = SVector(x * rk2 * rk / Σq, y * rk2 * rk / Σq,
                     z * rk * (rk2 + a2) / Σq)
        luc = 1.0 + k * dot(l⃗, ∇r)                  # l^ν (−dt + A' dr)_ν
        uz_t = 1.0 + f * luc
        uz = SVector(uz_t, (k * ∇r - (f * luc) * l⃗)...)
        lu_z = uz[1] + l⃗[1] * uz[2] + l⃗[2] * uz[3] + l⃗[3] * uz[4]
        nrm2 = -uz[1]^2 + uz[2]^2 + uz[3]^2 + uz[4]^2 + f * lu_z * lu_z
        u_zamo = uz / sqrt(-nrm2)
        if f >= 0.88
            u = u_zamo
        else
            u_stat = SVector(1.0 / sqrt(1.0 - f), 0.0, 0.0, 0.0)
            wb = clamp((f - 0.72) / 0.16, 0.0, 1.0)
            wb = wb * wb * (3.0 - 2.0 * wb)
            if wb == 0.0
                u = u_stat
            else
                # Both timelike and future-pointing, so the mix is timelike;
                # renormalise under g.
                um = (1.0 - wb) * u_stat + wb * u_zamo
                lu = um[1] + l⃗[1] * um[2] + l⃗[2] * um[3] + l⃗[3] * um[4]
                nn = -um[1]^2 + um[2]^2 + um[3]^2 + um[4]^2 + f * lu * lu
                u = um / sqrt(-nn)
            end
        end
    end

    ldot(A) = A[1] + l⃗[1] * A[2] + l⃗[2] * A[3] + l⃗[3] * A[4]
    gdot(A, B) = -A[1] * B[1] + A[2] * B[2] + A[3] * B[3] + A[4] * B[4] +
                 f * ldot(A) * ldot(B)

    Ef = SVector(0.0, fwd[1], fwd[2], fwd[3])
    Er = SVector(0.0, right[1], right[2], right[3])
    Eu = SVector(0.0, up[1], up[2], up[3])
    Ef = Ef + gdot(Ef, u) * u                    # project out u (g(u,u) = −1)
    Ef = Ef / sqrt(gdot(Ef, Ef))
    Er = Er + gdot(Er, u) * u
    Er = Er - gdot(Er, Ef) * Ef
    Er = Er / sqrt(gdot(Er, Er))
    Eu = Eu + gdot(Eu, u) * u
    Eu = Eu - gdot(Eu, Ef) * Ef - gdot(Eu, Er) * Er
    Eu = Eu / sqrt(gdot(Eu, Eu))

    # Optional Lorentz boost of the whole tetrad by the camera's 3-velocity
    # `beta` (components along Ef/Er/Eu, |beta| < 1). Rays are initialised in
    # the boosted frame, so aberration, motion Doppler, and beaming all follow
    # from the standard machinery downstream (p_t carries the full shift).
    b2 = dot(beta, beta)
    if b2 > 1.0e-12
        b2 = min(b2, 0.9801)                    # clamp |β| ≤ 0.99
        β = beta * sqrt(b2 / dot(beta, beta))
        γ = 1.0 / sqrt(1.0 - b2)
        bE = β[1] * Ef + β[2] * Er + β[3] * Eu  # β^i e_i (4-vector)
        u_b = γ * (u + bE)
        k = (γ - 1.0) / b2
        Ef_b = Ef + β[1] * (k * bE + γ * u)
        Er_b = Er + β[2] * (k * bE + γ * u)
        Eu_b = Eu + β[3] * (k * bE + γ * u)
        return u_b, Ef_b, Er_b, Eu_b
    end
    return u, Ef, Er, Eu
end

"""
    ks_init_photon(origin, direction, M, tet, fwd, right, up) -> (μ, p_t)

Null-ray initialisation from the camera tetrad `tet` (see
[`ks_camera_tetrad`](@ref)). The unit pixel direction `direction` is
decomposed in the camera's flat basis and rebuilt on the orthonormal tetrad,
giving the received photon `p = ω(u + n)`; the ray is then traced with
`q = n − u`, i.e. **backward in time**, which is what lets it legally exit
the horizon when the camera is inside. Works at any r > 0.
"""
function ks_init_photon(origin::SVector{3,Float64},
                        direction::SVector{3,Float64}, M::Float64,
                        tet::NTuple{4,SVector{4,Float64}},
                        fwd::SVector{3,Float64}, right::SVector{3,Float64},
                        up::SVector{3,Float64})
    u, Ef, Er, Eu = tet
    cf = dot(direction, fwd)
    cr = dot(direction, right)
    cu = dot(direction, up)
    q = cf * Ef + cr * Er + cu * Eu - u          # past-directed null, outward
    # Lower the index: p_μ = η_μν q^ν + f l_μ (l_ν q^ν),  l_μ = (1, x̂).
    r = norm(origin)
    f = 2M / r
    lq = q[1] + (origin[1] * q[2] + origin[2] * q[3] + origin[3] * q[4]) / r
    p_t = -q[1] + f * lq
    p = SVector(q[2], q[3], q[4]) + (f * lq / r) * origin
    return vcat(origin, p), p_t
end

"""
    render_preview(cam::Camera, spacetime::Schwarzschild, background;
                   settings::PreviewSettings=PreviewSettings())

Render a fast preview image. Uses a fixed-step RK4 integrator in Cartesian
Kerr–Schild coordinates (no polar or horizon coordinate singularities) and
samples the background when a ray escapes. Rays captured by the horizon are
black.

This is intentionally lower fidelity than `render`: simple disc colouring, no
adaptive stepping, and pinhole optics only. It is the CPU twin of the Metal
kernel `trace_kernel_mtl!`.
"""
function render_preview(cam::Camera, spacetime::Schwarzschild, background;
                        settings::PreviewSettings=PreviewSettings(),
                        disc::Union{AccretionDisc,Nothing}=nothing)
    width, height = settings.width, settings.height
    image = zeros(RGBf, width, height)
    M = spacetime.M
    # Escape radius: never smaller than 30M, so a camera deep inside still
    # traces rays out to a sensible sky distance.
    r_escape = settings.r_escape_factor * max(norm(cam.pos), 15.0 * M)
    dt = settings.dt
    # Dynamic step cap, kept in sync with render_preview_mtl: with
    # radius-adaptive steps the travel legs are logarithmic; the constant is
    # the strong-field winding budget.
    nmax = min(max(settings.nmax,
                   ceil(Int, (75.0 + 6.5 * log(r_escape / M)) * M / dt)),
               20_000)

    # Asymptotic coordinate velocity of the ray (for sky sampling by
    # direction rather than escape position).
    function ray_dir(μ, p_t)
        r = max(sqrt(μ[1]^2 + μ[2]^2 + μ[3]^2), 1e-12)
        f = 2M / r
        κ = (μ[1] * μ[4] + μ[2] * μ[5] + μ[3] * μ[6]) / r
        c1 = f * (-p_t + κ) / r
        v = SVector(μ[4] - c1 * μ[1], μ[5] - c1 * μ[2], μ[6] - c1 * μ[3])
        return v / max(norm(v), 1e-20)
    end

    # Backward-traced rays can exit the horizon (camera inside, r strictly
    # increasing) but can never legally enter it. Exact kill criterion: an
    # escaping null geodesic never has a turning point below the photon
    # sphere (periapsis > 3M requires b > b_crit), so any ray moving inward
    # below ~2.95M is sub-critical horizon-bound light — the shadow. This
    # also stops rays numerically bouncing off the horizon ridge and
    # escaping as phantom sky.
    r_band = 2.95 * M
    r_kill = 0.3 * M   # numerical safety net near the singularity

    tet = ks_camera_tetrad(cam.pos, cam.fwd, cam.right, cam.up_local, M)

    Threads.@threads :static for i in 1:width
        for j in 1:height
            u, v = sensor_coordinate(i, j, width, height)
            origin, direction = get_ray(cam, u, v)
            μ, p_t = ks_init_photon(origin, direction, M, tet,
                                    cam.fwd, cam.right, cam.up_local)
            hit = false
            cam_outside = norm(origin) > 2.05 * M
            r_prev = -1.0
            for _ in 1:nmax
                r = sqrt(μ[1]^2 + μ[2]^2 + μ[3]^2)
                # A camera-outside ray below 2M is always a numerical
                # overshoot of the horizon ridge — no legal path leads there.
                if r < r_kill ||
                   (r < r_band && r_prev > 0.0 && r < r_prev - 1.0e-4 * M) ||
                   (cam_outside && r < 2.0 * M)
                    hit = true   # horizon-redshifted (or fell apart): black
                    break
                end
                if r > r_escape
                    v = ray_dir(μ, p_t)
                    θ = acos(clamp(v[3], -1.0, 1.0))
                    image[i, j] = sample_background(background, θ,
                                                    atan(v[2], v[1]))
                    hit = true
                    break
                end
                r_prev = r
                z_prev = μ[3]
                # Radius-adaptive step (kept in sync with the Metal kernel):
                # curvature ~ M/r³, so h ∝ r keeps per-step bending uniform.
                h = dt * clamp(0.16 * r / M, 1.0, 8.0)
                μ = rk4_step_preview(μ, p_t, M, h)
                # Non-finite ray: paint black, matching the kernel bail-out.
                if !(μ[1] == μ[1]) || !(μ[3] == μ[3])
                    hit = true
                    break
                end
                # Equatorial (z = 0) disc crossing.
                if !isnothing(disc) && z_prev * μ[3] < 0.0
                    s = sqrt(μ[1]^2 + μ[2]^2)
                    if disc.inner_radius < s < disc.outer_radius
                        image[i, j] = _preview_disc_color(s, disc)
                        hit = true
                        break
                    end
                end
            end
            if !hit
                # Ran out of steps. Rays still deep in the strong field are
                # (near-)critical or horizon-hugging: black. Rays that made
                # real progress sample the sky along their final direction.
                r = max(sqrt(μ[1]^2 + μ[2]^2 + μ[3]^2), 1e-12)
                if r > 4.0 * M
                    v = ray_dir(μ, p_t)
                    θ = acos(clamp(v[3], -1.0, 1.0))
                    image[i, j] = sample_background(background, θ,
                                                    atan(v[2], v[1]))
                end
            end
        end
    end
    return image
end

"""
    render_preview(cam::ThinLensCamera, spacetime::Schwarzschild, background;
                   settings::PreviewSettings=PreviewSettings())

Thin-lens preview: renders pinhole optics at the chosen focal length.  Depth of
field is intentionally omitted in preview mode to keep the renderer fast; use the
full `render()` for final DoF.
"""
function render_preview(cam::ThinLensCamera, spacetime::Schwarzschild, background;
                        settings::PreviewSettings=PreviewSettings(),
                        disc::Union{AccretionDisc,Nothing}=nothing)
    fov = (cam.sensor_width / 2.0) / cam.focal_length
    pinhole = Camera(cam.pos, cam.pos + cam.fwd, cam.up_local, fov)
    return render_preview(pinhole, spacetime, background;
                          settings=settings, disc=disc)
end
