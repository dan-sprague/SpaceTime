"""
    Viewfinder / interactive preview renderer

Fast, fixed-step preview rendering and a GLMakie-based interactive viewfinder.
The preview renderer deliberately trades accuracy for speed: it uses a custom
RK4 integrator (no DifferentialEquations.jl overhead), ignores Doppler-shifted
disc events, and renders pinhole optics.  This is the same kernel you would
port to Metal for a GPU-accelerated viewfinder.
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

# ---------------------------------------------------------------------------
# Simple 3D scene geometry for the viewfinder
# ---------------------------------------------------------------------------

"""
    _wireframe_sphere(center, radius; n=64)

Return a vector of `Point3f` that draws three orthogonal great-circle meridians
through `center` with the given `radius`.  NaN-separated segments let a single
`lines!` plot render the whole wireframe.
"""
function _wireframe_sphere(center::SVector{3,T}, radius::Real; n::Int=64) where T
    points = Point3f[]
    c = Point3f(center[1], center[2], center[3])
    r = Float32(radius)

    # Equator in the xy-plane.
    for i in 0:n
        ϕ = 2.0f0 * Float32(pi) * i / n
        push!(points, c + r * Point3f(cos(ϕ), sin(ϕ), 0.0f0))
    end
    push!(points, Point3f(NaN32, NaN32, NaN32))

    # Meridian in the xz-plane.
    for i in 0:n
        ϕ = 2.0f0 * Float32(pi) * i / n
        push!(points, c + r * Point3f(cos(ϕ), 0.0f0, sin(ϕ)))
    end
    push!(points, Point3f(NaN32, NaN32, NaN32))

    # Meridian in the yz-plane.
    for i in 0:n
        ϕ = 2.0f0 * Float32(pi) * i / n
        push!(points, c + r * Point3f(0.0f0, cos(ϕ), sin(ϕ)))
    end

    return points
end

"""
    _camera_frustum(cam; len=0.3, aspect=1.5)

Return line-segment pairs for a small camera pyramid showing the camera position
and look direction.  The result is suitable for `linesegments!`.
"""
function _camera_frustum(cam::AbstractCamera; len::Real=0.3, aspect::Real=1.5)
    pos = Point3f(cam.pos[1], cam.pos[2], cam.pos[3])
    fwd = Point3f(cam.fwd[1], cam.fwd[2], cam.fwd[3])
    right = Point3f(cam.right[1], cam.right[2], cam.right[3])
    up = Point3f(cam.up_local[1], cam.up_local[2], cam.up_local[3])

    tip = pos + Float32(len) * fwd
    half_w = Float32(len * aspect * 0.5)
    half_h = Float32(len * 0.5)

    bl = tip - half_w * right - half_h * up
    br = tip + half_w * right - half_h * up
    tl = tip - half_w * right + half_h * up
    tr = tip + half_w * right + half_h * up

    segments = Point3f[]
    append!(segments, (bl, br, br, tr, tr, tl, tl, bl))   # base rectangle
    append!(segments, (bl, tip, br, tip, tl, tip, tr, tip)) # sides to apex
    push!(segments, pos)
    push!(segments, pos + Float32(2 * len) * fwd)           # look line
    return segments
end

"""
    _camera_sightline(cam; len=1.0)

Return a single line segment from the camera position along its look direction.
"""
function _camera_sightline(cam::AbstractCamera; len::Real=1.0)
    pos = Point3f(cam.pos[1], cam.pos[2], cam.pos[3])
    target = cam.pos + len * cam.fwd
    tgt = Point3f(target[1], target[2], target[3])
    return [pos, tgt]
end

"""
    _preview_disc_color(r, disc)

Simple non-Doppler colour for the preview accretion disc.  Inner regions are
brighter and yellower, outer regions dimmer and redder.
"""
function _preview_disc_color(r::Real, disc::AccretionDisc)
    t = clamp((r - disc.inner_radius) / (disc.outer_radius - disc.inner_radius),
              0.0, 1.0)
    return RGBf(1.0, 0.9 - 0.5 * t, 0.3 * (1.0 - t))
end

"""
    _disc_wireframe(disc; n=64)

Return a vector of `Point3f` drawing the inner and outer edges of the accretion
disc in the equatorial plane, plus a few radial spokes.
"""
function _disc_wireframe(disc::AccretionDisc; n::Int=64)
    points = Point3f[]
    inner = Float32(disc.inner_radius)
    outer = Float32(disc.outer_radius)

    # Inner circle.
    for i in 0:n
        ϕ = 2.0f0 * Float32(pi) * i / n
        push!(points, Point3f(inner * cos(ϕ), inner * sin(ϕ), 0.0f0))
    end
    push!(points, Point3f(NaN32, NaN32, NaN32))

    # Outer circle.
    for i in 0:n
        ϕ = 2.0f0 * Float32(pi) * i / n
        push!(points, Point3f(outer * cos(ϕ), outer * sin(ϕ), 0.0f0))
    end
    push!(points, Point3f(NaN32, NaN32, NaN32))

    # Radial spokes.
    for k in 0:7
        ϕ = 2.0f0 * Float32(pi) * k / 8.0f0
        push!(points, Point3f(inner * cos(ϕ), inner * sin(ϕ), 0.0f0))
        push!(points, Point3f(outer * cos(ϕ), outer * sin(ϕ), 0.0f0))
    end

    return points
end

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fly-cam state
# ---------------------------------------------------------------------------

const _Makie = GLMakie.Makie

"""
    FlyCamState

Mutable camera rig state for the viewfinder: world position plus yaw/pitch/roll
Euler angles (radians, world-z up). The preview worker snapshots this under a
lock while UI callbacks mutate it, so the camera itself stays an immutable
value type built on demand by `camera_from_state`.
"""
mutable struct FlyCamState
    pos::SVector{3,Float64}
    yaw::Float64
    pitch::Float64
    roll::Float64
end

function FlyCamState(cam::AbstractCamera)
    f = cam.fwd
    yaw = atan(f[2], f[1])
    pitch = asin(clamp(f[3], -1.0, 1.0))
    fwd0 = SVector(cos(pitch) * cos(yaw), cos(pitch) * sin(yaw), sin(pitch))
    right0 = normalize(cross(fwd0, SVector(0.0, 0.0, 1.0)))
    up0 = cross(right0, fwd0)
    roll = atan(dot(cam.up_local, right0), dot(cam.up_local, up0))
    FlyCamState(cam.pos, yaw, pitch, roll)
end

# Pitch is clamped short of ±π/2 so cross(fwd, ẑ) never degenerates.
const _PITCH_LIMIT = π / 2 - 0.02

"""
    camera_from_state(s::FlyCamState, focal, fstop, focus, thinlens)

Build a `Camera` (or `ThinLensCamera` when `thinlens`) from the fly-cam state
and lens settings. `focal` is in mm on a 36mm-wide sensor.
"""
function camera_from_state(s::FlyCamState, focal::Real, fstop::Real,
                           focus::Real, thinlens::Bool)
    fwd = _flycam_basis(s)[1]
    up_r = _flycam_up(s)
    target = s.pos + fwd
    if thinlens
        return ThinLensCamera(s.pos, target, up_r; focal_length=Float64(focal),
                              sensor_width=36.0, f_number=Float64(fstop),
                              focus_distance=Float64(focus))
    else
        return Camera(s.pos, target, up_r, Lens(Float64(focal)))
    end
end

"""Zero-roll orthonormal basis `(fwd, right, up)` for a fly-cam state."""
function _flycam_basis(s::FlyCamState)
    fwd = SVector(cos(s.pitch) * cos(s.yaw), cos(s.pitch) * sin(s.yaw),
                  sin(s.pitch))
    right = normalize(cross(fwd, SVector(0.0, 0.0, 1.0)))
    up = cross(right, fwd)
    return fwd, right, up
end

"""Rolled up vector for a fly-cam state."""
function _flycam_up(s::FlyCamState)
    _, right, up = _flycam_basis(s)
    return normalize(up * cos(s.roll) + right * sin(s.roll))
end

# ---------------------------------------------------------------------------
# Interactive viewfinder
# ---------------------------------------------------------------------------

"""
    viewfinder(cam, spacetime, background; settings=PreviewSettings(),
               title="SpaceTime Viewfinder", disc=nothing)

Open an interactive GLMakie viewfinder window for framing shots. The live
preview renders on the Apple GPU via Metal (`render_preview_mtl`); there is no
CPU fallback. Launch Julia as `julia -t auto,1` so the final render's worker
tasks run on default-pool threads while the UI keeps the interactive thread.

Controls:
- **Drag** on the preview to look around, **scroll** to dolly, **W/A/S/D** to
  fly, **Q/E** to descend/climb along world z (hold Shift for 5×), all while
  the mouse is over the preview.
- Sliders for roll, move speed, focal length, f-number and focus distance,
  plus a thin-lens toggle (preview always uses pinhole optics).
- A "Hectic" preset button that snaps to a dramatic close-to-the-disc
  composition with hot post-processing, ready to render.
- "Render 1-sample preview" runs the full renderer at preview resolution.

Final render panel: resolution, supersampling, filename, post-processing and
sensor/dust controls, and a "Render final image" button with a progress bar.
The final render runs on worker threads; the preview stays interactive.
"GPU draft" renders the same resolution on the Metal kernel instead
(`render_draft_mtl`: Float32, fixed step dt=0.02, 4 jittered rays/pixel, no
DoF or dust) — about 90% of the final look in a minute or two, saved as
`draft_<filename>` so it never overwrites the real render.

The preview loop coalesces requests (latest wins), so dragging never queues a
backlog of frames. A wireframe minimap next to the preview shows the black
hole, disc, and camera frustum.
"""
function viewfinder(cam::AbstractCamera, spacetime::Schwarzschild, background;
                    settings::PreviewSettings=PreviewSettings(),
                    title::String="SpaceTime Viewfinder",
                    disc::Union{AccretionDisc,Nothing}=nothing,
                    volume::Union{DiscVolume,Nothing}=nothing)
    if Threads.nthreads(:interactive) == 0 && Threads.nthreads(:default) > 1
        @warn """No interactive thread pool: CPU renders will share thread 1 \
        with the UI and the window will stall during "1-sample preview" and \
        "Render final image". Launch with `julia -t auto,1` (workers + one \
        interactive thread) for a responsive UI."""
    end
    fig = Figure(size=(1380, 900))

    # 2D render panel.
    ax = GLMakie.Axis(fig[1, 1], aspect=DataAspect(), title=title)
    img_obs = Observable(zeros(RGBf, settings.width, settings.height))
    image!(ax, img_obs)
    hidedecorations!(ax)
    for k in (:rectanglezoom, :dragpan, :scrollzoom, :limitreset)
        deregister_interaction!(ax, k)
    end

    # 3D minimap panel.
    scene_3d = GLMakie.LScene(fig[1, 2]; scenekw=(backgroundcolor=:black,))
    colsize!(fig.layout, 1, GLMakie.Relative(0.55))
    colsize!(fig.layout, 2, GLMakie.Relative(0.45))
    rowsize!(fig.layout, 1, GLMakie.Relative(0.48))
    bh_sphere_obs = Observable(Point3f[])
    cam_frustum_obs = Observable(Point3f[])
    cam_sightline_obs = Observable(Point3f[])
    cam_pos_obs = Observable(Point3f(0.0f0, 0.0f0, 0.0f0))
    disc_rings_obs = Observable(Point3f[])

    lines!(scene_3d, bh_sphere_obs; color=:orange, linewidth=1,
           label="Black hole")
    lines!(scene_3d, disc_rings_obs; color=:red, linewidth=1,
           label="Accretion disc")
    linesegments!(scene_3d, cam_frustum_obs; color=:cyan, linewidth=2)
    linesegments!(scene_3d, cam_sightline_obs; color=:green, linewidth=1)
    scatter!(scene_3d, cam_pos_obs; color=:cyan, markersize=8)

    # -------------------------------------------------------------------------
    # Camera state and Metal preview loop
    # -------------------------------------------------------------------------
    init_focal = cam isa ThinLensCamera ? cam.focal_length : 18.0 / cam.fov_factor
    init_fstop = cam isa ThinLensCamera ? cam.focal_length / cam.aperture : 2.8
    init_focus = cam isa ThinLensCamera ? cam.focus_distance : norm(cam.pos)
    init_focal = clamp(init_focal, 10.0, 200.0)

    state = FlyCamState(cam)
    state_lock = ReentrantLock()
    thinlens_obs = Observable(cam isa ThinLensCamera)
    focal_obs = Observable(Float64(init_focal))
    fstop_obs = Observable(Float64(init_fstop))
    focus_obs = Observable(Float64(init_focus))
    move_speed_obs = Observable(2.0)
    status_obs = Observable("Starting Metal preview…")
    cam_pos_label_obs = Observable("")

    build_camera() = lock(state_lock) do
        camera_from_state(state, focal_obs[], fstop_obs[], focus_obs[],
                          thinlens_obs[])
    end

    # MAIN THREAD ONLY: updates minimap gizmos and the position readout.
    minimap_disc_outer = isnothing(disc) ? 10.0 : disc.outer_radius
    function update_scene_3d!(cam_now::AbstractCamera)
        d = max(norm(cam_now.pos), 1.0)
        cam_frustum_obs[] = _camera_frustum(cam_now; len=0.05 * d)
        cam_sightline_obs[] = _camera_sightline(cam_now; len=d)
        cam_pos_obs[] = Point3f(cam_now.pos[1], cam_now.pos[2], cam_now.pos[3])

        # Auto-scale the minimap view with the camera's distance, with a wide
        # hysteresis band so it doesn't fight manual orbiting/zooming. The
        # orbit direction the user chose is preserved; only distance changes.
        target_dist = 2.2f0 * Float32(max(1.2 * minimap_disc_outer, 1.1 * d,
                                          10.0 * spacetime.M))
        cc = _Makie.cameracontrols(scene_3d.scene)
        eye = cc.eyeposition[]
        look = cc.lookat[]
        cur_dist = norm(eye .- look)
        if cur_dist < 0.5f0 * target_dist || cur_dist > 2.0f0 * target_dist
            dir = cur_dist > 1.0f-6 ? (eye .- look) ./ cur_dist :
                  Vec3f(0.7f0, -0.5f0, 0.5f0)
            update_cam!(scene_3d.scene, Vec3f(look .+ dir .* target_dist),
                        Vec3f(look), Vec3f(0, 0, 1))
        end
        yaw_deg, pitch_deg = lock(state_lock) do
            rad2deg(state.yaw), rad2deg(state.pitch)
        end
        p = round.(cam_now.pos; digits=1)
        cam_pos_label_obs[] = string("pos = (", p[1], ", ", p[2], ", ", p[3],
                                     ")  yaw = ", round(yaw_deg; digits=1),
                                     "°  pitch = ", round(pitch_deg; digits=1), "°")
    end

    ctx = MetalPreviewContext(background, settings.width, settings.height;
                              dt=settings.dt, nmax=settings.nmax,
                              r_escape_factor=settings.r_escape_factor,
                              disc=disc, volume=volume)

    # Latest-wins request coalescing: UI bumps a version and pokes the worker;
    # the worker re-renders until it has caught up, publishing only the newest
    # frame. All request_render calls happen on the main thread.
    render_version = Threads.Atomic{Int}(0)
    wakeup = Channel{Nothing}(1)
    img_chan = Channel{Tuple{Matrix{RGBf},AbstractCamera,Float64}}(1)

    function request_render()
        Threads.atomic_add!(render_version, 1)
        isready(wakeup) || put!(wakeup, nothing)
        return nothing
    end

    Threads.@spawn begin
        done = 0
        # Reused ping-pong buffers (two images so the GL texture upload never
        # races the next frame's write) and a frame-rate cap: rendering
        # faster than the display only floods thread 1 with uploads and GC.
        host_buf = Array{Float32,3}(undef, 3, settings.width, settings.height)
        img_a = Matrix{RGBf}(undef, settings.width, settings.height)
        img_b = Matrix{RGBf}(undef, settings.width, settings.height)
        flip = false
        min_period = 1 / 40
        try
            while true
                take!(wakeup)
                while done < render_version[]
                    v = render_version[]
                    cam_now = build_camera()
                    t0 = time()
                    # A failed frame must not kill the worker: log it, skip
                    # the frame, and keep serving future requests.
                    img = try
                        flip = !flip
                        render_preview_mtl!(flip ? img_a : img_b, host_buf,
                                            ctx, cam_now, spacetime)
                    catch e
                        e isa InvalidStateException && rethrow()
                        @error "Preview frame failed" exception=(e, catch_backtrace())
                        nothing
                    end
                    done = v
                    if img !== nothing
                        while isready(img_chan)
                            take!(img_chan)
                        end
                        put!(img_chan, (img, cam_now, time() - t0))
                    end
                    elapsed = time() - t0
                    elapsed < min_period && sleep(min_period - elapsed)
                end
            end
        catch e
            e isa InvalidStateException || rethrow()   # channel closed: exit
        end
    end

    # Thread-1 drainer: the only writer of preview frames into observables.
    last_img_size = Ref((settings.width, settings.height))
    @async try
        for (img, cam_now, elapsed) in img_chan
            img_obs[] = img
            if size(img) != last_img_size[]
                last_img_size[] = size(img)
                autolimits!(ax)
            end
            update_scene_3d!(cam_now)
            status_obs[] = string("Preview: ", round(1000 * elapsed; digits=1),
                                  " ms / ", round(1 / max(elapsed, 1e-6); digits=1),
                                  " fps")
        end
    catch e
        e isa InvalidStateException || rethrow()
    end

    on(events(fig.scene).window_open) do open
        if !open
            close(wakeup)
            close(img_chan)
        end
    end

    # -------------------------------------------------------------------------
    # Fly-cam input on the preview axis
    # -------------------------------------------------------------------------
    dragging = Ref(false)
    last_mouse = Ref(Point2f(0, 0))

    on(events(ax.scene).mousebutton) do ev
        if ev.button == Mouse.left
            if ev.action == Mouse.press && _Makie.is_mouseinside(ax.scene)
                dragging[] = true
                last_mouse[] = events(ax.scene).mouseposition[]
                return _Makie.Consume(true)
            elseif ev.action == Mouse.release
                dragging[] = false
            end
        end
        return _Makie.Consume(false)
    end

    on(events(ax.scene).mouseposition) do mp
        dragging[] || return _Makie.Consume(false)
        δ = mp .- last_mouse[]
        last_mouse[] = mp
        pxw = max(ax.scene.viewport[].widths[1], 1)
        # Full axis width ≈ full horizontal field of view.
        k = 2.0 * (18.0 / focal_obs[]) / pxw
        lock(state_lock) do
            state.yaw -= δ[1] * k
            state.pitch = clamp(state.pitch + δ[2] * k, -_PITCH_LIMIT, _PITCH_LIMIT)
        end
        request_render()
        return _Makie.Consume(true)
    end

    on(events(ax.scene).scroll) do sc
        (_Makie.is_mouseinside(ax.scene) && sc[2] != 0) || return _Makie.Consume(false)
        lock(state_lock) do
            fwd = _flycam_basis(state)[1]
            step = 0.05 * sc[2] * max(norm(state.pos), 2.0)
            state.pos += step * fwd
            rn = norm(state.pos)
            rn < 0.45 * spacetime.M && (state.pos *= 0.45 * spacetime.M / rn)
        end
        request_render()
        return _Makie.Consume(true)
    end

    # -------------------------------------------------------------------------
    # Control panel: three columns
    # -------------------------------------------------------------------------
    controls = GridLayout(fig[2, 1:2])

    # --- Column 1: camera & lens ---
    cam_col = GridLayout(controls[1, 1]; valign=:top, tellheight=false)
    crow = 1
    Label(cam_col[crow, 1], "Camera"; fontsize=16, halign=:left)
    crow += 1

    cam_sg = SliderGrid(
        cam_col[crow, 1],
        (label = "Roll (°)", range = -180.0:1.0:180.0, format = "{:.0f}",
         startvalue = rad2deg(state.roll)),
        (label = "Move speed", range = 0.1:0.1:20.0, format = "{:.1f}",
         startvalue = move_speed_obs[]),
        (label = "Focal length (mm)", range = 10.0:1.0:200.0, format = "{:.0f}",
         startvalue = init_focal),
        (label = "F-number", range = 1.0:0.1:22.0, format = "{:.1f}",
         startvalue = init_fstop),
        (label = "Focus distance", range = 1.0:1.0:1000.0, format = "{:.0f}",
         startvalue = init_focus),
        tellwidth = false, tellheight = true
    )
    on(cam_sg.sliders[1].value) do val
        lock(state_lock) do
            state.roll = deg2rad(val)
        end
        request_render()
    end
    on(cam_sg.sliders[2].value) do val
        move_speed_obs[] = val
    end
    on(cam_sg.sliders[3].value) do val
        focal_obs[] = val
        request_render()
    end
    on(cam_sg.sliders[4].value) do val
        fstop_obs[] = val
    end
    on(cam_sg.sliders[5].value) do val
        focus_obs[] = val
    end
    crow += 1

    lens_grid = GridLayout(cam_col[crow, 1])
    Label(lens_grid[1, 1], "Thin lens (final render DoF)"; halign=:left)
    thinlens_toggle = Toggle(lens_grid[1, 2]; active=thinlens_obs[])
    on(thinlens_toggle.active) do active
        thinlens_obs[] = active
    end
    Label(lens_grid[2, 1], "Volumetric disc"; halign=:left)
    volume_toggle = Toggle(lens_grid[2, 2]; active=!isnothing(volume))
    on(volume_toggle.active) do active
        set_volume_enabled!(ctx, active)
        request_render()
    end
    # The CPU renders (1-sample, final) honour the same switch.
    active_volume() = (volume_toggle.active[] ? volume : nothing)
    crow += 1

    btn_grid = GridLayout(cam_col[crow, 1])
    force_btn = Button(btn_grid[1, 1]; label="Force render")
    hectic_btn = Button(btn_grid[1, 2]; label="Hectic preset")
    full_preview_btn = Button(btn_grid[1, 3]; label="1-sample preview")
    on(force_btn.clicks) do _
        request_render()
    end
    crow += 1

    # tellwidth=false: long status text must not widen the column past its
    # relative share (it would push the whole column off the window edge).
    Label(cam_col[crow, 1], status_obs; halign=:left, tellwidth=false)
    crow += 1
    Label(cam_col[crow, 1], cam_pos_label_obs; halign=:left, tellwidth=false)
    crow += 1
    Label(cam_col[crow, 1],
          "Drag: look · Scroll: dolly · WASD: move · Q/E: down/up (z) · Shift: fast";
          halign=:left, color=:gray, tellwidth=false, word_wrap=true)
    rowgap!(cam_col, 6)

    # --- Column 2: post-processing ---
    post_col = GridLayout(controls[1, 2]; valign=:top, tellheight=false)
    Label(post_col[1, 1], "Post-processing"; fontsize=16, halign=:left)
    # Two side-by-side slider grids: 13 stacked rows would be taller than the
    # controls row and push the last sliders off screen.
    post_pair = GridLayout(post_col[2, 1])
    post_sg_a = SliderGrid(
        post_pair[1, 1],
        (label = "Gain", range = 0.0:0.01:2.0, format = "{:.2f}", startvalue = 1.0),
        (label = "Exposure (EV)", range = -5.0:0.1:5.0, format = "{:.1f}", startvalue = 0.0),
        (label = "Gamma", range = 0.1:0.05:3.0, format = "{:.2f}", startvalue = 2.2),
        (label = "Bloom strength", range = 0.0:0.05:2.0, format = "{:.2f}", startvalue = 0.6),
        (label = "Bloom threshold", range = 0.0:0.05:2.0, format = "{:.2f}", startvalue = 0.5),
        (label = "Bloom radius", range = 1.0:1.0:50.0, format = "{:.0f}", startvalue = 15.0),
        (label = "Bloom power", range = 0.1:0.1:3.0, format = "{:.1f}", startvalue = 1.5),
        valign = :top, tellwidth = false, tellheight = true
    )
    post_sg_b = SliderGrid(
        post_pair[1, 2],
        (label = "Streak strength", range = 0.0:0.05:2.0, format = "{:.2f}", startvalue = 0.3),
        (label = "Streak length", range = 0.05:0.05:1.0, format = "{:.2f}", startvalue = 0.4),
        (label = "Streak width", range = 0.5:0.5:5.0, format = "{:.1f}", startvalue = 1.5),
        (label = "Star spikes", range = 2:1:8, format = "{:.0f}", startvalue = 4),
        (label = "Color preserve", range = 0.0:0.05:1.0, format = "{:.2f}", startvalue = 0.75),
        (label = "Contrast", range = -1.0:0.05:1.0, format = "{:.2f}", startvalue = 0.0),
        valign = :top, tellwidth = false, tellheight = true
    )
    # Fixed widths: the grids only report a collapsed minimum width, so
    # relative sizing here would shrink-wrap and overlap them.
    colsize!(post_pair, 1, GLMakie.Fixed(270))
    colsize!(post_pair, 2, GLMakie.Fixed(270))
    colgap!(post_pair, 14)
    # Kept in the original 13-slider order: preset and snapshot code index it.
    post_sliders_all = vcat(post_sg_a.sliders, post_sg_b.sliders)
    rowgap!(post_col, 6)

    # --- Column 3: sensor, dust, final render ---
    out_col = GridLayout(controls[1, 3]; valign=:top, tellheight=false)
    orow = 1
    Label(out_col[orow, 1], "Sensor & output"; fontsize=16, halign=:left)
    orow += 1

    sensor_grid = GridLayout(out_col[orow, 1])
    Label(sensor_grid[1, 1], "ISO"; halign=:left)
    iso_slider = Slider(sensor_grid[1, 2]; range=50.0:50.0:12800.0,
                        startvalue=100.0, tellwidth=false)
    Label(sensor_grid[2, 1], "Read noise (e⁻)"; halign=:left)
    read_noise_slider = Slider(sensor_grid[2, 2]; range=0.0:0.1:10.0,
                               startvalue=2.0, tellwidth=false)
    Label(sensor_grid[3, 1], "Exposure time (s)"; halign=:left)
    exp_time_tb = Textbox(sensor_grid[3, 2]; stored_string="1.0",
                          validator=Float64, tellwidth=false)
    Label(sensor_grid[4, 1], "Saturation"; halign=:left)
    saturation_tb = Textbox(sensor_grid[4, 2]; stored_string="1000000.0",
                            validator=Float64, tellwidth=false)
    Label(sensor_grid[5, 1], "Dust density"; halign=:left)
    dust_density_slider = Slider(sensor_grid[5, 2]; range=0.0:0.001:0.1,
                                 startvalue=0.0, tellwidth=false)
    Label(sensor_grid[6, 1], "Dust mass"; halign=:left)
    dust_mass_slider = Slider(sensor_grid[6, 2]; range=0.0:0.1:5.0,
                              startvalue=1.0, tellwidth=false)
    Label(sensor_grid[7, 1], "ACES tonemap"; halign=:left)
    tonemap_toggle = Toggle(sensor_grid[7, 2]; active=true)
    Label(sensor_grid[8, 1], "Lens dust count"; halign=:left)
    lens_dust_count_slider = Slider(sensor_grid[8, 2]; range=0:1:50,
                                    startvalue=0, tellwidth=false)
    Label(sensor_grid[9, 1], "Micro streaks"; halign=:left)
    micro_streaks_count_slider = Slider(sensor_grid[9, 2]; range=0:1:20,
                                        startvalue=0, tellwidth=false)
    Label(sensor_grid[10, 1], "Auto balance"; halign=:left)
    auto_balance_toggle = Toggle(sensor_grid[10, 2]; active=false)
    colsize!(sensor_grid, 1, GLMakie.Auto())
    colsize!(sensor_grid, 2, GLMakie.Relative(0.55))
    rowgap!(sensor_grid, 4)
    orow += 1

    # Two textbox pairs per row keeps the column short enough to fit on
    # screen together with the ten sensor rows above it.
    settings_grid = GridLayout(out_col[orow, 1])
    Label(settings_grid[1, 1], "Width"; halign=:left)
    width_tb = Textbox(settings_grid[1, 2]; stored_string="3840", validator=Int,
                       tellwidth=false)
    Label(settings_grid[1, 3], "Height"; halign=:left)
    height_tb = Textbox(settings_grid[1, 4]; stored_string="2160", validator=Int,
                        tellwidth=false)
    Label(settings_grid[2, 1], "Samples"; halign=:left)
    samples_tb = Textbox(settings_grid[2, 2]; stored_string="4", validator=Int,
                         tellwidth=false)
    Label(settings_grid[2, 3], "File"; halign=:left)
    filename_tb = Textbox(settings_grid[2, 4]; stored_string="render.png",
                          tellwidth=false)
    colsize!(settings_grid, 2, GLMakie.Relative(0.28))
    colsize!(settings_grid, 4, GLMakie.Relative(0.28))
    rowgap!(settings_grid, 4)
    orow += 1

    final_btn_grid = GridLayout(out_col[orow, 1]; halign=:left)
    render_btn = Button(final_btn_grid[1, 1]; label="Render final image")
    draft_btn = Button(final_btn_grid[1, 2]; label="GPU draft")
    live_btn = Button(final_btn_grid[1, 3]; label="Save live view")
    colgap!(final_btn_grid, 8)
    orow += 1

    rowgap!(out_col, 6)

    # Progress bar: a full-width strip at the bottom of the figure (its own
    # layout row), so it can never be clipped off by a tall control column.
    progress_obs = Observable(0.0)
    progress_label_obs = Observable("Ready")
    prog_row = GridLayout(fig[3, 1:2])
    pax = GLMakie.Axis(prog_row[1, 1]; height=14, limits=(0, 1, 0, 1),
                       backgroundcolor=RGBf(0.15, 0.15, 0.15))
    hidedecorations!(pax)
    hidespines!(pax)
    for k in (:rectanglezoom, :dragpan, :scrollzoom, :limitreset)
        deregister_interaction!(pax, k)
    end
    poly!(pax, @lift(Rect2f(0.0, 0.0, max($progress_obs, 1e-4), 1.0));
          color=:seagreen)
    Label(prog_row[1, 2], progress_label_obs; halign=:left, width=340)
    colgap!(prog_row, 12)

    colsize!(controls, 1, GLMakie.Relative(0.29))
    colsize!(controls, 2, GLMakie.Relative(0.42))
    colsize!(controls, 3, GLMakie.Relative(0.29))
    colgap!(controls, 20)

    # -------------------------------------------------------------------------
    # Hectic preset: the "COOL SCENE" composition (see examples/hero_shot.jl)
    # -------------------------------------------------------------------------
    on(hectic_btn.clicks) do _
        world_up = SVector(0.0, 0.0, 1.0)
        world_right = SVector(0.0, 1.0, 0.0)
        θ_roll = deg2rad(20.0)
        tilted_up = normalize(world_up * cos(θ_roll) + world_right * sin(θ_roll))
        preset_cam = Camera(SVector(30.0, 1.1, 1.6), SVector(0.0, 0.0, 0.0),
                            tilted_up, 0.55)
        s = FlyCamState(preset_cam)
        lock(state_lock) do
            state.pos = s.pos
            state.yaw = s.yaw
            state.pitch = s.pitch
            state.roll = s.roll
        end
        set_close_to!(cam_sg.sliders[1], rad2deg(s.roll))
        set_close_to!(cam_sg.sliders[3], 33.0)   # ≈ fov_factor 0.55
        set_close_to!(cam_sg.sliders[4], 2.0)
        set_close_to!(cam_sg.sliders[5], 27.0)   # ≈ distance to disc inner edge
        thinlens_toggle.active[] = true
        for (sl, val) in zip(post_sliders_all,
                             (1.0, 1.2, 0.2, 1.0, 0.5, 10.0, 1.5,
                              2.0, 0.1, 1.0, 4.0, 0.75, 0.0))
            set_close_to!(sl, val)
        end
        set_close_to!(iso_slider, 400.0)
        request_render()
    end

    # -------------------------------------------------------------------------
    # Full-quality renders (worker threads + thread-1 finishers)
    # -------------------------------------------------------------------------
    rendering = Ref(false)   # thread-1 only
    progress_atomic = Threads.Atomic{Float64}(0.0)
    set_progress!(p) = (Threads.atomic_xchg!(progress_atomic, Float64(p)); nothing)

    on(full_preview_btn.clicks) do _
        rendering[] && return
        rendering[] = true
        cam_now = build_camera()
        vol_now = active_volume()
        update_scene_3d!(cam_now)
        status_obs[] = "Rendering 1-sample preview…"
        result = Channel{Any}(1)
        Threads.@spawn begin
            try
                t0 = time()
                img = render(cam_now, spacetime, background;
                             disc=disc, volume=vol_now, width=settings.width,
                             height=settings.height, samples=1)
                put!(result, (:ok, img, time() - t0))
            catch e
                @error "1-sample preview failed" exception=(e, catch_backtrace())
                put!(result, (:error, e))
            end
        end
        @async begin
            res = take!(result)
            rendering[] = false
            if res[1] === :ok
                img_obs[] = res[2]
                status_obs[] = string("1-sample preview: ",
                                      round(res[3]; digits=2), " s")
            else
                status_obs[] = "Preview error: $(res[2])"
            end
        end
    end

    # Shared handler for the CPU final render and the GPU draft render. The
    # two differ only in who traces the rays and in the dust stage (the GPU
    # kernel has no dust model, so drafts skip it).
    function start_photo_render(draft::Bool)
        rendering[] && return
        width_val = tryparse(Int, width_tb.stored_string[])
        height_val = tryparse(Int, height_tb.stored_string[])
        samples_val = tryparse(Int, samples_tb.stored_string[])
        t_exp_val = tryparse(Float64, exp_time_tb.stored_string[])
        saturation_val = tryparse(Float64, saturation_tb.stored_string[])

        if isnothing(width_val) || isnothing(height_val) ||
           isnothing(samples_val) || isnothing(t_exp_val) || isnothing(saturation_val)
            progress_label_obs[] = "Error: invalid numeric input"
            return
        end
        if width_val <= 0 || height_val <= 0 || samples_val <= 0 ||
           t_exp_val <= 0.0 || saturation_val <= 0.0
            progress_label_obs[] = "Error: resolution/samples/sensor values must be positive"
            return
        end

        # Snapshot all widget state on the UI thread before spawning.
        cam_now = build_camera()
        vol_now = active_volume()
        filename = filename_tb.stored_string[]
        post_sliders = post_sliders_all
        gain = post_sliders[1].value[]
        exposure = post_sliders[2].value[]
        gamma = post_sliders[3].value[]
        bloom_strength = post_sliders[4].value[]
        threshold = post_sliders[5].value[]
        bloom_radius = post_sliders[6].value[]
        bloom_power = post_sliders[7].value[]
        streak_strength = post_sliders[8].value[]
        streak_length = post_sliders[9].value[]
        streak_width = post_sliders[10].value[]
        n_spikes = Int(round(post_sliders[11].value[]))
        hue_preserve = post_sliders[12].value[]
        contrast = post_sliders[13].value[]
        do_auto_balance = auto_balance_toggle.active[]
        iso = iso_slider.value[]
        read_noise = read_noise_slider.value[]
        dust_density = dust_density_slider.value[]
        dust_mass = dust_mass_slider.value[]
        tonemap = tonemap_toggle.active[] ? :aces : :reinhard
        lens_dust_count = Int(round(lens_dust_count_slider.value[]))
        micro_streaks_count = Int(round(micro_streaks_count_slider.value[]))
        dust = InterstellarDust(; density=dust_density, dust_mass=dust_mass)
        save_name = draft ? "draft_" * basename(filename) : filename
        active_btn = draft ? draft_btn : render_btn
        idle_label = active_btn.label[]

        rendering[] = true
        active_btn.label[] = "Rendering…"
        set_progress!(0.0)
        result = Channel{Any}(1)

        Threads.@spawn begin
            try
                img = if draft
                    render_draft_mtl(ctx, cam_now, spacetime;
                                     width=width_val, height=height_val,
                                     samples=2, dt=0.02,
                                     progress=set_progress!)
                else
                    render(cam_now, spacetime, background;
                           disc=disc, dust=dust, volume=vol_now,
                           width=width_val, height=height_val,
                           samples=samples_val,
                           progress=set_progress!)
                end
                set_progress!(1.0)
                if !draft && !isnothing(disc)
                    apply_dust_post!(img, cam_now, spacetime, dust, disc)
                end
                img = postprocess(img; gain=gain, exposure=exposure, gamma=gamma,
                                  bloom_strength=bloom_strength, threshold=threshold,
                                  bloom_radius=bloom_radius, bloom_power=bloom_power,
                                  streak_strength=streak_strength,
                                  streak_length=streak_length,
                                  streak_width=streak_width,
                                  n_spikes=n_spikes, tonemap=tonemap,
                                  tonemap_hue_preserve=hue_preserve,
                                  contrast=contrast)
                if lens_dust_count > 0
                    apply_lens_dust!(img; lens_dust=LensDust(count=lens_dust_count))
                end
                if micro_streaks_count > 0
                    apply_micro_streaks!(img; streaks=MicroStreaks(count=micro_streaks_count))
                end
                sensor_expose!(img; iso=iso, t_exp=t_exp_val,
                               read_noise_e=read_noise,
                               saturation=saturation_val)
                apply_vignette!(img; strength=0.3)
                apply_lens_distortion!(img; k1=-0.02)
                do_auto_balance && auto_balance!(img)
                img = map(clamp01nan, img)
                # The render buffer is [width, height]; rotate so the saved
                # file has the same orientation as the on-screen preview.
                FileIO.save(save_name, rotr90(img))
                put!(result, (:ok, img, save_name))
            catch e
                @error "Photo render failed" draft exception=(e, catch_backtrace())
                put!(result, (:error, e))
            end
        end

        # Thread-1 poller: mirrors the atomic into the progress bar while the
        # workers run, then finishes up when the result lands.
        @async begin
            while !isready(result)
                p = progress_atomic[]
                progress_obs[] = p
                progress_label_obs[] = p >= 1.0 ? "Post-processing…" :
                                       string(round(Int, 100 * p), "%")
                sleep(0.1)
            end
            res = take!(result)
            rendering[] = false
            active_btn.label[] = idle_label
            if res[1] === :ok
                progress_obs[] = 1.0
                progress_label_obs[] = "Saved: $(res[3])"
                final_img = res[2]
                sx = max(1, cld(size(final_img, 1), settings.width))
                sy = max(1, cld(size(final_img, 2), settings.height))
                img_obs[] = final_img[1:sx:end, 1:sy:end]
                if size(img_obs[]) != last_img_size[]
                    last_img_size[] = size(img_obs[])
                    autolimits!(ax)
                end
            else
                progress_obs[] = 0.0
                progress_label_obs[] = "Error: $(res[2])"
            end
        end
    end

    on(render_btn.clicks) do _
        start_photo_render(false)
    end
    on(draft_btn.clicks) do _
        start_photo_render(true)
    end

    # "Save live view": the GPU preview look, exactly as on screen — no
    # post-processing, no sensor model — at the output resolution (the draft
    # tracer supersamples, so it is a clean version of the same image).
    on(live_btn.clicks) do _
        rendering[] && return
        width_val = tryparse(Int, width_tb.stored_string[])
        height_val = tryparse(Int, height_tb.stored_string[])
        if isnothing(width_val) || isnothing(height_val) ||
           width_val <= 0 || height_val <= 0
            progress_label_obs[] = "Error: invalid resolution"
            return
        end
        cam_now = build_camera()
        save_name = "live_" * basename(filename_tb.stored_string[])
        idle_label = live_btn.label[]
        rendering[] = true
        live_btn.label[] = "Rendering…"
        set_progress!(0.0)
        result = Channel{Any}(1)
        Threads.@spawn begin
            try
                img = render_draft_mtl(ctx, cam_now, spacetime;
                                       width=width_val, height=height_val,
                                       samples=2, dt=0.02,
                                       progress=set_progress!)
                img = map(clamp01nan, img)
                FileIO.save(save_name, rotr90(img))
                put!(result, (:ok, img, save_name))
            catch e
                @error "Live-view render failed" exception=(e, catch_backtrace())
                put!(result, (:error, e))
            end
        end
        @async begin
            while !isready(result)
                p = progress_atomic[]
                progress_obs[] = p
                progress_label_obs[] = string(round(Int, 100 * p), "%")
                sleep(0.1)
            end
            res = take!(result)
            rendering[] = false
            live_btn.label[] = idle_label
            if res[1] === :ok
                progress_obs[] = 1.0
                progress_label_obs[] = "Saved: $(res[3])"
                final_img = res[2]
                sx = max(1, cld(size(final_img, 1), settings.width))
                sy = max(1, cld(size(final_img, 2), settings.height))
                img_obs[] = final_img[1:sx:end, 1:sy:end]
                if size(img_obs[]) != last_img_size[]
                    last_img_size[] = size(img_obs[])
                    autolimits!(ax)
                end
            else
                progress_obs[] = 0.0
                progress_label_obs[] = "Error: $(res[2])"
            end
        end
    end

    # -------------------------------------------------------------------------
    # Initial state
    # -------------------------------------------------------------------------
    r_bh = 2.0f0 * Float32(spacetime.M)
    bh_sphere_obs[] = _wireframe_sphere(SVector(0.0, 0.0, 0.0), r_bh)
    if !isnothing(disc)
        disc_rings_obs[] = _disc_wireframe(disc)
    end

    disc_outer = isnothing(disc) ? 10.0 : disc.outer_radius
    cam_dist = Float32(norm(state.pos))
    axis_limit = max(Float32(1.2 * disc_outer), Float32(0.25 * cam_dist),
                     5.0f0 * r_bh)
    center!(scene_3d.scene)
    update_cam!(scene_3d.scene, Vec3f(0.7f0, -0.5f0, 0.5f0) * axis_limit,
                Vec3f(0, 0, 0), Vec3f(0, 0, 1))

    display(fig)
    update_scene_3d!(build_camera())
    request_render()

    # Keyboard fly loop (thread 1, 30 Hz): moves while keys are held and the
    # mouse is over the preview, rendering only when something changed.
    @async while events(fig.scene).window_open[]
        if _Makie.is_mouseinside(ax.scene)
            moved = false
            v = move_speed_obs[] / 30.0
            ispressed(fig, Keyboard.left_shift) && (v *= 5.0)
            lock(state_lock) do
                fwd, right, _ = _flycam_basis(state)
                world_z = SVector(0.0, 0.0, 1.0)
                if ispressed(fig, Keyboard.w)
                    state.pos += v * fwd; moved = true
                end
                if ispressed(fig, Keyboard.s)
                    state.pos -= v * fwd; moved = true
                end
                if ispressed(fig, Keyboard.a)
                    state.pos -= v * right; moved = true
                end
                if ispressed(fig, Keyboard.d)
                    state.pos += v * right; moved = true
                end
                # Q/E move along world z (not camera up), so vertical motion
                # stays vertical regardless of pitch.
                if ispressed(fig, Keyboard.q)
                    state.pos -= v * world_z; moved = true
                end
                if ispressed(fig, Keyboard.e)
                    state.pos += v * world_z; moved = true
                end
                # Keep clear of the singularity (integrator kill radius 0.3M).
                rn = norm(state.pos)
                rn < 0.45 * spacetime.M && (state.pos *= 0.45 * spacetime.M / rn)
            end
            moved && request_render()
        end
        sleep(1 / 30)
    end

    return fig
end
