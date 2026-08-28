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

"""
    ks_init_photon(origin, direction, M) -> (μ::SVector{6}, p_t)

Null-ray initialisation in Kerr–Schild coordinates: chooses the covariant
momentum so the coordinate velocity at `origin` is exactly the unit
`direction` (closed form; the null condition gives `p_t² = 1 − f(1 − κ²)`).
"""
function ks_init_photon(origin::SVector{3,Float64},
                        direction::SVector{3,Float64}, M::Float64)
    r = norm(origin)
    f = 2M / r
    κd = dot(origin, direction) / r
    p_t = -sqrt(max(1.0 - f * (1.0 - κd^2), 1e-12))
    β = f * (κd - p_t) / max(1.0 - f, 1e-6)
    p = direction + (β / r) * origin
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
    r_horizon = 2.1 * M
    r_escape = settings.r_escape_factor * norm(cam.pos)
    dt = settings.dt
    # Dynamic step cap, kept in sync with render_preview_mtl: rays must be able
    # to reach the escape radius even from distant cameras.
    nmax = min(max(settings.nmax, ceil(Int, 3.0 * r_escape / dt)), 20_000)

    Threads.@threads :static for i in 1:width
        for j in 1:height
            u, v = sensor_coordinate(i, j, width, height)
            origin, direction = get_ray(cam, u, v)
            μ, p_t = ks_init_photon(origin, direction, M)
            hit = false
            for _ in 1:nmax
                r = sqrt(μ[1]^2 + μ[2]^2 + μ[3]^2)
                if r < r_horizon || r > r_escape
                    hit = r < r_horizon
                    if r > r_escape
                        θ = acos(clamp(μ[3] / r, -1.0, 1.0))
                        ϕ = atan(μ[2], μ[1])
                        image[i, j] = sample_background(background, θ, ϕ)
                        hit = true
                    end
                    break
                end
                z_prev = μ[3]
                μ = rk4_step_preview(μ, p_t, M, dt)
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
                # Ran out of steps: sample the sky at the last position.
                r = max(sqrt(μ[1]^2 + μ[2]^2 + μ[3]^2), 1e-12)
                θ = acos(clamp(μ[3] / r, -1.0, 1.0))
                image[i, j] = sample_background(background, θ, atan(μ[2], μ[1]))
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
                        render_preview_mtl(ctx, cam_now, spacetime)
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

    Label(cam_col[crow, 1], status_obs; halign=:left)
    crow += 1
    Label(cam_col[crow, 1], cam_pos_label_obs; halign=:left)
    crow += 1
    Label(cam_col[crow, 1],
          "Drag: look · Scroll: dolly · WASD: move · Q/E: down/up (z) · Shift: fast";
          halign=:left, color=:gray)
    rowgap!(cam_col, 6)

    # --- Column 2: post-processing ---
    post_col = GridLayout(controls[1, 2]; valign=:top, tellheight=false)
    Label(post_col[1, 1], "Post-processing"; fontsize=16, halign=:left)
    post_sg = SliderGrid(
        post_col[2, 1],
        (label = "Gain", range = 0.0:0.01:2.0, format = "{:.2f}", startvalue = 1.0),
        (label = "Exposure (EV)", range = -5.0:0.1:5.0, format = "{:.1f}", startvalue = 0.0),
        (label = "Gamma", range = 0.1:0.05:3.0, format = "{:.2f}", startvalue = 2.2),
        (label = "Bloom strength", range = 0.0:0.05:2.0, format = "{:.2f}", startvalue = 0.6),
        (label = "Bloom threshold", range = 0.0:0.05:2.0, format = "{:.2f}", startvalue = 0.5),
        (label = "Bloom radius", range = 1.0:1.0:50.0, format = "{:.0f}", startvalue = 15.0),
        (label = "Bloom power", range = 0.1:0.1:3.0, format = "{:.1f}", startvalue = 1.5),
        (label = "Streak strength", range = 0.0:0.05:2.0, format = "{:.2f}", startvalue = 0.3),
        (label = "Streak length", range = 0.05:0.05:1.0, format = "{:.2f}", startvalue = 0.4),
        (label = "Streak width", range = 0.5:0.5:5.0, format = "{:.1f}", startvalue = 1.5),
        (label = "Star spikes", range = 2:1:8, format = "{:.0f}", startvalue = 4),
        (label = "Color preserve", range = 0.0:0.05:1.0, format = "{:.2f}", startvalue = 0.75),
        (label = "Contrast", range = -1.0:0.05:1.0, format = "{:.2f}", startvalue = 0.0),
        tellwidth = false, tellheight = true
    )
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

    settings_grid = GridLayout(out_col[orow, 1])
    Label(settings_grid[1, 1], "Width"; halign=:left)
    width_tb = Textbox(settings_grid[1, 2]; stored_string="3840", validator=Int,
                       tellwidth=false)
    Label(settings_grid[2, 1], "Height"; halign=:left)
    height_tb = Textbox(settings_grid[2, 2]; stored_string="2160", validator=Int,
                        tellwidth=false)
    Label(settings_grid[3, 1], "Samples"; halign=:left)
    samples_tb = Textbox(settings_grid[3, 2]; stored_string="4", validator=Int,
                         tellwidth=false)
    Label(settings_grid[4, 1], "File"; halign=:left)
    filename_tb = Textbox(settings_grid[4, 2]; stored_string="render.png",
                          tellwidth=false)
    colsize!(settings_grid, 1, GLMakie.Auto())
    colsize!(settings_grid, 2, GLMakie.Relative(0.55))
    rowgap!(settings_grid, 4)
    orow += 1

    final_btn_grid = GridLayout(out_col[orow, 1]; halign=:left)
    render_btn = Button(final_btn_grid[1, 1]; label="Render final image")
    draft_btn = Button(final_btn_grid[1, 2]; label="GPU draft")
    colgap!(final_btn_grid, 8)
    orow += 1

    # Progress bar: a decoration-free mini axis with a filled rectangle.
    progress_obs = Observable(0.0)
    progress_label_obs = Observable("Ready")
    pax = GLMakie.Axis(out_col[orow, 1]; height=14, limits=(0, 1, 0, 1),
                       backgroundcolor=RGBf(0.15, 0.15, 0.15))
    hidedecorations!(pax)
    hidespines!(pax)
    for k in (:rectanglezoom, :dragpan, :scrollzoom, :limitreset)
        deregister_interaction!(pax, k)
    end
    poly!(pax, @lift(Rect2f(0.0, 0.0, max($progress_obs, 1e-4), 1.0));
          color=:seagreen)
    orow += 1

    Label(out_col[orow, 1], progress_label_obs; halign=:left)
    rowgap!(out_col, 6)

    colsize!(controls, 1, GLMakie.Relative(0.29))
    colsize!(controls, 2, GLMakie.Relative(0.42))
    colsize!(controls, 3, GLMakie.Relative(0.29))
    colgap!(controls, 20)

    # -------------------------------------------------------------------------
    # Hectic preset: the demonstration.jl "COOL SCENE" composition
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
        for (sl, val) in zip(post_sg.sliders,
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
        post_sliders = post_sg.sliders
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
            end
            moved && request_render()
        end
        sleep(1 / 30)
    end

    return fig
end
