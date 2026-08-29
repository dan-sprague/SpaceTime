# ---------------------------------------------------------------------------
# Flythrough: a minimal real-time flight app around the black hole
# ---------------------------------------------------------------------------

# (label, value) tuples: Makie's Menu labels Pairs by their full string form,
# so tuples are required for the labels to render correctly.
const _FLY_RESOLUTIONS = [
    ("480 × 270", (480, 270)),
    ("640 × 360", (640, 360)),
    ("960 × 540", (960, 540)),
    ("1280 × 720", (1280, 720)),
    ("1920 × 1080", (1920, 1080)),
]

"""
Build a `MetalPreviewContext` at a new resolution that shares the heavy GPU
state (background texture, blackbody LUT, volume grid, volume params and the
compiled-kernel cache) with `base`, allocating only the per-resolution output
and parameter buffers. Sharing `vol_on` means the volumetric toggle applies
across every resolution; sharing `kernel` means switching resolution never
recompiles.
"""
function _fly_resolution_ctx(base::MetalPreviewContext, width::Int, height::Int)
    MetalPreviewContext(base.bg_gpu,
                        MtlArray{Float32,3}(undef, 3, width, height),
                        MtlVector{Float32}(undef, 20),
                        MtlVector{Float32}(undef, 3),
                        base.disc_params, base.bb_lut, base.vol_gpu,
                        base.vol_params, width, height, base.dt, base.nmax,
                        base.r_escape_factor, base.has_volume, base.vol_on,
                        base.kernel)
end

"""Human-readable regime tag for a camera at radius `r` (in units of M)."""
function _fly_regime(r::Float64, M::Float64)
    r < 2.0 * M && return "  ·  INSIDE THE HORIZON"
    r < 3.0 * M && return "  ·  inside the photon sphere"
    r < 6.0 * M && return "  ·  below the ISCO"
    return ""
end

"""
    flythrough(cam, spacetime, background; disc=nothing, volume=nothing,
               settings=PreviewSettings(width=960, height=540),
               title="Black Hole Flythrough")

A minimal real-time flight app: one full-window ray-traced view of the black
hole (Metal GPU, Kerr–Schild geodesics) and a thin control bar. No photo
controls — the only purpose is to fly.

- **Drag** to look, **scroll** to dolly, **W/A/S/D** to move, **Q/E** for
  world-vertical, **Z/C** to roll, **Shift** for 5× speed. Keys act while the
  mouse is over the view.
- **Resolution** dropdown switches the render size live (contexts share the
  GPU background/volume, so switching is instant).
- **Auto speed** scales movement with altitude above the horizon, so the
  approach slows as you dive toward — and inside — the photon sphere.
- The readout shows `r` in units of M and flags horizon / photon-sphere /
  ISCO crossings.

Launch with `julia -t auto,1 --project` so the render worker has its own
thread. Returns the `Figure`.
"""
function flythrough(cam::AbstractCamera, spacetime::Schwarzschild, background;
                    disc::Union{AccretionDisc,Nothing}=nothing,
                    volume::Union{DiscVolume,Nothing}=nothing,
                    settings::PreviewSettings=PreviewSettings(width=960, height=540),
                    title::String="Black Hole Flythrough")
    if Threads.nthreads(:interactive) == 0 && Threads.nthreads(:default) == 1
        @warn """Single-threaded launch: the GPU worker will share thread 1 \
        with the UI. Launch with `julia -t auto,1` for the smoothest flight."""
    end
    M = spacetime.M
    fig = Figure(size=(1280, 800))
    ax = GLMakie.Axis(fig[1, 1], aspect=DataAspect(), title=title)
    img_obs = Observable(zeros(RGBf, settings.width, settings.height))
    image!(ax, img_obs)
    hidedecorations!(ax)
    for k in (:rectanglezoom, :dragpan, :scrollzoom, :limitreset)
        deregister_interaction!(ax, k)
    end
    rowsize!(fig.layout, 1, GLMakie.Auto(true))

    # ------------------------------------------------------------------
    # Camera state and Metal contexts (one per resolution, heavy state shared)
    # ------------------------------------------------------------------
    state = FlyCamState(cam)
    state_lock = ReentrantLock()
    focal_obs = Observable(24.0)
    move_speed_obs = Observable(2.0)
    auto_speed_obs = Observable(true)
    status_obs = Observable("Starting Metal preview…")

    base_ctx = MetalPreviewContext(background, settings.width, settings.height;
                                   dt=settings.dt, nmax=settings.nmax,
                                   r_escape_factor=settings.r_escape_factor,
                                   disc=disc, volume=volume)
    ctx_cache = Dict{Tuple{Int,Int},MetalPreviewContext}(
        (settings.width, settings.height) => base_ctx)
    ctx_ref = Ref{MetalPreviewContext}(base_ctx)

    build_camera() = lock(state_lock) do
        camera_from_state(state, focal_obs[], 2.8, norm(state.pos), false)
    end

    # Pre-compile both kernel variants (the cache is shared by every
    # resolution) so the first volumetric-toggle flip doesn't hitch mid-flight.
    if base_ctx.has_volume
        set_volume_enabled!(base_ctx, false)
        render_preview_mtl(base_ctx, build_camera(), spacetime)
        set_volume_enabled!(base_ctx, true)
    end
    render_preview_mtl(base_ctx, build_camera(), spacetime)

    # Latest-wins request coalescing (same pattern as the viewfinder): the UI
    # bumps a version and pokes the worker; the worker re-renders until it has
    # caught up, publishing only the newest frame.
    render_version = Threads.Atomic{Int}(0)
    wakeup = Channel{Nothing}(1)
    img_chan = Channel{Tuple{Matrix{RGBf},Float64,Float64}}(1)

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
                    ctx_now = lock(state_lock) do
                        ctx_ref[]
                    end
                    t0 = time()
                    img = try
                        render_preview_mtl(ctx_now, cam_now, spacetime)
                    catch e
                        e isa InvalidStateException && rethrow()
                        @error "Flythrough frame failed" exception=(e, catch_backtrace())
                        nothing
                    end
                    done = v
                    if img !== nothing
                        while isready(img_chan)
                            take!(img_chan)
                        end
                        put!(img_chan, (img, time() - t0, norm(cam_now.pos)))
                    end
                end
            end
        catch e
            e isa InvalidStateException || rethrow()   # channel closed: exit
        end
    end

    # Thread-1 drainer: the only writer of frames into observables.
    last_img_size = Ref((settings.width, settings.height))
    @async try
        for (img, elapsed, r) in img_chan
            img_obs[] = img
            if size(img) != last_img_size[]
                last_img_size[] = size(img)
                autolimits!(ax)
            end
            status_obs[] = string(round(1000 * elapsed; digits=1), " ms · ",
                                  round(1 / max(elapsed, 1e-6); digits=1),
                                  " fps · r = ", round(r / M; digits=2), "M",
                                  _fly_regime(r, M))
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

    # ------------------------------------------------------------------
    # Fly-cam input
    # ------------------------------------------------------------------
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
        k = 2.0 * (18.0 / focal_obs[]) / pxw
        lock(state_lock) do
            state.yaw -= δ[1] * k
            state.pitch = clamp(state.pitch + δ[2] * k,
                                -_PITCH_LIMIT, _PITCH_LIMIT)
        end
        request_render()
        return _Makie.Consume(true)
    end

    on(events(ax.scene).scroll) do sc
        (_Makie.is_mouseinside(ax.scene) && sc[2] != 0) || return _Makie.Consume(false)
        lock(state_lock) do
            fwd = _flycam_basis(state)[1]
            step = 0.05 * sc[2] * max(norm(state.pos) - 1.9 * M, 0.2)
            state.pos += step * fwd
            rn = norm(state.pos)
            rn < 0.45 * M && (state.pos *= 0.45 * M / rn)
        end
        request_render()
        return _Makie.Consume(true)
    end

    # ------------------------------------------------------------------
    # Control bar
    # ------------------------------------------------------------------
    bar = GridLayout(fig[2, 1]; tellwidth=false, tellheight=true)

    Label(bar[1, 1], "Resolution"; halign=:right)
    res_menu = Menu(bar[1, 2]; options=_FLY_RESOLUTIONS,
                    default="960 × 540", width=130)

    Label(bar[1, 3], "Speed"; halign=:right)
    speed_sl = Slider(bar[1, 4]; range=0.1:0.1:20.0,
                      startvalue=move_speed_obs[], width=110)
    connect!(move_speed_obs, speed_sl.value)

    auto_toggle = Toggle(bar[1, 5]; active=auto_speed_obs[])
    Label(bar[1, 6], "Auto speed"; halign=:left)
    connect!(auto_speed_obs, auto_toggle.active)

    Label(bar[1, 7], "Focal (mm)"; halign=:right)
    focal_sl = Slider(bar[1, 8]; range=10.0:1.0:100.0,
                      startvalue=focal_obs[], width=110)
    on(focal_sl.value) do f
        focal_obs[] = f
        request_render()
    end

    local vol_toggle = nothing
    if !isnothing(volume)
        vol_toggle = Toggle(bar[1, 9]; active=true)
        Label(bar[1, 10], "Volumetric"; halign=:left)
        on(vol_toggle.active) do a
            set_volume_enabled!(base_ctx, a)   # vol_on is shared by every ctx
            request_render()
        end
    end

    Label(bar[2, 1:8], status_obs; halign=:left, fontsize=13, color=:gray)
    Label(bar[3, 1:8],
          "Drag: look · Scroll: dolly · WASD: move · Q/E: down/up · Z/C: roll · Shift: fast";
          halign=:left, fontsize=12, color=:gray)

    on(res_menu.selection) do res
        isnothing(res) && return
        w, h = res
        ctx_new = get!(ctx_cache, (w, h)) do
            _fly_resolution_ctx(base_ctx, w, h)
        end
        lock(state_lock) do
            ctx_ref[] = ctx_new
        end
        request_render()
        return nothing
    end

    display(fig)
    request_render()

    # Keyboard fly loop (thread 1, 30 Hz): moves while keys are held and the
    # mouse is over the view, rendering only when something changed.
    @async while events(fig.scene).window_open[]
        if _Makie.is_mouseinside(ax.scene)
            moved = false
            v = move_speed_obs[] / 30.0
            ispressed(fig, Keyboard.left_shift) && (v *= 5.0)
            lock(state_lock) do
                # Auto speed: scale with altitude above the horizon so the
                # dive through the photon sphere is flyable, not a teleport;
                # inside the horizon scale with radius instead, so motion
                # never freezes.
                if auto_speed_obs[]
                    rn = norm(state.pos)
                    v *= clamp(0.12 * max(rn - 1.9 * M, 0.25 * rn), 0.02, 8.0)
                end
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
                if ispressed(fig, Keyboard.q)
                    state.pos -= v * world_z; moved = true
                end
                if ispressed(fig, Keyboard.e)
                    state.pos += v * world_z; moved = true
                end
                if ispressed(fig, Keyboard.z)
                    state.roll -= 1.5 / 30; moved = true
                end
                if ispressed(fig, Keyboard.c)
                    state.roll += 1.5 / 30; moved = true
                end
                # Keep clear of the singularity (the integrator's safety
                # kill radius is 0.3M).
                rn = norm(state.pos)
                rn < 0.45 * M && (state.pos *= 0.45 * M / rn)
            end
            moved && request_render()
        end
        sleep(1 / 30)
    end

    return fig
end
