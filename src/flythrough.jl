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
                        MtlVector{Float32}(undef, 28),
                        MtlVector{Float32}(undef, 4),
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
    # Swappable lens: 0 = rectilinear (focal slider applies); > 0 = equidistant
    # fisheye with that vertical half-angle. Fisheye is what frames the whole
    # escape porthole — its angular radius never drops below ~80° on the dive,
    # beyond any rectilinear focal length.
    fisheye_obs = Observable(0.0)
    # Relativistic shading: gravitational blueshift + Doppler of the camera
    # applied to the sky (Planck-locus tint + brightness) and disc.
    rel_obs = Observable(false)
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

    # Live fluid disc (Tier-2 turbulence): stepped by the render worker only,
    # keeping all Metal work off thread 1. `live_flag` is set by the UI
    # toggle; sim time advances with wall time at `SIM_RATE` M per second
    # (inner-edge orbit ≈ 33 M, so one lap every ~33 s at rate 1).
    fluid_sim = (isnothing(volume) || isnothing(disc)) ? nothing :
                DiscFluidSim(volume, disc; M=spacetime.M)
    live_flag = Ref(false)
    last_sim_t = Ref(time())
    SIM_RATE = 1.0

    # Asynchronous reprojection: between full traces, re-display the last
    # traced frame warped by the camera rotation since it was traced (exact
    # for rotation; translation error is corrected by the next full trace,
    # forced at least every FULL_TRACE_PERIOD).
    reproj_flag = Ref(false)
    FULL_TRACE_PERIOD = 0.1

    function request_render()
        Threads.atomic_add!(render_version, 1)
        isready(wakeup) || put!(wakeup, nothing)
        return nothing
    end

    Threads.@spawn begin
        done = 0
        # Per-resolution ping-pong buffers. Reusing them removes the ~8 MB of
        # per-frame garbage that caused GC stutter at flight frame rates; two
        # images alternate so the GL texture upload never races the next
        # frame's write.
        BufT = Tuple{Array{Float32,3},Matrix{RGBf},Matrix{RGBf}}
        bufs = Dict{Tuple{Int,Int},BufT}()
        # Reprojection state: retained copy of the last traced frame plus the
        # camera/lens it was traced with, and the warp scratch buffers.
        WBufT = Tuple{MtlArray{Float32,3},MtlArray{Float32,3},MtlVector{Float32}}
        wbufs = Dict{Tuple{Int,Int},WBufT}()
        prev_cam = Ref{Any}(nothing)
        prev_fe = Ref(0.0)
        prev_dims = Ref((0, 0))
        last_full = Ref(0.0)
        flip = false
        min_period = 1 / 40   # rendering faster than this only floods thread 1
        warp_period = 1 / 60  # warped frames are cheap; let them run faster
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
                    fe_now = fisheye_obs[]
                    dims = (ctx_now.width, ctx_now.height)
                    # Warp when reprojection is on, the retained frame is
                    # fresh (< FULL_TRACE_PERIOD) and compatible (same lens
                    # and resolution); otherwise trace and retain.
                    warped = reproj_flag[] && prev_cam[] !== nothing &&
                             t0 - last_full[] < FULL_TRACE_PERIOD &&
                             fe_now == prev_fe[] && dims == prev_dims[]
                    img = try
                        if !warped && live_flag[] && !isnothing(fluid_sim) &&
                           ctx_now.vol_on[]
                            wall = time()
                            # Step at ≤25 Hz: a sim step costs ~20 ms of
                            # dispatch overhead, so per-frame stepping would
                            # halve the preview rate for invisible gains.
                            if wall - last_sim_t[] >= 0.04
                                dt_sim = clamp(wall - last_sim_t[], 0.0, 0.15) * SIM_RATE
                                last_sim_t[] = wall
                                step_sim!(fluid_sim, ctx_now; dt=dt_sim)
                            end
                        end
                        host, ia, ib = get!(bufs, dims) do
                            (Array{Float32,3}(undef, 3, ctx_now.width, ctx_now.height),
                             Matrix{RGBf}(undef, ctx_now.width, ctx_now.height),
                             Matrix{RGBf}(undef, ctx_now.width, ctx_now.height))
                        end
                        flip = !flip
                        if warped
                            prev_gpu, wout, wpar = wbufs[dims]
                            warp_preview_mtl!(flip ? ia : ib, host, ctx_now,
                                              wout, prev_gpu, wpar, cam_now,
                                              prev_cam[]; fisheye_deg=fe_now)
                        else
                            frame = render_preview_mtl!(flip ? ia : ib, host,
                                                        ctx_now, cam_now,
                                                        spacetime;
                                                        fisheye_deg=fe_now,
                                                        relativistic=rel_obs[])
                            if reproj_flag[]
                                prev_gpu, _, _ = get!(wbufs, dims) do
                                    (MtlArray{Float32}(undef, 3, dims...),
                                     MtlArray{Float32}(undef, 3, dims...),
                                     MtlVector{Float32}(undef, 22))
                                end
                                copyto!(prev_gpu, ctx_now.out_gpu)
                                prev_cam[] = cam_now
                                prev_fe[] = fe_now
                                prev_dims[] = dims
                            end
                            last_full[] = time()
                            frame
                        end
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
                    elapsed = time() - t0
                    period = warped ? warp_period : min_period
                    elapsed < period && sleep(period - elapsed)
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

    disc_on = Ref(true)   # master disc switch state, shared by both toggles
    local vol_toggle = nothing
    if !isnothing(volume)
        vol_toggle = Toggle(bar[1, 9]; active=true)
        Label(bar[1, 10], "Volumetric"; halign=:left)
        on(vol_toggle.active) do a
            set_volume_enabled!(base_ctx, a && disc_on[])   # vol_on shared
            request_render()
        end
    end

    Label(bar[1, 11], "Lens"; halign=:right)
    lens_menu = Menu(bar[1, 12];
                     options=[("Rectilinear", 0.0),
                              ("Fisheye 180°", 90.0),
                              ("Fisheye 235°", 117.5)],
                     default="Rectilinear", width=140)
    on(lens_menu.selection) do deg
        isnothing(deg) && return
        fisheye_obs[] = deg
        request_render()
        return nothing
    end

    # Master disc switch: kills both the thin plane and (when off) the
    # volumetric gas, leaving the pure lensed starfield. Re-enabling
    # restores the volumetric toggle's own state.
    if !isnothing(disc)
        disc_toggle = Toggle(bar[1, 13]; active=true)
        Label(bar[1, 14], "Disc"; halign=:left)
        on(disc_toggle.active) do a
            disc_on[] = a
            set_disc_enabled!(base_ctx, disc, a)
            vol_now = a && !isnothing(vol_toggle) && vol_toggle.active[]
            set_volume_enabled!(base_ctx, vol_now)
            request_render()
        end
    end

    # Live fluid disc: a stable-fluids solver on the disc grid, stepped by
    # the render worker (Metal work stays off thread 1). The ticker below
    # keeps frames coming while the camera is still.
    if !isnothing(fluid_sim)
        live_toggle = Toggle(bar[1, 15]; active=false)
        Label(bar[1, 16], "Live gas"; halign=:left)
        on(live_toggle.active) do a
            live_flag[] = a
            request_render()
        end
        # Ticker: keep frames (and sim steps) coming while the camera rests.
        @async while events(fig.scene).window_open[]
            live_flag[] && base_ctx.vol_on[] && request_render()
            sleep(1 / 24)
        end
    end

    # Render row: trace-time camera options (depth of field cannot be added
    # in post) plus the two offline renderers. Both save 4K linear HDR raws
    # (32-bit float TIFF + TOML sidecar) for the standalone post app; the
    # live preview keeps running while a render traces.
    Label(bar[2, 1], "Thin lens"; halign=:right)
    tl_toggle = Toggle(bar[2, 2]; active=false)
    Label(bar[2, 3], "f/"; halign=:right)
    fnum_sl = Slider(bar[2, 4]; range=1.0:0.1:22.0, startvalue=2.8, width=110)
    Label(bar[2, 5], "Focus (M)"; halign=:right)
    focus_sl = Slider(bar[2, 6]; range=0.5:0.5:60.0, startvalue=27.0, width=110)
    gpu_btn = Button(bar[2, 7]; label="GPU render")
    cpu_btn = Button(bar[2, 8]; label="CPU render")
    rel_toggle = Toggle(bar[2, 9]; active=false)
    Label(bar[2, 10], "Rel. shading"; halign=:left)
    on(rel_toggle.active) do a
        rel_obs[] = a
        request_render()
    end

    # Asynchronous reprojection: cheap rotation-warped frames between full
    # traces (VR "timewarp") for high-refresh mouse-look.
    reproj_toggle = Toggle(bar[2, 11]; active=false)
    Label(bar[2, 12], "Reproject"; halign=:left)
    on(reproj_toggle.active) do a
        reproj_flag[] = a
        request_render()
    end

    rendering = Ref(false)
    function render_metadata(cam_now, fe, ap, extra)
        merge!(Dict{String,Any}(
            "camera_pos" => collect(cam_now.pos),
            "camera_fwd" => collect(cam_now.fwd),
            "camera_up" => collect(cam_now.up_local),
            "fov_factor" => cam_now.fov_factor,
            "fisheye_deg" => fe, "M" => spacetime.M,
            "aperture_world" => ap, "focus_dist" => focus_sl.value[],
            "disc" => disc_on[], "volumetric" => base_ctx.vol_on[]), extra)
    end

    on(gpu_btn.clicks) do _
        rendering[] && return
        rendering[] = true
        cam_now = build_camera()
        fe = fisheye_obs[]
        ap = tl_toggle.active[] ? focus_sl.value[] / fnum_sl.value[] : 0.0
        tl_toggle.active[] && fe > 0.0 &&
            (status_obs[] = "Note: DoF is ignored with a fisheye lens")
        # With DoF each supersampling pass is one bokeh sample, so trace more.
        smp = (ap > 0.0 && fe <= 0.0) ? 4 : 2
        focusd = focus_sl.value[]
        fname = "flyraw_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS") * ".tiff"
        gpu_btn.label[] = "Tracing…"
        result = Channel{Any}(1)
        Threads.@spawn begin
            try
                img = render_draft_mtl(base_ctx, cam_now, spacetime;
                                       width=3840, height=2160, samples=smp,
                                       dt=0.02, fisheye_deg=fe,
                                       aperture_world=ap, focus_dist=focusd,
                                       relativistic=rel_obs[])
                save_raw(fname, img; metadata=render_metadata(cam_now, fe, ap,
                    Dict{String,Any}("renderer" => "gpu_draft",
                                     "width" => 3840, "height" => 2160,
                                     "samples" => smp, "dt" => 0.02,
                                     "relativistic" => rel_obs[])))
                put!(result, (:ok, fname))
            catch e
                @error "GPU render failed" exception=(e, catch_backtrace())
                put!(result, (:error, e))
            end
        end
        @async begin
            res = take!(result)
            rendering[] = false
            gpu_btn.label[] = "GPU render"
            status_obs[] = res[1] === :ok ?
                "Saved: $(res[2]) (+ .toml) — grade with examples/postprocess_demo.jl" :
                "GPU render error (see terminal)"
        end
    end

    # CPU reference render: full Float64 adaptive integration with dust-free
    # scene as configured. Spherical chart + static camera: exterior,
    # rectilinear only.
    cpu_progress = Threads.Atomic{Float64}(0.0)
    on(cpu_btn.clicks) do _
        rendering[] && return
        fisheye_obs[] > 0.0 &&
            (status_obs[] = "CPU renderer is rectilinear only — switch Lens"; return)
        r_cam = lock(state_lock) do
            norm(state.pos)
        end
        r_cam < 2.6 * M &&
            (status_obs[] = "CPU renderer needs r > 2.6M (spherical chart)"; return)
        rendering[] = true
        cam_cpu = lock(state_lock) do
            camera_from_state(state, focal_obs[], fnum_sl.value[],
                              focus_sl.value[], tl_toggle.active[])
        end
        ap = tl_toggle.active[] ? focus_sl.value[] / fnum_sl.value[] : 0.0
        disc_now = disc_on[] ? disc : nothing
        vol_now = base_ctx.vol_on[] ? volume : nothing
        fname = "flyraw_cpu_" * Dates.format(Dates.now(), "yyyymmdd_HHMMSS") * ".tiff"
        cpu_btn.label[] = "Tracing…"
        Threads.atomic_xchg!(cpu_progress, 0.0)
        result = Channel{Any}(1)
        Threads.@spawn begin
            try
                img = render(cam_cpu, spacetime, background;
                             disc=disc_now, volume=vol_now,
                             width=1920, height=1080, samples=2,
                             relativistic=rel_obs[],
                             progress=p -> (Threads.atomic_xchg!(cpu_progress,
                                                                 Float64(p)); nothing))
                save_raw(fname, img; metadata=render_metadata(cam_cpu, 0.0, ap,
                    Dict{String,Any}("renderer" => "cpu",
                                     "width" => 1920, "height" => 1080,
                                     "samples" => 2,
                                     "relativistic" => rel_obs[])))
                put!(result, (:ok, fname))
            catch e
                @error "CPU render failed" exception=(e, catch_backtrace())
                put!(result, (:error, e))
            end
        end
        @async begin
            while !isready(result)
                status_obs[] = string("CPU render: ",
                                      round(Int, 100 * cpu_progress[]), "%")
                sleep(0.25)
            end
            res = take!(result)
            rendering[] = false
            cpu_btn.label[] = "CPU render"
            status_obs[] = res[1] === :ok ?
                "Saved: $(res[2]) (+ .toml) — grade with examples/postprocess_demo.jl" :
                "CPU render error (see terminal)"
        end
    end

    Label(bar[3, 1:8], status_obs; halign=:left, fontsize=13, color=:gray)
    Label(bar[4, 1:8],
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
    # mouse is over the view, rendering only when something changed. The
    # try/catch keeps a transient error (e.g. during window teardown) from
    # silently killing the loop.
    @async while events(fig.scene).window_open[]
        try
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
        catch e
            @error "Flythrough key loop error" exception=(e, catch_backtrace())
        end
        sleep(1 / 30)
    end

    return fig
end
