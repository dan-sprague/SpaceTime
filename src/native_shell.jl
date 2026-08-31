# ---------------------------------------------------------------------------
# Native shell: the flight simulator without Makie
# ---------------------------------------------------------------------------
#
# A bare-metal presentation path for the flythrough: a GLFW window with no GL
# context, a CAMetalLayer attached to its content view, and the preview
# renderer's GPU output packed to BGRA and blitted straight into the layer's
# drawable. The traced frame never leaves the GPU — no host download, no
# Matrix{RGBf} conversion, no scene graph — and the game loop owns input and
# physics directly. This is the shell a shipped build would use; the GLMakie
# apps remain the studio tools.
#
# Everything Objective-C is done through ObjectiveC.jl (already a Metal.jl
# dependency); the window handle comes from GLFW's native-access API.

using GLFW
using ObjectiveC: @objc, id, Object
using Libdl: dlopen
using Metal: MTL

# CGSize for -[CAMetalLayer setDrawableSize:] (two Cdoubles on 64-bit).
struct _CGSize
    width::Cdouble
    height::Cdouble
end

# Pack the renderer's (3, W, H) Float32 output into BGRA8 texture order.
# Row 0 of the texture is the top of the image, which is column j = H of the
# render (the PNG save path applies rotr90 for the same reason). NaN guards:
# clamp propagates NaN, and a checked convert would trap the GPU.
function _pack_bgra_kernel!(dst, src, W, H)
    i = thread_position_in_grid().x
    i > W * H && return
    x = (i - 1) % W + 1
    j = H - (i - 1) ÷ W
    r = clamp(src[1, x, j], 0.0f0, 1.0f0)
    g = clamp(src[2, x, j], 0.0f0, 1.0f0)
    b = clamp(src[3, x, j], 0.0f0, 1.0f0)
    r = r == r ? r : 0.0f0
    g = g == g ? g : 0.0f0
    b = b == b ? b : 0.0f0
    dst[i] = unsafe_trunc(UInt32, r * 255.0f0 + 0.5f0) << 16 |
             unsafe_trunc(UInt32, g * 255.0f0 + 0.5f0) << 8 |
             unsafe_trunc(UInt32, b * 255.0f0 + 0.5f0) | 0xff000000
    return nothing
end

"""
A CAMetalLayer presentation target bound to a GLFW window. `present!` packs a
`(3, W, H)` Float32 GPU array and pushes it to the screen; `nextDrawable`
paces the loop at the display's refresh when frames are cheap.
"""
mutable struct MetalPresenter{Q}
    layer::id{Object}
    queue::Q             # Metal.jl's batched global queue: command buffers
                         # created on it order after batched kernel launches
    pack_gpu::MtlArray{UInt32,1}
    pack_kernel::Base.RefValue{Any}   # compiled-once kernel, like _WARP_KERNEL
    width::Int
    height::Int
end

function MetalPresenter(win::GLFW.Window, width::Int, height::Int)
    dlopen("/System/Library/Frameworks/QuartzCore.framework/QuartzCore")
    nsview = ccall((:glfwGetCocoaView, GLFW.libglfw), Ptr{Cvoid},
                   (Ptr{Cvoid},), win.handle)
    nsview == C_NULL && error("no Cocoa view for the GLFW window")
    view = reinterpret(id{Object}, nsview)
    layer = @objc [CAMetalLayer new]::id{Object}
    dev = Metal.device()
    @objc [layer::id{Object} setDevice:dev::id{MTL.MTLDevice}]::Nothing
    @objc [layer::id{Object} setPixelFormat:UInt64(80)::UInt64]::Nothing  # BGRA8Unorm
    @objc [layer::id{Object} setFramebufferOnly:false::Bool]::Nothing
    @objc [layer::id{Object} setDrawableSize:_CGSize(width, height)::_CGSize]::Nothing
    @objc [view::id{Object} setWantsLayer:true::Bool]::Nothing
    @objc [view::id{Object} setLayer:layer::id{Object}]::Nothing
    queue = Metal.global_queue(dev)
    MetalPresenter{typeof(queue)}(layer, queue,
                                  MtlArray{UInt32}(undef, width * height),
                                  Ref{Any}(nothing), width, height)
end

"""
    present!(p::MetalPresenter, src::MtlArray{Float32,3})

Pack `src` (the renderer's `(3, W, H)` output) to BGRA and blit it into the
next drawable. Blocks until a drawable is free (≈ vsync when saturated).
Command-buffer order on the shared global queue keeps the pack after any
in-flight trace/warp kernels; an explicit flush publishes Metal.jl's batched
launches before ours commits.
"""
function present!(p::MetalPresenter, src::MtlArray{Float32,3})
    W, H = p.width, p.height
    n = W * H
    if p.pack_kernel[] === nothing
        p.pack_kernel[] = @metal launch=false _pack_bgra_kernel!(p.pack_gpu,
                                                                 src, W, H)
    end
    kern = p.pack_kernel[]
    threads = min(kern.pipeline.maxTotalThreadsPerThreadgroup, n)
    kern(p.pack_gpu, src, W, H; threads=threads, groups=cld(n, threads))
    Metal.flush!()
    drawable = @objc [p.layer::id{Object} nextDrawable]::id{Object}
    reinterpret(Ptr{Cvoid}, drawable) == C_NULL && return false
    texptr = @objc [drawable::id{Object} texture]::id{MTL.MTLTexture}
    tex = MTL.MTLTexture(texptr)
    cb = MTL.MTLCommandBuffer(p.queue)
    enc = MTL.MTLBlitCommandEncoder(cb)
    @objc [enc::id{MTL.MTLBlitCommandEncoder} copyFromBuffer:p.pack_gpu.data[]::id{MTL.MTLBuffer}
           sourceOffset:UInt(0)::Csize_t
           sourceBytesPerRow:UInt(4 * W)::Csize_t
           sourceBytesPerImage:UInt(4 * n)::Csize_t
           sourceSize:MTL.MTLSize(W, H, 1)::MTL.MTLSize
           toTexture:tex::id{MTL.MTLTexture}
           destinationSlice:UInt(0)::Csize_t
           destinationLevel:UInt(0)::Csize_t
           destinationOrigin:MTL.MTLOrigin(0, 0, 0)::MTL.MTLOrigin]::Nothing
    close(enc)
    @objc [cb::id{MTL.MTLCommandBuffer} presentDrawable:drawable::id{Object}]::Nothing
    MTL.commit!(cb)
    return true
end

"""
    fly_native(cam, spacetime, background; disc=nothing, volume=nothing,
               width=960, height=540, winwidth=1600, winheight=900,
               title="Spacetime Simulator")

The flight simulator in a native Metal window — no Makie. Renders at
`width × height` and lets Core Animation scale to the window; the traced
frame never leaves the GPU. Same flight model as [`flythrough`](@ref): the
camera is a [`ShipState`](@ref) on a true GR worldline and thrust is proper
acceleration in the ship frame, but the viewport renders from the local
reference observer — ship speed reads out in telemetry, not as aberration.
Runs on the calling (main) thread until the window closes.

Controls: drag to look; W/S A/D Q/E thrust; Space retro-burn; Shift ×4 burn;
Z/C roll; `[`/`]` thrust setting; `-`/`=` time warp; V volumetric gas;
R relativistic shading; L lens (rectilinear/fisheye); X reset ship; Esc quit.
Telemetry lives in the window title.
"""
function fly_native(cam::AbstractCamera, spacetime::Schwarzschild, background;
                    disc::Union{AccretionDisc,Nothing}=nothing,
                    volume::Union{DiscVolume,Nothing}=nothing,
                    width::Int=960, height::Int=540,
                    winwidth::Int=1600, winheight::Int=900,
                    title::String="Spacetime Simulator",
                    max_seconds::Float64=Inf)   # finite for smoke tests
    M = spacetime.M
    ctx = MetalPreviewContext(background, width, height;
                              dt=0.1, nmax=1000, disc=disc, volume=volume)

    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    win = GLFW.CreateWindow(winwidth, winheight, title)
    presenter = MetalPresenter(win, width, height)

    # Flight state (single-threaded: no locks needed).
    state = FlyCamState(cam)
    spawn_pos = SVector{3,Float64}(cam.pos)
    ship = ShipState(spawn_pos, M)
    focal = 24.0
    fisheye = 0.0
    relativistic = false
    thrust = 0.05
    twarp = 2.0
    # The view renders from the local reference observer's frame — the ship's
    # velocity does NOT boost the camera tetrad (no aberration/motion Doppler;
    # only the black hole's lensing). Per Dan: the flight is relativistic,
    # the viewport isn't.
    beta = SVector(0.0, 0.0, 0.0)

    # Reprojection state.
    prev_gpu = MtlArray{Float32}(undef, 3, width, height)
    warp_out = MtlArray{Float32}(undef, 3, width, height)
    warp_params = MtlVector{Float32}(undef, 22)
    prev_cam = Ref{Any}(nothing)
    prev_fe = Ref(0.0)
    prev_beta = Ref(SVector(0.0, 0.0, 0.0))
    last_full = Ref(0.0)
    trace_cost = Ref(0.05)
    last_present = Ref(0.0)

    build_cam() = camera_from_state(state, focal, 2.8, norm(state.pos), false)

    # Warm every kernel (trace variants, warp, pack) before the clock starts:
    # first-call compilation costs seconds and would otherwise hitch the
    # opening frames of the flight.
    let cam0 = build_cam()
        if ctx.has_volume
            set_volume_enabled!(ctx, false)
            _trace_preview_gpu!(ctx, cam0, spacetime)
            set_volume_enabled!(ctx, true)
        end
        _trace_preview_gpu!(ctx, cam0, spacetime)
        copyto!(prev_gpu, ctx.out_gpu)
        _warp_gpu!(ctx, warp_out, prev_gpu, warp_params, cam0, cam0)
        Metal.synchronize()
        present!(presenter, ctx.out_gpu)
    end

    down(k) = GLFW.GetKey(win, k)
    edge = Dict{GLFW.Key,Bool}()
    pressed_once(k) = begin
        now = down(k)
        was = get(edge, k, false)
        edge[k] = now
        now && !was
    end

    dragging = false
    last_mouse = (0.0, 0.0)
    t_start = time()
    last_wall = time()
    last_title = 0.0
    n_traces = 0
    n_warps = 0

    while !GLFW.WindowShouldClose(win) && time() - t_start < max_seconds
        GLFW.PollEvents()
        wall = time()
        dwall = clamp(wall - last_wall, 0.0, 0.1)
        last_wall = wall

        # --- input -----------------------------------------------------
        down(GLFW.KEY_ESCAPE) && GLFW.SetWindowShouldClose(win, true)
        if GLFW.GetMouseButton(win, GLFW.MOUSE_BUTTON_1)
            mp = GLFW.GetCursorPos(win)
            if dragging
                fbw = GLFW.GetFramebufferSize(win).width
                k = 2.0 * (18.0 / focal) / max(fbw, 1)
                state.yaw -= (mp.x - last_mouse[1]) * k
                state.pitch = clamp(state.pitch - (mp.y - last_mouse[2]) * k,
                                    -_PITCH_LIMIT, _PITCH_LIMIT)
            end
            dragging = true
            last_mouse = (mp.x, mp.y)
        else
            dragging = false
        end
        droll = ((down(GLFW.KEY_C) ? 1.0 : 0.0) -
                 (down(GLFW.KEY_Z) ? 1.0 : 0.0)) * 1.5 * dwall
        state.roll += droll
        pressed_once(GLFW.KEY_V) && ctx.has_volume &&
            set_volume_enabled!(ctx, !ctx.vol_on[])
        pressed_once(GLFW.KEY_R) && (relativistic = !relativistic)
        pressed_once(GLFW.KEY_L) && (fisheye = fisheye > 0.0 ? 0.0 : 100.0)
        pressed_once(GLFW.KEY_X) && (ship = ShipState(spawn_pos, M))
        down(GLFW.KEY_LEFT_BRACKET) && (thrust = max(0.005, thrust / 1.03))
        down(GLFW.KEY_RIGHT_BRACKET) && (thrust = min(0.5, thrust * 1.03))
        down(GLFW.KEY_MINUS) && (twarp = max(0.0, twarp - 8.0 * dwall))
        down(GLFW.KEY_EQUAL) && (twarp = min(30.0, twarp + 8.0 * dwall))

        # --- physics (same model as the flythrough ticker) -------------
        dτ = twarp * dwall
        acc = SVector((down(GLFW.KEY_W) ? 1.0 : 0.0) - (down(GLFW.KEY_S) ? 1.0 : 0.0),
                      (down(GLFW.KEY_D) ? 1.0 : 0.0) - (down(GLFW.KEY_A) ? 1.0 : 0.0),
                      (down(GLFW.KEY_E) ? 1.0 : 0.0) - (down(GLFW.KEY_Q) ? 1.0 : 0.0))
        na = norm(acc)
        na > 1.0 && (acc = acc / na)
        amax = thrust * (down(GLFW.KEY_LEFT_SHIFT) ? 4.0 : 1.0)
        fwd, right, _ = _flycam_basis(state)
        upr = _flycam_up(state)
        β, _ = ship_velocity(ship, M, fwd, right, upr)
        sp = norm(β)
        if down(GLFW.KEY_SPACE) && sp > 0.0
            if sp < 1.5 * amax * dτ
                τ0, t0 = ship.τ, ship.t
                ship = ShipState(ship.x, M)
                ship.τ, ship.t = τ0, t0
                acc = SVector(0.0, 0.0, 0.0)
            else
                acc = -β / sp
            end
        end
        a_vec = acc * amax
        a_mag = norm(a_vec)
        if dτ > 0.0
            if a_mag > 0.0
                tetb = ks_camera_tetrad(ship.x, fwd, right, upr, M; beta=β)
                step_ship!(ship, M, dτ; accel=a_vec,
                           axes=(tetb[2], tetb[3], tetb[4]))
            else
                step_ship!(ship, M, dτ)
            end
        end
        norm(ship.x) < 0.5 * M && (ship = ShipState(spawn_pos, M))
        state.pos = ship.x
        β, γ = ship_velocity(ship, M, fwd, right, upr)   # telemetry only
        sp = norm(β)

        # --- render: full trace when stale, warp for pure rotation -----
        cam_now = build_cam()
        compat = prev_cam[] !== nothing && fisheye == prev_fe[] &&
                 norm(beta - prev_beta[]) < 0.02
        fresh = wall - last_full[] < max(0.1, 1.5 * trace_cost[])
        if compat && fresh
            _warp_gpu!(ctx, warp_out, prev_gpu, warp_params, cam_now,
                       prev_cam[]; fisheye_deg=prev_fe[])
            present!(presenter, warp_out)
            last_present[] = time()
            n_warps += 1
        else
            t0 = time()
            on_band = compat ? function ()
                # Keep the display fed during a long trace: warp at most
                # once per ~14 ms, at the freshest look direction.
                time() - last_present[] < 0.014 && return
                GLFW.PollEvents()
                camw = build_cam()
                _warp_gpu!(ctx, warp_out, prev_gpu, warp_params, camw,
                           prev_cam[]; fisheye_deg=prev_fe[])
                present!(presenter, warp_out)
                last_present[] = time()
                return
            end : nothing
            _trace_preview_gpu!(ctx, cam_now, spacetime; fisheye_deg=fisheye,
                                relativistic=relativistic, beta=beta,
                                band_rows=on_band === nothing ? 0 :
                                          cld(height,
                                              clamp(round(Int, trace_cost[] / 0.012),
                                                    4, 24)),
                                on_band=on_band)
            copyto!(prev_gpu, ctx.out_gpu)
            prev_cam[] = cam_now
            prev_fe[] = fisheye
            prev_beta[] = beta
            present!(presenter, ctx.out_gpu)
            last_present[] = time()
            last_full[] = time()
            trace_cost[] = time() - t0
            n_traces += 1
        end

        # --- telemetry in the title bar (cheap, 4 Hz) ------------------
        if wall - last_title > 0.25
            last_title = wall
            r = norm(ship.x)
            regime = r < 2.0 ? "INSIDE HORIZON" : r < 3.0 ? "PHOTON SPHERE" :
                     r < 6.0 ? "BELOW ISCO" : ""
            GLFW.SetWindowTitle(win, @sprintf(
                "%s — r %.2fM %s · β %.3fc γ %.2f · a %s · thr %.2f · warp %.1f · τ %.1fs t %.1fs · trace %.0fms",
                title, r, regime, sp, γ,
                a_mag > 0 ? @sprintf("%.2f", a_mag) : "0 (free fall)",
                thrust, twarp, ship.τ * 0.49255, ship.t * 0.49255,
                1000 * trace_cost[]))
        end
    end
    elapsed = time() - t_start
    @printf("fly_native: %.1f s, %d traces + %d warps = %.1f frames/s presented\n",
            elapsed, n_traces, n_warps, (n_traces + n_warps) / elapsed)
    GLFW.DestroyWindow(win)
    return nothing
end
