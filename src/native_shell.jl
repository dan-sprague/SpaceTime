# ---------------------------------------------------------------------------
# Native shell: the simulator without Makie
# ---------------------------------------------------------------------------
#
# A bare-metal presentation path: a GLFW window with no GL context, a
# CAMetalLayer attached to its content view, and the renderer's GPU output
# packed to BGRA and blitted straight into the layer's drawable. The frame
# never leaves the GPU — no host download, no Matrix{RGBf} conversion, no
# scene graph — and the game loop owns input and physics directly. This is
# the shell a shipped build would use; the GLMakie apps remain the studio
# tools.
#
# Rendering uses the layered engine (`update_sky_fan!` +
# `render_layered_gpu!`): an exact per-frame deflection fan turns every sky
# pixel into a table lookup, and only the disc/gas pays for per-pixel
# geodesic integration, at reduced resolution.
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

# Pack the renderer's (3, W, H) Float32 output into BGRA8 texture order,
# through the display transform: exposure, an ACES-style filmic curve (tames
# the disc's linear-radiance blowout), and sRGB-ish gamma. Row 0 of the
# texture is the top of the image, which is column j = H of the render (the
# PNG save path applies rotr90 for the same reason). NaN guards: clamp
# propagates NaN, and a checked convert would trap the GPU.
function _pack_bgra_kernel!(dst, src, W, H, exposure, filmic)
    i = thread_position_in_grid().x
    i > W * H && return
    x = (i - 1) % W + 1
    j = H - (i - 1) ÷ W
    r = src[1, x, j] * exposure
    g = src[2, x, j] * exposure
    b = src[3, x, j] * exposure
    r = r == r ? r : 0.0f0
    g = g == g ? g : 0.0f0
    b = b == b ? b : 0.0f0
    if filmic > 0.5f0
        r = (r * (2.51f0 * r + 0.03f0)) / (r * (2.43f0 * r + 0.59f0) + 0.14f0)
        g = (g * (2.51f0 * g + 0.03f0)) / (g * (2.43f0 * g + 0.59f0) + 0.14f0)
        b = (b * (2.51f0 * b + 0.03f0)) / (b * (2.43f0 * b + 0.59f0) + 0.14f0)
        r = exp(log(max(r, 1.0f-6)) * 0.454545f0)
        g = exp(log(max(g, 1.0f-6)) * 0.454545f0)
        b = exp(log(max(b, 1.0f-6)) * 0.454545f0)
    end
    r = clamp(r, 0.0f0, 1.0f0)
    g = clamp(g, 0.0f0, 1.0f0)
    b = clamp(b, 0.0f0, 1.0f0)
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
in-flight render kernels; an explicit flush publishes Metal.jl's batched
launches before ours commits.
"""
function present!(p::MetalPresenter, src::MtlArray{Float32,3};
                  exposure::Float32=1.0f0, filmic::Bool=true)
    W, H = p.width, p.height
    n = W * H
    if p.pack_kernel[] === nothing
        p.pack_kernel[] = @metal launch=false _pack_bgra_kernel!(
            p.pack_gpu, src, W, H, exposure, filmic ? 1.0f0 : 0.0f0)
    end
    kern = p.pack_kernel[]
    threads = min(kern.pipeline.maxTotalThreadsPerThreadgroup, n)
    kern(p.pack_gpu, src, W, H, exposure, filmic ? 1.0f0 : 0.0f0;
         threads=threads, groups=cld(n, threads))
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
               fan_n=4096, title="Spacetime Simulator")

The simulator in a native Metal window — no Makie. Every frame: an exact
deflection fan (`fan_n` RK4 geodesics for the current radius) drives the
lensed sky and shadow at native resolution; the disc/gas layer renders
fresh at half display resolution while moving (quarter under load) and at
full resolution at rest; the composite is presented without ever leaving
the GPU.

The default camera is an **omnipotent free camera**: W/S A/D move
forward/right, Q/E move along world-vertical, all at flat velocity (`[`/`]`
speed, Shift ×5, speed auto-scales with altitude). Press **F** to hand the
camera to the GR ship instead ([`ShipState`](@ref)): thrust, free fall,
retro-burn (Space), time warp (`-`/`=`) — the viewport still renders from
the local reference observer.

Other keys: drag to look; Z/C roll; V volumetric gas; R relativistic
shading; L lens (rectilinear/fisheye); T/G exposure; P filmic display
transform on/off; X reset position; Esc quit. When the camera rests, one
full-native-resolution gas pass renders (stills are sharp) and the GPU
parks until something changes.
Telemetry lives in the window title. Runs on the calling (main) thread
until the window closes.
"""
function fly_native(cam::AbstractCamera, spacetime::Schwarzschild, background;
                    disc::Union{AccretionDisc,Nothing}=nothing,
                    volume::Union{DiscVolume,Nothing}=nothing,
                    width::Int=960, height::Int=540,
                    winwidth::Int=1600, winheight::Int=900,
                    fan_n::Int=4096,
                    title::String="Spacetime Simulator",
                    max_seconds::Float64=Inf)   # finite for smoke tests
    M = spacetime.M
    ctx = MetalPreviewContext(background, width, height;
                              dt=0.1, nmax=1000, disc=disc, volume=volume)

    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    win = GLFW.CreateWindow(winwidth, winheight, title)
    presenter = MetalPresenter(win, width, height)

    # Camera state (single-threaded: no locks needed).
    state = FlyCamState(cam)
    spawn_pos = SVector{3,Float64}(cam.pos)
    ship = ShipState(spawn_pos, M)
    flight = false          # F toggles the GR ship; default is the free cam
    focal = 24.0
    fisheye = 0.0
    relativistic = false
    speed = 2.0             # free-cam speed at reference altitude
    thrust = 0.05           # ship max proper acceleration, c²/M
    twarp = 2.0             # ship proper time per wall second, M
    exposure = 1.0f0        # display transform (T/G keys)
    filmic = true           # ACES-style display curve (P toggles)

    # Layered engine state: deflection fan + the disc/gas layer. While
    # moving, the layer renders FRESH every frame at half display resolution
    # (a quarter-res rung when the frame runs hot) — no temporal history:
    # reprojected history echoes badly next to the photon ring, where the
    # parallax of wound light paths is extreme. At rest, one full-resolution
    # pass. Sky/shadow are always per-frame exact and native.
    sky = SkyFanState(n=fan_n)
    layer_on = disc !== nothing || volume !== nothing
    L = MtlArray{Float32,3}(undef, 4, width, height)          # rest: native
    Lh = MtlArray{Float32,3}(undef, 4, width ÷ 2, height ÷ 2) # moving
    Lq = MtlArray{Float32,3}(undef, 4, width ÷ 4, height ÷ 4) # moving, hot
    if !layer_on
        empty_layer = zeros(Float32, 4, width, height)
        empty_layer[4, :, :] .= 1.0f0    # fully transparent: sky only
        copyto!(L, empty_layer)
    end
    # Gas gate: rays that provably stay outside this radius carry no disc or
    # gas and short-circuit to pure transparency in the layer pass.
    gate = 0.0
    disc !== nothing && (gate = max(gate, disc.outer_radius))
    volume !== nothing &&
        (gate = max(gate, hypot(exp(volume.log_s_out), volume.z_max)))
    gate *= 1.05

    build_cam() = camera_from_state(state, focal, 2.8, norm(state.pos), false)

    # Warm every kernel (fan, both layer variants, composite, pack) before
    # the clock starts: first-call compilation costs seconds and would
    # otherwise hitch the opening frames.
    let cam0 = build_cam()
        update_sky_fan!(sky, ctx, state.pos, spacetime; gate=gate)
        if ctx.has_volume
            set_volume_enabled!(ctx, false)
            render_layered_gpu!(ctx.out_gpu, L, ctx, sky, cam0,
                                spacetime; trace_layer=layer_on)
            set_volume_enabled!(ctx, true)
        end
        render_layered_gpu!(ctx.out_gpu, L, ctx, sky, cam0,
                            spacetime; trace_layer=layer_on)
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
    frame_ms = 16.0
    nframes = 0
    β = SVector(0.0, 0.0, 0.0)
    γ = 1.0
    a_mag = 0.0
    last_sig = nothing
    REFINE_PASSES = 32
    passes = 0
    refine_row = 0
    mpr = 1.0e-4            # measured seconds per refined row (EMA)
    accum = MtlArray{Float32,3}(undef, 3, width, height)

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
        state.roll += ((down(GLFW.KEY_C) ? 1.0 : 0.0) -
                       (down(GLFW.KEY_Z) ? 1.0 : 0.0)) * 1.5 * dwall
        pressed_once(GLFW.KEY_V) && ctx.has_volume &&
            set_volume_enabled!(ctx, !ctx.vol_on[])
        pressed_once(GLFW.KEY_R) && (relativistic = !relativistic)
        pressed_once(GLFW.KEY_L) && (fisheye = fisheye > 0.0 ? 0.0 : 100.0)
        pressed_once(GLFW.KEY_P) && (filmic = !filmic)
        down(GLFW.KEY_T) && (exposure = min(20.0f0, exposure * 1.04f0))
        down(GLFW.KEY_G) && (exposure = max(0.05f0, exposure / 1.04f0))
        if pressed_once(GLFW.KEY_F)
            flight = !flight
            flight && (ship = ShipState(state.pos, M))
        end
        if pressed_once(GLFW.KEY_X)
            state.pos = spawn_pos
            ship = ShipState(spawn_pos, M)
        end
        if flight
            down(GLFW.KEY_LEFT_BRACKET) && (thrust = max(0.005, thrust / 1.03))
            down(GLFW.KEY_RIGHT_BRACKET) && (thrust = min(0.5, thrust * 1.03))
            down(GLFW.KEY_MINUS) && (twarp = max(0.0, twarp - 8.0 * dwall))
            down(GLFW.KEY_EQUAL) && (twarp = min(30.0, twarp + 8.0 * dwall))
        else
            down(GLFW.KEY_LEFT_BRACKET) && (speed = max(0.1, speed / 1.03))
            down(GLFW.KEY_RIGHT_BRACKET) && (speed = min(50.0, speed * 1.03))
        end

        # --- movement --------------------------------------------------
        fwd, right, _ = _flycam_basis(state)
        upr = _flycam_up(state)
        if flight
            # GR ship: same model as the flythrough's flight mode.
            dτ = twarp * dwall
            acc = SVector(
                (down(GLFW.KEY_W) ? 1.0 : 0.0) - (down(GLFW.KEY_S) ? 1.0 : 0.0),
                (down(GLFW.KEY_D) ? 1.0 : 0.0) - (down(GLFW.KEY_A) ? 1.0 : 0.0),
                (down(GLFW.KEY_E) ? 1.0 : 0.0) - (down(GLFW.KEY_Q) ? 1.0 : 0.0))
            na = norm(acc)
            na > 1.0 && (acc = acc / na)
            amax = thrust * (down(GLFW.KEY_LEFT_SHIFT) ? 4.0 : 1.0)
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
            β, γ = ship_velocity(ship, M, fwd, right, upr)   # telemetry
        else
            # Omnipotent free camera: flat velocity while keys are held.
            v = speed * dwall * (down(GLFW.KEY_LEFT_SHIFT) ? 5.0 : 1.0)
            rn = norm(state.pos)
            v *= clamp(0.12 * max(rn - 1.9 * M, 0.25 * rn), 0.02, 8.0)
            world_z = SVector(0.0, 0.0, 1.0)
            down(GLFW.KEY_W) && (state.pos += v * fwd)
            down(GLFW.KEY_S) && (state.pos -= v * fwd)
            down(GLFW.KEY_A) && (state.pos -= v * right)
            down(GLFW.KEY_D) && (state.pos += v * right)
            down(GLFW.KEY_Q) && (state.pos -= v * world_z)
            down(GLFW.KEY_E) && (state.pos += v * world_z)
            rn = norm(state.pos)
            rn < 0.45 * M && (state.pos *= 0.45 * M / rn)
        end

        # --- render: fan + layer + composite, all on the GPU -----------
        # Change detection drives three states: moving (fast bounded gas
        # layer, 60 fps), just stopped (one full-native-resolution gas pass
        # — stills get offline sharpness), parked (nothing to render; the
        # layer retains the last drawable and the GPU idles).
        sig = (state.pos, state.yaw, state.pitch, state.roll, fisheye,
               relativistic, ctx.vol_on[], exposure, filmic)
        cam_now = build_cam()
        if sig != last_sig
            if haskey(ENV, "SPACETIME_DEBUG") && last_sig !== nothing
                for (ci, (a, b)) in enumerate(zip(sig, last_sig))
                    a != b && println("sig[", ci, "] changed: ", b, " -> ", a)
                end
            end
            last_sig = sig
            passes = 0
            refine_row = 0
            t0 = time()
            update_sky_fan!(sky, ctx, state.pos, spacetime; gate=gate, dt=0.05)
            if layer_on
                # Fresh gas every frame — half display resolution, dropping
                # a rung when the frame budget runs hot. No history: nothing
                # to echo.
                render_layered_gpu!(ctx.out_gpu, frame_ms > 20.0 ? Lq : Lh,
                                    ctx, sky, cam_now, spacetime;
                                    fisheye_deg=fisheye,
                                    relativistic=relativistic)
            else
                render_layered_gpu!(ctx.out_gpu, L, ctx, sky, cam_now,
                                    spacetime; fisheye_deg=fisheye,
                                    relativistic=relativistic,
                                    trace_layer=false)
            end
            present!(presenter, ctx.out_gpu; exposure=exposure, filmic=filmic)
            frame_ms = 0.9 * frame_ms + 0.1 * 1000 * (time() - t0)
            nframes += 1
        elseif passes < REFINE_PASSES
            # At rest: time-sliced progressive refinement. Full-resolution
            # jittered passes (R2 low-discrepancy sequence) keep summing
            # into `accum` — every pass is real rays, so the still image
            # converges to a supersampled photograph — but the work is
            # submitted in row bands of ~7 ms and synchronized per band, so
            # the event loop keeps its cadence and any input aborts between
            # bands. Presented at pass boundaries; the presenter's exposure
            # factor divides by the pass count.
            t0 = time()
            ju = passes == 0 ? 0.5 : mod(0.5 + 0.7548776662466927 * passes, 1.0)
            jv = passes == 0 ? 0.5 : mod(0.5 + 0.5698402909980532 * passes, 1.0)
            band = clamp(round(Int, 0.007 / mpr), 16, height - refine_row)
            render_layered_gpu!(accum, L, ctx, sky, cam_now, spacetime;
                                fisheye_deg=fisheye,
                                relativistic=relativistic,
                                trace_layer=layer_on, ju=ju, jv=jv,
                                accumulate=passes > 0,
                                row0=refine_row, rows=band)
            Metal.synchronize()
            mpr = 0.7 * mpr + 0.3 * (time() - t0) / band
            refine_row += band
            if refine_row >= height
                refine_row = 0
                passes += 1
                present!(presenter, accum;
                         exposure=exposure / Float32(passes), filmic=filmic)
                nframes += 1
            end
        else
            sleep(0.006)
        end

        # --- telemetry in the title bar (cheap, 4 Hz) ------------------
        if wall - last_title > 0.25
            last_title = wall
            r = norm(state.pos)
            regime = r < 2.0 ? "INSIDE HORIZON" : r < 3.0 ? "PHOTON SPHERE" :
                     r < 6.0 ? "BELOW ISCO" : ""
            info = flight ?
                @sprintf("FLIGHT · β %.3fc γ %.2f · a %s · thr %.2f · warp %.1f · τ %.1fs t %.1fs",
                         norm(β), γ,
                         a_mag > 0 ? @sprintf("%.2f", a_mag) : "0 (free fall)",
                         thrust, twarp, ship.τ * 0.49255, ship.t * 0.49255) :
                @sprintf("free cam · spd %.1f", speed)
            GLFW.SetWindowTitle(win, @sprintf(
                "%s — r %.2fM %s · %s · %.1f ms (%.0f fps) · spp %d/%d",
                title, r, regime, info, frame_ms,
                1000.0 / max(frame_ms, 1.0e-3), passes, REFINE_PASSES))
        end
    end
    elapsed = time() - t_start
    @printf("fly_native: %.1f s, %d frames = %.1f fps, last frame %.1f ms\n",
            elapsed, nframes, nframes / elapsed, frame_ms)
    GLFW.DestroyWindow(win)
    return nothing
end
