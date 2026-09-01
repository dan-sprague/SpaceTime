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
using Libdl: dlopen, dlsym
using Metal: MTL

# CGSize for -[CAMetalLayer setDrawableSize:] (two Cdoubles on 64-bit).
struct _CGSize
    width::Cdouble
    height::Cdouble
end

# ---------------------------------------------------------------------------
# Display grade
# ---------------------------------------------------------------------------
#
# The presenter carries a 16-float grade block applied on the GPU during
# packing, plus a bloom/streak chain faithful to `postprocess()` (the video
# pipeline): per-channel bright pass, quarter-res Gaussian bloom with a
# sixteenth-res wide stage (the Moffat kernel's long tails), a 4-spike
# exponential streak pass, hue-preserving ACES, free gamma power, grain:
#   [1] exposure (linear ev) [2] filmic on [3] crush   [4] saturation
#   [5] vignette  [6:8] white-balance gains (r, g, b)
#   [9] bloom strength [10] bloom threshold [11] tonemap hue-preserve k
#   [12] gamma power (0.4545 = sRGB; 5 = the gamma-0.2 crush)
#   [13] grain amount  [14] streak fraction of scattered light
#   [15] streak peak gain (compensates the quarter-res line width — the
#        video's FFT streaks are pixel-thin and ~4× brighter per pixel)
#   [16] spare

# Per-channel bright-pass 4×4 box downsample into the quarter-res source
# (matches postprocess(): bright = max(c·ev − threshold, 0)).
function _bloom_down_kernel!(dst, src, W, H, BW, BH, e, thresh)
    i = thread_position_in_grid().x
    i > BW * BH && return
    bx = (i - 1) % BW
    by = (i - 1) ÷ BW
    r = 0.0f0; g = 0.0f0; b = 0.0f0
    for oy in 1:4, ox in 1:4
        sx = min(4 * bx + ox, W)
        sy = min(4 * by + oy, H)
        r += src[1, sx, sy]
        g += src[2, sx, sy]
        b += src[3, sx, sy]
    end
    r = r * 0.0625f0 * e
    g = g * 0.0625f0 * e
    b = b * 0.0625f0 * e
    r = r == r ? r : 0.0f0
    g = g == g ? g : 0.0f0
    b = b == b ? b : 0.0f0
    dst[1, bx + 1, by + 1] = max(r - thresh, 0.0f0)
    dst[2, bx + 1, by + 1] = max(g - thresh, 0.0f0)
    dst[3, bx + 1, by + 1] = max(b - thresh, 0.0f0)
    return nothing
end

# 2×2 box downsample (quarter → sixteenth) for the wide bloom stage.
function _bloom_half_kernel!(dst, src, SW, SH, DW, DH)
    i = thread_position_in_grid().x
    i > DW * DH && return
    x = (i - 1) % DW
    y = (i - 1) ÷ DW
    for c in 1:3
        v = 0.0f0
        for oy in 1:2, ox in 1:2
            v += src[c, min(2 * x + ox, SW), min(2 * y + oy, SH)]
        end
        dst[c, x + 1, y + 1] = 0.25f0 * v
    end
    return nothing
end

# 4-spike star streaks: exponential-decay line blur along four directions
# (0°, 45°, 90°, 135°), the GPU analogue of generate_streak_kernel.
# `L` is the decay length in source pixels.
function _streak_kernel!(dst, src, BW, BH, L)
    i = thread_position_in_grid().x
    i > BW * BH && return
    x = (i - 1) % BW + 1
    y = (i - 1) ÷ BW + 1
    r = 0.0f0; g = 0.0f0; b = 0.0f0
    wsum = 0.0f0
    invL = 1.0f0 / L
    d = 0
    while d < 4
        dx = d == 0 ? 1.0f0 : d == 1 ? 0.7071f0 : d == 2 ? 0.0f0 : -0.7071f0
        dy = d == 0 ? 0.0f0 : d == 1 ? 0.7071f0 : d == 2 ? 1.0f0 : 0.7071f0
        for t in 1:20
            s = Float32(t) * 2.5f0
            w = exp(-s * invL)
            for sgn in (-1.0f0, 1.0f0)
                sx = clamp(unsafe_trunc(Int32, Float32(x) + sgn * s * dx),
                           Int32(1), Int32(BW))
                sy = clamp(unsafe_trunc(Int32, Float32(y) + sgn * s * dy),
                           Int32(1), Int32(BH))
                r += w * src[1, sx, sy]
                g += w * src[2, sx, sy]
                b += w * src[3, sx, sy]
                wsum += w
            end
        end
        d += 1
    end
    w0 = 1.0f0
    r += w0 * src[1, x, y]; g += w0 * src[2, x, y]; b += w0 * src[3, x, y]
    wsum += w0
    dst[1, x, y] = r / wsum
    dst[2, x, y] = g / wsum
    dst[3, x, y] = b / wsum
    return nothing
end

# Small integer hash → [0, 1) for sensor-grain noise.
@inline function _grain_hash(x::Int32, y::Int32)
    h = x * Int32(374761393) + y * Int32(668265263)
    h = (h ⊻ (h >> 13)) * Int32(1274126177)
    h = h ⊻ (h >> 16)
    return Float32(h & Int32(0x00FFFFFF)) * 5.9604645f-8
end

# 9-tap separable Gaussian blur along (dx, dy) at bloom resolution.
function _bloom_blur_kernel!(dst, src, BW, BH, dx, dy)
    i = thread_position_in_grid().x
    i > BW * BH && return
    x = (i - 1) % BW + 1
    y = (i - 1) ÷ BW + 1
    r = 0.0f0; g = 0.0f0; b = 0.0f0
    wsum = 0.0f0
    for t in -4:4
        w = t == 0 ? 0.205f0 : (abs(t) == 1 ? 0.180f0 : abs(t) == 2 ? 0.124f0 :
                                abs(t) == 3 ? 0.066f0 : 0.028f0)
        sx = clamp(x + t * dx, 1, BW)
        sy = clamp(y + t * dy, 1, BH)
        r += w * src[1, sx, sy]
        g += w * src[2, sx, sy]
        b += w * src[3, sx, sy]
        wsum += w
    end
    dst[1, x, y] = r / wsum
    dst[2, x, y] = g / wsum
    dst[3, x, y] = b / wsum
    return nothing
end

# Pack the renderer's (3, W, H) Float32 output into BGRA8 texture order
# through the full display grade. Row 0 of the texture is the top of the
# image, which is column j = H of the render (the PNG save path applies
# rotr90 for the same reason). NaN guards: clamp propagates NaN, and a
# checked convert would trap the GPU. `escale` is an extra linear factor
# (the progressive-refinement pass average).
@inline function _grade_bilinear(buf, c, fx, fy, BW, BH)
    x0 = clamp(unsafe_trunc(Int32, floor(fx)), Int32(1), Int32(BW - 1))
    y0 = clamp(unsafe_trunc(Int32, floor(fy)), Int32(1), Int32(BH - 1))
    tx = clamp(fx - Float32(x0), 0.0f0, 1.0f0)
    ty = clamp(fy - Float32(y0), 0.0f0, 1.0f0)
    return buf[c, x0, y0] * (1 - tx) * (1 - ty) +
           buf[c, x0 + 1, y0] * tx * (1 - ty) +
           buf[c, x0, y0 + 1] * (1 - tx) * ty +
           buf[c, x0 + 1, y0 + 1] * tx * ty
end

@inline _aces(x) = clamp((x * (2.51f0 * x + 0.03f0)) /
                         (x * (2.43f0 * x + 0.59f0) + 0.14f0), 0.0f0, 1.0f0)

function _pack_bgra_kernel!(dst, src, bloomg, bloomw, blooms,
                            W, H, BW, BH, WW, WH, grade, palette, escale)
    i = thread_position_in_grid().x
    i > W * H && return
    x = (i - 1) % W + 1
    j = H - (i - 1) ÷ W
    # One-time register loads of the grade block (device-memory reads inside
    # the hot path below would repeat per use).
    g_exp = grade[1]; g_film = grade[2]; g_crush = grade[3]
    g_sat = grade[4]; g_vig = grade[5]
    g_wbr = grade[6]; g_wbg = grade[7]; g_wbb = grade[8]
    g_bloom = grade[9]; g_hue = grade[11]; g_gp = grade[12]
    g_grain = grade[13]; g_sfrac = grade[14]; g_sgain = grade[15]
    g_qlev = grade[17]; g_dith = grade[18]; g_pal = grade[19]
    e = g_exp * escale
    r = src[1, x, j] * e * g_wbr
    g = src[2, x, j] * e * g_wbg
    b = src[3, x, j] * e * g_wbb
    r = r == r ? r : 0.0f0
    g = g == g ? g : 0.0f0
    b = b == b ? b : 0.0f0
    if g_bloom > 0.0f0
        # Scattered light: Gaussian core + wide stage (Moffat-like tails),
        # plus the streak pass, split by the streak fraction.
        bfrac = 1.0f0 - g_sfrac
        fx = (Float32(x) - 0.5f0) * Float32(BW) / Float32(W) + 0.5f0
        fy = (Float32(j) - 0.5f0) * Float32(BH) / Float32(H) + 0.5f0
        wx = (Float32(x) - 0.5f0) * Float32(WW) / Float32(W) + 0.5f0
        wy = (Float32(j) - 0.5f0) * Float32(WH) / Float32(H) + 0.5f0
        for c in 1:3
            core = _grade_bilinear(bloomg, c, fx, fy, BW, BH)
            wide = _grade_bilinear(bloomw, c, wx, wy, WW, WH)
            bl = bfrac * (0.6f0 * core + 0.4f0 * wide)
            if g_sfrac > 0.0f0
                bl += g_sfrac * g_sgain *
                      _grade_bilinear(blooms, c, fx, fy, BW, BH)
            end
            if c == 1
                r += g_bloom * bl
            elseif c == 2
                g += g_bloom * bl
            else
                b += g_bloom * bl
            end
        end
    end
    l = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
    r = max(l + g_sat * (r - l), 0.0f0)
    g = max(l + g_sat * (g - l), 0.0f0)
    b = max(l + g_sat * (b - l), 0.0f0)
    if g_film > 0.5f0
        # ACES per channel, blended with the hue-preserving variant
        # (tonemap the luminance, rescale the triple) by grade[11] —
        # postprocess()'s tonemap_hue_preserve.
        pr = _aces(r); pg = _aces(g); pb = _aces(b)
        if g_hue > 0.0f0
            Y = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
            sY = _aces(Y) / max(Y, 1.0f-8)
            r = (1.0f0 - g_hue) * pr + g_hue * clamp(r * sY, 0.0f0, 1.0f0)
            g = (1.0f0 - g_hue) * pg + g_hue * clamp(g * sY, 0.0f0, 1.0f0)
            b = (1.0f0 - g_hue) * pb + g_hue * clamp(b * sY, 0.0f0, 1.0f0)
        else
            r = pr; g = pg; b = pb
        end
        r = exp(log(max(r, 1.0f-6)) * g_gp)
        g = exp(log(max(g, 1.0f-6)) * g_gp)
        b = exp(log(max(b, 1.0f-6)) * g_gp)
    end
    if g_crush != 1.0f0
        r = exp(log(max(r, 1.0f-6)) * g_crush)
        g = exp(log(max(g, 1.0f-6)) * g_crush)
        b = exp(log(max(b, 1.0f-6)) * g_crush)
    end
    if g_grain > 0.0f0
        # Sensor grain in display space (the video applies sensor_expose!
        # after the grade): luma-scaled, zero-mean.
        l2 = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
        n = g_grain * (_grain_hash(Int32(x), Int32(j)) - 0.5f0) *
            2.0f0 * sqrt(max(l2, 2.0f-3))
        r += n; g += n; b += n
    end
    if g_vig > 0.0f0
        vu = (Float32(x) - 0.5f0 * Float32(W)) / (0.5f0 * Float32(H))
        vv = (Float32(j) - 0.5f0 * Float32(H)) / (0.5f0 * Float32(H))
        f = max(1.0f0 - g_vig * 0.25f0 * (vu * vu + vv * vv), 0.0f0)
        r *= f; g *= f; b *= f
    end
    r = clamp(r, 0.0f0, 1.0f0)
    g = clamp(g, 0.0f0, 1.0f0)
    b = clamp(b, 0.0f0, 1.0f0)
    # Arcade palette: quantize each channel to `g_qlev` levels, with a 4x4
    # Bayer ordered dither so the disc's smooth temperature ramp breaks into
    # a stable pattern instead of hard bands. Ordered rather than random on
    # purpose — the pattern is fixed to the pixel grid, so it does not crawl
    # when the camera moves, which is the failure mode that makes low-res
    # rendering read as broken rather than stylised.
    if g_pal > 1.5f0 || g_qlev > 1.5f0
        # 4x4 Bayer from the recursive 2x2 definition, branch-free.
        bx0 = Int32((x - 1) & 1);  by0 = Int32((j - 1) & 1)
        bx1 = Int32(((x - 1) >> 1) & 1);  by1 = Int32(((j - 1) >> 1) & 1)
        m = 4 * (2 * bx1 + by1 * (3 - 4 * bx1)) + (2 * bx0 + by0 * (3 - 4 * bx0))
        d = g_dith * ((Float32(m) + 0.5f0) * 0.0625f0 - 0.5f0)
        if g_pal > 1.5f0
            # Palette mode: collapse to N tones by LUMINANCE, not per channel.
            # Per-channel quantization keeps three independent ramps and so
            # keeps the picture colourful; mapping brightness onto one ramp is
            # what actually reduces the tone count, and lets entry 1 be true
            # black so the sky reads as empty rather than tinted.
            n = g_pal
            lum = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
            k = unsafe_trunc(Int32, clamp(lum * n + d, 0.0f0, n - 1.0f0)) + Int32(1)
            r = palette[1, k]; g = palette[2, k]; b = palette[3, k]
        else
            L = g_qlev - 1.0f0
            r = clamp(floor(r * L + 0.5f0 + d) / L, 0.0f0, 1.0f0)
            g = clamp(floor(g * L + 0.5f0 + d) / L, 0.0f0, 1.0f0)
            b = clamp(floor(b * L + 0.5f0 + d) / L, 0.0f0, 1.0f0)
        end
    end
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
    pack_kernel::Base.RefValue{Any}   # compiled-once kernels, like _WARP_KERNEL
    bloom_kernels::Base.RefValue{Any}
    grade::MtlVector{Float32}         # display-grade block (see set_grade!)
    grade_host::Vector{Float32}
    palette::MtlArray{Float32,2}      # (3, PALETTE_MAX) arcade palette
    bloom_a::MtlArray{Float32,3}      # quarter-res bloom ping-pong
    bloom_b::MtlArray{Float32,3}
    bloom_s::MtlArray{Float32,3}      # quarter-res streaks
    bloom_w1::MtlArray{Float32,3}     # sixteenth-res wide stage
    bloom_w2::MtlArray{Float32,3}
    width::Int
    height::Int
end

"""
Matplotlib's **plasma** ramp at deciles: dark blue-violet through magenta and
orange to yellow. Sampled and interpolated by [`plasma_palette`](@ref).
"""
const PALETTE_MAX = 32

const PLASMA_ANCHORS = (
    (0.050f0, 0.030f0, 0.528f0), (0.255f0, 0.014f0, 0.615f0),
    (0.418f0, 0.001f0, 0.658f0), (0.563f0, 0.052f0, 0.642f0),
    (0.693f0, 0.165f0, 0.565f0), (0.798f0, 0.280f0, 0.470f0),
    (0.881f0, 0.393f0, 0.383f0), (0.949f0, 0.518f0, 0.296f0),
    (0.987f0, 0.652f0, 0.211f0), (0.988f0, 0.816f0, 0.145f0),
    (0.940f0, 0.975f0, 0.131f0))

"""
    plasma_palette(n; black=true, lo=0.0, hi=1.0)

`(3, n)` array of plasma colours for the arcade palette. With `black`, the
first entry is pure black and the remaining `n-1` span `lo`..`hi` of the ramp —
so the sky bottoms out at true black rather than plasma's dark violet, and the
tones above it are few and deliberate.
"""
function plasma_palette(n::Int; black::Bool=true, lo::Real=0.0, hi::Real=1.0)
    n >= 2 || throw(ArgumentError("palette needs at least 2 entries"))
    pal = Array{Float32}(undef, 3, n)
    k0 = black ? 2 : 1
    if black
        pal[:, 1] .= 0.0f0
    end
    m = n - k0
    for k in k0:n
        t = m == 0 ? Float64(hi) : lo + (hi - lo) * (k - k0) / m
        u = clamp(t, 0.0, 1.0) * (length(PLASMA_ANCHORS) - 1)
        i = clamp(floor(Int, u), 0, length(PLASMA_ANCHORS) - 2)
        f = Float32(u - i)
        a = PLASMA_ANCHORS[i + 1]; b = PLASMA_ANCHORS[i + 2]
        for c in 1:3
            pal[c, k] = a[c] + f * (b[c] - a[c])
        end
    end
    return pal
end

"""
    set_palette!(p::MetalPresenter, pal)

Install an arcade palette: a `(3, n)` array of linear RGB. The pack kernel maps
each pixel's **luminance** to one of the `n` entries (ordered-dithered), so the
frame collapses to exactly `n` tones. `set_palette!(p, nothing)` restores the
full-colour path.
"""
function set_palette!(p::MetalPresenter, pal::Union{AbstractMatrix,Nothing})
    if pal === nothing
        p.grade_host[19] = 0.0f0
    else
        size(pal, 1) == 3 || throw(ArgumentError("palette must be (3, n)"))
        n = size(pal, 2)
        n <= size(p.palette, 2) ||
            throw(ArgumentError("palette has $n entries; buffer holds $(size(p.palette, 2))"))
        host = zeros(Float32, 3, size(p.palette, 2))
        host[:, 1:n] .= Float32.(pal)
        copyto!(p.palette, host)
        p.grade_host[19] = Float32(n)
    end
    copyto!(p.grade, p.grade_host)
    return nothing
end

function MetalPresenter(win::GLFW.Window, width::Int, height::Int;
                        nearest::Bool=false)
    qc = dlopen("/System/Library/Frameworks/QuartzCore.framework/QuartzCore")
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
    if nearest
        # Arcade mode renders at a low internal resolution and lets the
        # compositor blow it up. The default CAMetalLayer magnification filter
        # is linear, which turns pixel art into mush; `kCAFilterNearest` keeps
        # the pixel grid hard. Doing it here rather than in a shader means the
        # upscale is free — the window server was going to scale the drawable
        # either way.
        nf = unsafe_load(convert(Ptr{id{Object}}, dlsym(qc, :kCAFilterNearest)))
        @objc [layer::id{Object} setMagnificationFilter:nf::id{Object}]::Nothing
        @objc [layer::id{Object} setMinificationFilter:nf::id{Object}]::Nothing
    end
    @objc [view::id{Object} setWantsLayer:true::Bool]::Nothing
    @objc [view::id{Object} setLayer:layer::id{Object}]::Nothing
    queue = Metal.global_queue(dev)
    grade_host = Float32[1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0.4545, 0, 0, 0,
                         0, 0, 0, 0, 0]
    grade = MtlArray(grade_host)
    bw, bh = cld(width, 4), cld(height, 4)
    ww, wh = cld(bw, 2), cld(bh, 2)
    MetalPresenter{typeof(queue)}(layer, queue,
                                  MtlArray{UInt32}(undef, width * height),
                                  Ref{Any}(nothing), Ref{Any}(nothing),
                                  grade, grade_host,
                                  MtlArray(zeros(Float32, 3, PALETTE_MAX)),
                                  MtlArray{Float32,3}(undef, 3, bw, bh),
                                  MtlArray{Float32,3}(undef, 3, bw, bh),
                                  MtlArray{Float32,3}(undef, 3, bw, bh),
                                  MtlArray{Float32,3}(undef, 3, ww, wh),
                                  MtlArray{Float32,3}(undef, 3, ww, wh),
                                  width, height)
end

"""
    set_grade!(p::MetalPresenter; exposure, filmic, crush, saturation,
               vignette, wb, bloom, bloom_threshold)

Update the presenter's display grade (any subset of the fields; `wb` is an
`(r, g, b)` gain tuple). Applied on the next `present!`.
"""
function set_grade!(p::MetalPresenter; exposure::Real=p.grade_host[1],
                    filmic::Bool=p.grade_host[2] > 0.5,
                    crush::Real=p.grade_host[3],
                    saturation::Real=p.grade_host[4],
                    vignette::Real=p.grade_host[5],
                    wb::NTuple{3,<:Real}=(p.grade_host[6], p.grade_host[7],
                                          p.grade_host[8]),
                    bloom::Real=p.grade_host[9],
                    bloom_threshold::Real=p.grade_host[10],
                    hue_preserve::Real=p.grade_host[11],
                    gamma_power::Real=p.grade_host[12],
                    grain::Real=p.grade_host[13],
                    streak_fraction::Real=p.grade_host[14],
                    streak_gain::Real=p.grade_host[15],
                    quantize::Real=p.grade_host[17],
                    dither::Real=p.grade_host[18])
    p.grade_host[1:18] .= Float32[exposure, filmic ? 1 : 0, crush, saturation,
                            vignette, wb[1], wb[2], wb[3], bloom,
                            bloom_threshold, hue_preserve, gamma_power,
                            grain, streak_fraction, streak_gain, 0,
                            quantize, dither]
    copyto!(p.grade, p.grade_host)
    return nothing
end

"""
    present!(p::MetalPresenter, src::MtlArray{Float32,3})

Pack `src` (the renderer's `(3, W, H)` output) to BGRA and blit it into the
next drawable. Blocks until a drawable is free (≈ vsync when saturated).
Command-buffer order on the shared global queue keeps the pack after any
in-flight render kernels; an explicit flush publishes Metal.jl's batched
launches before ours commits.
"""
# Input-pump deadline control (see the pump call site in `fly_native`).
# `PUMP_BLOCKED_MS` is the `present!` block above which we conclude something
# other than the pump set the frame's pace, making its elapsed time a true
# measurement rather than the pump's own padding.
const PUMP_BLOCKED_MS = 0.5
const PUMP_MIN_MS = 1.0
const PUMP_MAX_MS = 120.0
const PUMP_DECAY = 0.85

function present!(p::MetalPresenter, src::MtlArray{Float32,3};
                  escale::Float32=1.0f0)
    W, H = p.width, p.height
    BW, BH = size(p.bloom_a, 2), size(p.bloom_a, 3)
    n = W * H
    WW, WH = size(p.bloom_w1, 2), size(p.bloom_w1, 3)
    if p.grade_host[9] > 0.0f0
        if p.bloom_kernels[] === nothing
            p.bloom_kernels[] = (
                @metal(launch=false, _bloom_down_kernel!(
                    p.bloom_a, src, W, H, BW, BH, 1.0f0, 1.0f0)),
                @metal(launch=false, _bloom_blur_kernel!(
                    p.bloom_b, p.bloom_a, BW, BH, 1, 0)),
                @metal(launch=false, _streak_kernel!(
                    p.bloom_s, p.bloom_a, BW, BH, 1.0f0)),
                @metal(launch=false, _bloom_half_kernel!(
                    p.bloom_w1, p.bloom_a, BW, BH, WW, WH)),
                @metal(launch=false, _bloom_blur_kernel!(
                    p.bloom_w2, p.bloom_w1, WW, WH, 1, 0)))
        end
        kd, kb, ks, kh, kw = p.bloom_kernels[]
        nb = BW * BH
        nw = WW * WH
        td = min(kd.pipeline.maxTotalThreadsPerThreadgroup, nb)
        kd(p.bloom_a, src, W, H, BW, BH, p.grade_host[1] * escale,
           p.grade_host[10]; threads=td, groups=cld(nb, td))
        tb = min(kb.pipeline.maxTotalThreadsPerThreadgroup, nb)
        kb(p.bloom_b, p.bloom_a, BW, BH, 1, 0; threads=tb, groups=cld(nb, tb))
        kb(p.bloom_a, p.bloom_b, BW, BH, 0, 1; threads=tb, groups=cld(nb, tb))
        if p.grade_host[14] > 0.0f0
            # Streaks read the BLURRED bright pass: a compact far-away
            # source sampled with strided nearest taps turns into dashed
            # rays; pre-smoothing the source keeps them continuous.
            # Decay length ≈ 0.1·W, the porthole recipe.
            ts = min(ks.pipeline.maxTotalThreadsPerThreadgroup, nb)
            ks(p.bloom_s, p.bloom_a, BW, BH, Float32(0.1f0 * BW);
               threads=ts, groups=cld(nb, ts))
        end
        th = min(kh.pipeline.maxTotalThreadsPerThreadgroup, nw)
        kh(p.bloom_w1, p.bloom_a, BW, BH, WW, WH; threads=th,
           groups=cld(nw, th))
        tw = min(kw.pipeline.maxTotalThreadsPerThreadgroup, nw)
        kw(p.bloom_w2, p.bloom_w1, WW, WH, 1, 0; threads=tw,
           groups=cld(nw, tw))
        kw(p.bloom_w1, p.bloom_w2, WW, WH, 0, 1; threads=tw,
           groups=cld(nw, tw))
    end
    if p.pack_kernel[] === nothing
        p.pack_kernel[] = @metal launch=false _pack_bgra_kernel!(
            p.pack_gpu, src, p.bloom_a, p.bloom_w1, p.bloom_s,
            W, H, BW, BH, WW, WH, p.grade, p.palette, escale)
    end
    kern = p.pack_kernel[]
    threads = min(kern.pipeline.maxTotalThreadsPerThreadgroup, n)
    kern(p.pack_gpu, src, p.bloom_a, p.bloom_w1, p.bloom_s,
         W, H, BW, BH, WW, WH, p.grade, p.palette, escale;
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

The sky is the **procedural starfield** by default, not the equirectangular
starmap: a fixed angular texture magnifies into blobs exactly where lensing
stretches the sky hardest, at the photon ring. `B` cycles stars → starmap →
both; `starfield=false` restores the texture, and `star_texture_weight` keeps
some of it under the stars (the map is good at diffuse Milky Way glow and bad
at point sources). Star colour has its own white point, independent of the
disc's, so regrading the disc never re-tints the sky.

`quality` (**O** cycles it live) sets the photon ring's own render stride:
`:performance` 4, `:balanced` 2, `:quality` 1. The bulk gas ladder stays deep
in every preset — it is what holds the frame budget — because the ring is the
one feature a coarse rung cannot carry. Measured at 1440p with the bulk at
stride 8: a stride-2 ring reaches 98% of the sharpness of tracing the *whole*
layer at stride 2, for 24% of the frame cost (21.6 ms against 89.4 ms), and a
stride-1 ring beats it outright at under half the cost. `ring_rmin` (4M) is
the periapsis below which a ray counts as wound.

Other keys: drag to look; Z/C roll; 5 (or V) thin-disc/volumetric gas;
B sky; O quality;
R relativistic shading; L lens (rectilinear/fisheye); **1/2/3/4 grade presets**
(neutral / film / hectic / **porthole** — the escape-video recipe:
hue-preserving ACES, gamma-0.2 crush, 4-spike streaks, grain); T/G
exposure; P filmic curve on/off; X reset
position; Esc quit. When the camera rests, refinement passes accumulate a
supersampled still, then the GPU parks until something changes.
Telemetry lives in the window title. Runs on the calling (main) thread
until the window closes.
"""
function fly_native(cam::AbstractCamera, spacetime::AbstractSpacetime, background;
                    disc::Union{AccretionDisc,Nothing}=nothing,
                    volume::Union{DiscVolume,Nothing}=nothing,
                    width::Int=960, height::Int=540,
                    winwidth::Int=1600, winheight::Int=900,
                    fan_n::Int=4096,
                    title::String="Spacetime Simulator",
                    starfield::Bool=true, star_texture_weight::Real=0.0,
                    star_psf_pixels::Real=1.0, star_density::Real=384,
                    quality::Symbol=:balanced, ring_rmin::Real=4.0,
                    arcade::Bool=(spin(spacetime) != 0),
                    quantize::Real=0, dither::Real=1.0,
                    palette::Union{Nothing,Integer,AbstractMatrix}=nothing,
                    max_seconds::Float64=Inf)   # finite for smoke tests
    M = spacetime.M
    # Arcade mode. The deflection fan needs spherical symmetry, so Kerr cannot
    # use it -- a Kerr "fan" would be a 2D table the size of the frame. Every
    # pixel is traced directly instead, which is only affordable at a low
    # internal resolution, which is exactly the pixel-art look. On by default
    # for any spinning spacetime; the layered Schwarzschild engine below is
    # untouched and still runs whenever this is off.
    ctx = MetalPreviewContext(background, width, height;
                              dt=0.1, nmax=1000, disc=disc, volume=volume)

    # Procedural stars in place of the starmap. The 4k equirectangular texture
    # is a fixed angular grid, so lensing magnifies its texels into visible
    # blobs wherever the deflection stretches the sky — worst exactly at the
    # photon ring, where the interesting structure is. The procedural field is
    # evaluated along the asymptotic direction at the pixel's own footprint, so
    # stars stay point-like at every resolution and under any magnification.
    # Star colour comes from the context's own LUT, so this works with no disc.
    star_tw = Float64(star_texture_weight)
    star_on = starfield
    if star_on
        # `starfield_mtl` scans a fixed 3x3 cell neighbourhood, which is only
        # valid while the PSF is narrower than one cell. The PSF width is
        # sigma = psf_pixels * 2 * fov_factor / height, so it grows as the
        # render gets shorter: what is 0.5 cells at 1440 rows is 9 cells at
        # 144, and every star gets truncated at the window edge into a clipped
        # square. Cap the density so a cell always covers 5 sigma.
        σ = star_psf_pixels * 2 * cam.fov_factor / height
        dmax = floor(2π / max(5σ, 1e-9))
        dens = min(Float64(star_density), dmax)
        dens < star_density &&
            @info "starfield density capped for this render height" star_density dens height
        set_starfield!(ctx; strength=1.0, texture_weight=star_tw,
                       height=height, fov_factor=cam.fov_factor,
                       psf_pixels=star_psf_pixels, density=dens)
    end

    GLFW.WindowHint(GLFW.CLIENT_API, GLFW.NO_API)
    win = GLFW.CreateWindow(winwidth, winheight, title)
    presenter = MetalPresenter(win, width, height; nearest=arcade)

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
    grade_rev = 0           # bumped on any grade change (re-presents stills)
    sky_mode = 0            # 0 procedural stars, 1 starmap, 2 both (B cycles)
    function apply_grade!(n)
        # Bloom thresholds sit above star brightness (~1 linear): the bloom
        # chain is quarter-res, and thresholds that let ordinary stars in
        # smear the whole background into soft blobs. Only HDR sources —
        # the disc, the ring — should scatter.
        if n == 1        # neutral: filmic curve only
            set_grade!(presenter; exposure=exposure, filmic=filmic,
                       crush=1.0, saturation=1.0, vignette=0.0,
                       wb=(1.0, 1.0, 1.0), bloom=0.0, hue_preserve=0.0,
                       gamma_power=0.4545, grain=0.0, streak_fraction=0.0,
                       streak_gain=1.0)
        elseif n == 2    # film: gentle warmth, bloom, vignette
            set_grade!(presenter; exposure=exposure, filmic=filmic,
                       crush=1.15, saturation=1.12, vignette=0.30,
                       wb=(1.02, 1.0, 0.97), bloom=0.8, bloom_threshold=1.5,
                       hue_preserve=0.3, gamma_power=0.4545, grain=0.0,
                       streak_fraction=0.0)
        elseif n == 3    # hectic: crushed, saturated, dripping bloom
            set_grade!(presenter; exposure=exposure, filmic=filmic,
                       crush=1.5, saturation=1.25, vignette=0.45,
                       wb=(1.05, 1.0, 0.94), bloom=1.6, bloom_threshold=1.2,
                       hue_preserve=0.3, gamma_power=0.4545, grain=0.0,
                       streak_fraction=0.0)
        else             # porthole: the escape-video recipe (postprocess():
                         # ev 2^0.8, hue-preserve 0.75, gamma 0.2 ⇒ x^5,
                         # streaks carry 2/3 of scattered light, ISO grain,
                         # vignette 0.3). Threshold raised from the video's
                         # 0.5 — quarter-res bloom must not catch stars.
            set_grade!(presenter; exposure=1.741 * exposure, filmic=true,
                       crush=1.0, saturation=0.94, vignette=0.30,
                       wb=(1.0, 1.0, 1.0), bloom=1.0, bloom_threshold=1.0,
                       hue_preserve=0.55, gamma_power=5.0, grain=0.02,
                       streak_fraction=0.667, streak_gain=4.0)
        end
        grade_rev += 1
        return nothing
    end

    # Layered engine state: deflection fan + the disc/gas layer. While
    # moving, the layer renders FRESH every frame at half display resolution
    # (a quarter-res rung when the frame runs hot) — no temporal history:
    # reprojected history echoes badly next to the photon ring, where the
    # parallax of wound light paths is extreme. At rest, one full-resolution
    # pass. Sky/shadow are always per-frame exact and native.
    sky = SkyFanState(n=fan_n)
    layer_on = disc !== nothing || volume !== nothing
    L = MtlArray{Float32,3}(undef, 4, width, height)          # rest: native
    # Moving rungs: gas layer resolution ladder, walked by a hysteresis
    # controller (dwell + separate up/down thresholds — flip-flopping
    # between rungs every few frames is worse than either rung).
    # The ladder has to reach far enough down to hold the frame budget with the
    # camera *inside* the gas, where the layer marches every step through dense
    # volume: quarter-res left the controller pinned at the bottom rung with
    # frames still over budget. Everything downstream is paced per frame — the
    # input poll included — so a rung that cannot hold the budget is felt as
    # latency, not just as a lower frame rate.
    rungs = [MtlArray{Float32,3}(undef, 4, cld(width, s), cld(height, s))
             for s in (2, 3, 4, 6, 8)]
    # Graphics quality: the bulk ladder is left deep in every preset — it is
    # what holds the frame budget — and the setting instead picks the stride of
    # the *ring* pass, a second finer trace over the wound rays that build the
    # photon ring. That is the one feature the bulk rungs cannot carry: it is
    # the highest-frequency thing in the frame, so downscaling it stair-steps
    # it while the smooth gas around it downscales for free.
    QUALITY = (:performance, :balanced, :quality)
    RING_STRIDE = Dict(:performance => 4, :balanced => 2, :quality => 1)
    quality in QUALITY ||
        throw(ArgumentError("quality must be one of $QUALITY, got $quality"))
    qi = findfirst(==(quality), QUALITY)
    ring_bufs = Dict{Int,MtlArray{Float32,3}}()
    ring_buf(s) = get!(ring_bufs, s) do
        # Seeded fully transparent, not `undef`: the ring pass dispatches only
        # the box that can hold wound rays, so every pixel outside it keeps
        # whatever the buffer already held. The composite never samples there
        # (its blend weight is the same periapsis test the pass culls on), but
        # a buffer that starts as garbage would make any future widening of
        # that box a debugging problem rather than a no-op.
        b = MtlArray{Float32,3}(undef, 4, cld(width, s), cld(height, s))
        seed = zeros(Float32, 4, cld(width, s), cld(height, s))
        seed[4, :, :] .= 1.0f0
        copyto!(b, seed)
        b
    end
    # The combined quality ladder the controller walks: (bulk rung, ring
    # stride), coarsest ring last and `0` for no ring pass at all. `quality`
    # sets the *finest* ring the ladder may use, not a fixed one — otherwise
    # the ring escapes the frame budget exactly where it is most expensive.
    function build_levels(q)
        f = RING_STRIDE[q]
        nr = length(rungs)
        lv = [(min(k, nr), f) for k in 1:nr]
        for m in (2, 4)
            push!(lv, (nr, min(f * m, 8)))
        end
        push!(lv, (nr, 0))
        return lv
    end
    LEVELS = build_levels(quality)
    level = 1
    rung = 1     # derived from LEVELS each frame; seed for the no-layer path
    last_rung_change = time()
    last_move_time = time()
    cur_stride = 2          # volumetric march stride currently set on ctx
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
    apply_grade!(2)   # default look: the film preset
    # Palette settings survive every later `apply_grade!`, because `set_grade!`
    # defaults them to whatever is already in the buffer.
    set_grade!(presenter; quantize=quantize, dither=dither)
    # A palette overrides per-channel quantization: it maps luminance onto one
    # ramp, which is what actually cuts the tone count. An Integer asks for
    # that many plasma steps with black at the bottom.
    palette === nothing || set_palette!(presenter,
        palette isa Integer ? plasma_palette(palette) : palette)
    let cam0 = build_cam()
        if arcade
            _trace_preview_gpu!(ctx, cam0, spacetime)
            Metal.synchronize()
            present!(presenter, ctx.out_gpu)
            @goto warmed
        end
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
        @label warmed
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
    # Input-pump deadline, in ms, tracked independently of `frame_ms` (see the
    # pump call site). Additive-increase to 90% of a measured GPU frame,
    # multiplicative-decrease whenever the pump outlasts the GPU.
    pump_ms = 8.0
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
        # Thin plane vs volumetric gas. 5 sits with the numeric row; V is the
        # original binding and still works.
        (pressed_once(GLFW.KEY_V) || pressed_once(GLFW.KEY_5)) &&
            ctx.has_volume && set_volume_enabled!(ctx, !ctx.vol_on[])
        pressed_once(GLFW.KEY_R) && (relativistic = !relativistic)
        # O cycles graphics quality (the photon-ring pass); it changes what is
        # rendered, so the still has to be thrown away and retraced.
        if pressed_once(GLFW.KEY_O)
            qi = qi % length(QUALITY) + 1
            LEVELS = build_levels(QUALITY[qi])
            level = clamp(level, 1, length(LEVELS))
            grade_rev += 1
        end
        pressed_once(GLFW.KEY_L) && (fisheye = fisheye > 0.0 ? 0.0 : 100.0)
        # B cycles the sky: procedural stars → the starmap → both. Direct A/B
        # is the only honest way to judge a starfield.
        if star_on && pressed_once(GLFW.KEY_B)
            sky_mode = (sky_mode + 1) % 3
            set_starfield!(ctx;
                           strength=sky_mode == 1 ? 0.0 : 1.0,
                           texture_weight=sky_mode == 0 ? star_tw : 1.0,
                           height=height, fov_factor=18.0 / focal,
                           psf_pixels=star_psf_pixels)
        end
        if pressed_once(GLFW.KEY_P)
            filmic = !filmic
            set_grade!(presenter; filmic=filmic)
            grade_rev += 1
        end
        if down(GLFW.KEY_T) || down(GLFW.KEY_G)
            exposure = down(GLFW.KEY_T) ? min(20.0f0, exposure * 1.04f0) :
                                          max(0.05f0, exposure / 1.04f0)
            set_grade!(presenter; exposure=exposure)
            grade_rev += 1
        end
        pressed_once(GLFW.KEY_1) && apply_grade!(1)
        pressed_once(GLFW.KEY_2) && apply_grade!(2)
        pressed_once(GLFW.KEY_3) && apply_grade!(3)
        pressed_once(GLFW.KEY_4) && apply_grade!(4)
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

        # Free-camera motion as a function of a time slice, so it can be run
        # both once per frame and again inside the GPU wait below. It re-reads
        # the key state and the camera basis every call, so a release lands on
        # the very next slice.
        function step_freecam!(dt)
            dt <= 0.0 && return
            f, rt, _ = _flycam_basis(state)
            v = speed * dt * (down(GLFW.KEY_LEFT_SHIFT) ? 5.0 : 1.0)
            rn = norm(state.pos)
            v *= clamp(0.12 * max(rn - 1.9 * M, 0.25 * rn), 0.02, 8.0)
            world_z = SVector(0.0, 0.0, 1.0)
            down(GLFW.KEY_W) && (state.pos += v * f)
            down(GLFW.KEY_S) && (state.pos -= v * f)
            down(GLFW.KEY_A) && (state.pos -= v * rt)
            down(GLFW.KEY_D) && (state.pos += v * rt)
            down(GLFW.KEY_Q) && (state.pos -= v * world_z)
            down(GLFW.KEY_E) && (state.pos += v * world_z)
            rn = norm(state.pos)
            rn < 0.45 * M && (state.pos *= 0.45 * M / rn)
            return
        end

        # The frame is enqueued in well under a millisecond and the CPU then
        # sits idle for the whole GPU frame — measured 98-99% idle at 1440p,
        # 0.4 ms of enqueue against 20-53 ms of waiting. Input was sampled once
        # per frame, so that entire idle window was also the input latency, and
        # the camera kept moving long after a key was released. Spending the
        # wait on `PollEvents` costs nothing that was being used. GLFW's own
        # guidance is that `PollEvents` — not `WaitEventsTimeout`, which sleeps
        # and adds latency of its own — is the primitive for continuous
        # rendering.
        function pump_input!(deadline)
            while time() < deadline
                GLFW.PollEvents()
                now = time()
                flight || step_freecam!(clamp(now - last_wall, 0.0, 0.02))
                last_wall = now
                sleep(0.001)
            end
            return
        end
        if flight
            # GR ship: the point-mass worldline of `src/ship.jl`.
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
            step_freecam!(dwall)
        end

        # --- render: fan + layer + composite, all on the GPU -----------
        # Change detection drives three states: moving (fast bounded gas
        # layer, 60 fps), just stopped (one full-native-resolution gas pass
        # — stills get offline sharpness), parked (nothing to render; the
        # layer retains the last drawable and the GPU idles).
        sig = (state.pos, state.yaw, state.pitch, state.roll, fisheye,
               relativistic, ctx.vol_on[], grade_rev, sky_mode)
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
            if arcade
                # One direct trace per frame, no fan and no layer ladder: at
                # arcade resolutions the whole frame is cheaper than the fan
                # alone would be, and there is nothing to schedule.
                _trace_preview_gpu!(ctx, cam_now, spacetime;
                                    fisheye_deg=fisheye,
                                    relativistic=relativistic)
            else
            update_sky_fan!(sky, ctx, state.pos, spacetime; gate=gate, dt=0.05)
            if layer_on
                # Fresh gas every frame — no history, nothing to echo. The
                # controller walks a combined ladder of (bulk rung, ring
                # stride): the ring pass has to be on it, because near the
                # hole the wound-ray annulus grows and those rays are the
                # dearest in the frame. Measured at 1440p with the bulk
                # pinned at stride 8, the bulk stays flat at 13-16 ms from
                # 26M down to 6M while a fixed stride-2 ring goes 20 -> 66 ms.
                # A ring outside the controller's authority means it drops the
                # bulk rung, which changes nothing, and the frame stays long.
                #
                # Down-steps take no dwell: entering the annulus can quadruple
                # the frame in one step, and one level per 0.25 s took seconds
                # to escape — which is felt as the camera failing to stop,
                # since input is polled once per rendered frame. Up-steps keep
                # the dwell so the ladder cannot flip-flop.
                # Two budgets. Dropping a *bulk* rung is nearly free
                # visually — the gas is smooth and downscales well — so it
                # happens at 22 ms. Coarsening the *ring* is the one step that
                # visibly costs, so it waits for a genuinely bad frame. That
                # split is only safe because input no longer rides on the frame
                # time; while it did, the controller had to buy latency with
                # image quality, and the ring was what it spent.
                budget = LEVELS[min(level + 1, length(LEVELS))][2] !=
                         LEVELS[level][2] ? 45.0 : 22.0
                if frame_ms > budget && level < length(LEVELS)
                    level += 1
                    last_rung_change = wall
                    frame_ms = 16.0    # reseed the EMA at the new level
                elseif frame_ms < 11.0 && level > 1 &&
                       wall - last_rung_change > 1.0
                    level -= 1
                    last_rung_change = wall
                    frame_ms = 16.0
                end
                rung, rs = LEVELS[level]
                # Coarser rungs also march the gas more coarsely — the
                # in-slab cost is dominated by volume sampling.
                want = rung == 1 ? 2 : 4
                if want != cur_stride
                    set_march_stride!(ctx, want)
                    cur_stride = want
                end
                render_layered_gpu!(ctx.out_gpu, rungs[rung],
                                    ctx, sky, cam_now, spacetime;
                                    fisheye_deg=fisheye,
                                    relativistic=relativistic,
                                    ring_out=rs > 0 ? ring_buf(rs) : nothing,
                                    ring_rmin=rs > 0 ? ring_rmin : 0.0)
            else
                render_layered_gpu!(ctx.out_gpu, L, ctx, sky, cam_now,
                                    spacetime; fisheye_deg=fisheye,
                                    relativistic=relativistic,
                                    trace_layer=false)
            end
            end
            # Keep input live while the GPU finishes. The deadline is its own
            # state, NOT a fraction of `frame_ms`: deriving it from the frame
            # time it is itself padding makes the loop self-sustaining. With
            # `deadline = t0 + 0.0009*frame_ms` and the pump inside the
            # measurement, a frame whose GPU cost collapses still decays at
            # only 1% per frame and settles at ten times the residual cost —
            # simulated, a 6.5 ms frame seeded at 16 ms still reads 15.0 ms
            # after 400 frames. The controller then spends image quality
            # paying off the loop's own padding.
            #
            # The signal that separates the two is how long `present!` blocks.
            # It waits on the drawable queue, which backs up behind the GPU and
            # the display refresh alike, so a real block means something other
            # than this pump is setting the pace — the pump is not the limiter
            # and may track the frame. Returning immediately means the pump WAS
            # the limiter, so back it off geometrically until a frame blocks
            # again. That converges in ~10 frames rather than ~400 and has no
            # spurious fixed point.
            #
            # Deliberately NOT a `Metal.synchronize()` here, which would be the
            # more direct measurement: it waits for our own command buffers and
            # so drains the pipeline every frame, costing the CPU/GPU overlap
            # the loop depends on. Measured at 1440p, that cost far more than
            # the better estimate was worth.
            pump_input!(t0 + 0.001 * pump_ms)
            t_present = time()
            present!(presenter, ctx.out_gpu)
            block_ms = 1000 * (time() - t_present)
            pump_ms = block_ms > PUMP_BLOCKED_MS ?
                      clamp(0.9 * frame_ms, PUMP_MIN_MS, PUMP_MAX_MS) :
                      max(PUMP_MIN_MS, PUMP_DECAY * pump_ms)
            frame_ms = 0.9 * frame_ms + 0.1 * 1000 * (time() - t0)
            last_move_time = time()
            nframes += 1
        elseif !arcade && passes < REFINE_PASSES &&
               wall - last_move_time > 0.15
            # At rest: time-sliced progressive refinement. Full-resolution
            # jittered passes (R2 low-discrepancy sequence) keep summing
            # into `accum` — every pass is real rays, so the still image
            # converges to a supersampled photograph — but the work is
            # submitted in row bands of ~7 ms and synchronized per band, so
            # the event loop keeps its cadence and any input aborts between
            # bands. Presented at pass boundaries; the presenter's exposure
            # factor divides by the pass count.
            t0 = time()
            if cur_stride != 2
                set_march_stride!(ctx, 2)   # stills refine at full quality
                cur_stride = 2
            end
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
                present!(presenter, accum; escale=1.0f0 / Float32(passes))
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
