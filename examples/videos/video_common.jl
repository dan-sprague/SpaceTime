# Shared helpers for the video render scripts.
using FileIO
using FreeTypeAbstraction
using Images: RGB

"""
    frame_t(f, nframes)

Normalised position of frame `f` in a sequence of `nframes`, in `[0, 1]`.

The obvious `(f - 1) / (nframes - 1)` is `0/0` for a single frame, and the NaN
propagates all the way into the renderer, which fails on an `InexactError`
several call frames from the cause. `NFRAMES=1` is the natural way to check a
look, so it should render the first frame rather than crash.
"""
frame_t(f::Integer, nframes::Integer) =
    nframes <= 1 ? 0.0 : (f - 1) / (nframes - 1)

"""
    frame_span(total, nframes)

`total` divided over the gaps between `nframes` frames — the per-frame step for
a quantity that spans `total` across the whole sequence. Guards the same
single-frame division as [`frame_t`](@ref).
"""
frame_span(total::Real, nframes::Integer) = total / max(nframes - 1, 1)

# --- Telemetry HUD -----------------------------------------------------------
const _HUD_FACE = Ref{Any}(nothing)
function _hud_face()
    if _HUD_FACE[] === nothing
        # Menlo first: Monaco serves embedded mono bitmaps at some small sizes,
        # which FreeTypeAbstraction cannot rasterize.
        for p in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Monaco.ttf")
            isfile(p) && (_HUD_FACE[] = FTFont(p); break)
        end
    end
    return _HUD_FACE[]
end

"""
    draw_hud!(img, lines)

Draw telemetry `lines` into a display-oriented (rows × cols) image: pale steel
monospace text with a soft drop shadow, upper-left, sized relative to frame
height. Mutates and returns `img`; no-op if no system font is found.
"""
function draw_hud!(img, lines; px=max(12, round(Int, size(img, 1) / 40)),
                   color=RGB{Float32}(0.79, 0.84, 0.92), alpha=0.55f0)
    face = _hud_face()
    face === nothing && return img
    nr, nc = size(img)
    x0 = round(Int, nr / 20)
    y = round(Int, nr / 22) + px
    lh = round(Int, 1.5 * px)
    sh = zeros(Float32, nr, nc)   # shadow coverage
    fg = zeros(Float32, nr, nc)   # glyph coverage
    for s in lines
        renderstring!(sh, s, face, px, y + 1, x0 + 1; fcolor=1.0f0)
        renderstring!(fg, s, face, px, y, x0; fcolor=1.0f0)
        y += lh
    end
    T = eltype(img)
    @inbounds for idx in eachindex(img)
        sv = sh[idx] * 0.5f0 * alpha
        fv = fg[idx] * alpha
        if sv > 0.002f0 || fv > 0.002f0
            c = img[idx]
            img[idx] = T(c.r * (1 - sv) * (1 - fv) + color.r * fv,
                         c.g * (1 - sv) * (1 - fv) + color.g * fv,
                         c.b * (1 - sv) * (1 - fv) + color.b * fv)
        end
    end
    return img
end

"""
    hud_region(r) -> label

Regime label for Schwarzschild radius `r` in units of M.
"""
hud_region(r) = r < 2.0 ? "INSIDE EVENT HORIZON" :
                r < 3.0 ? "INSIDE PHOTON SPHERE" :
                r < 6.0 ? "BELOW ISCO" :
                r < 20.0 ? "ACCRETION DISC" : "OPEN SPACE"

# Lossless-compress each linear master as soon as it is written. libtiff's
# tiffcp (brew install libtiff) rewrites the frame with deflate + the TIFF
# floating-point predictor — bit-identical Float32, still a plain TIFF any
# grading app reads. Render sampling noise limits the ratio to ~1.3x on real
# frames (the noise is true entropy). If tiffcp is missing the frame is simply
# left uncompressed.
const TIFFCP = Sys.which("tiffcp")

function save_master(path::AbstractString, img)
    save(path, img)
    TIFFCP === nothing && return
    tmp = path * ".ztmp.tiff"
    try
        run(pipeline(`$TIFFCP -c zip:3:p9 $path $tmp`, stderr=devnull))
        mv(tmp, path; force=true)
    catch
        rm(tmp; force=true)   # keep the uncompressed frame on any failure
    end
    return
end
