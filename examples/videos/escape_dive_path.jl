# Camera path for the "porthole escape + dive" video: the original escape arc
# (escape_path.jl, untouched) compressed into t ∈ [0, 0.62], a beat of
# stillness at the hero framing, then a committed dive back through the disc,
# across the horizon, to r = 0.55M where the frame dies to black.
include(joinpath(@__DIR__, "escape_path.jl"))   # provides KEYS, _cr, path_at

const DIVE_T0 = 0.62

# (t, pos, target, up, fisheye_deg) — first key matches the escape's end state.
const DIVE_KEYS = [
    (0.620, SVector(30.0, 1.1, 1.6),    SVector(0.0, 0.0, 0.0), SVector(0.0, 0.342, 0.940), 29.0),
    (0.700, SVector(27.5, 1.0, 1.45),   SVector(0.0, 0.0, 0.0), SVector(0.0, 0.38, 0.925),  33.0),
    (0.800, SVector(19.0, 0.75, 1.0),   SVector(0.0, 0.0, 0.0), SVector(0.0, 0.48, 0.88),   45.0),
    (0.880, SVector(10.0, 0.45, 0.55),  SVector(0.0, 0.0, 0.0), SVector(0.0, 0.60, 0.80),   62.0),
    (0.940, SVector(4.2, 0.2, 0.22),    SVector(0.0, 0.0, 0.0), SVector(0.0, 0.72, 0.69),   80.0),
    (0.975, SVector(1.7, 0.07, 0.08),   SVector(0.0, 0.0, 0.0), SVector(0.0, 0.82, 0.57),   92.0),
    (1.000, SVector(0.55, 0.02, 0.02),  SVector(0.0, 0.0, 0.0), SVector(0.0, 0.86, 0.51),  100.0),
]

function dive_path_at(t)
    t = clamp(t, 0.0, 1.0)
    t <= DIVE_T0 && return path_at(t / DIVE_T0)
    n = length(DIVE_KEYS)
    k = findlast(K -> K[1] <= t, DIVE_KEYS)
    k = min(k, n - 1)
    t1, t2 = DIVE_KEYS[k][1], DIVE_KEYS[k+1][1]
    s = (t - t1) / (t2 - t1)
    i0, i3 = max(k - 1, 1), min(k + 2, n)
    interp(f) = _cr(f(DIVE_KEYS[i0]), f(DIVE_KEYS[k]), f(DIVE_KEYS[k+1]),
                    f(DIVE_KEYS[i3]), s)
    pos = interp(K -> K[2])
    tgt = interp(K -> K[3])
    up = normalize(interp(K -> K[4]))
    fe = interp(K -> K[5])
    return pos, tgt, up, fe   # no radius clamp: this path crosses the horizon
end

"""
    dive_beta(t; T_M, h) -> SVector{3} (coordinate-frame velocity, |β| clamped)

Camera 3-velocity from the path by central finite difference, with the whole
video spanning `T_M` units of M-time (c = 1). Tapered to zero over
r ∈ [3.5M, 2.5M]: inside 2.5M the renderer's tetrad is already a radial
free-faller carrying its own physical velocity, so the scripted boost hands
over smoothly instead of double-counting.
"""
function dive_beta(t; T_M=220.0, h=8.0e-3)
    # Wide central difference: h spans ~±7 frames, low-passing the Catmull-Rom
    # acceleration kinks at keyframes — under aberration a β kink is a visible
    # whole-sky "snap" between frames.
    ta, tb = clamp(t - h, 0.0, 1.0), clamp(t + h, 0.0, 1.0)
    pa, _, _, _ = dive_path_at(ta)
    pb, _, _, _ = dive_path_at(tb)
    v = (pb - pa) / (max(tb - ta, 1.0e-9) * T_M)
    sp = norm(v)
    # Smooth speed limit (C^∞): ≈ identity below ~0.5c, asymptotes to 0.92c.
    sp > 1.0e-9 && (v = v * (0.92 * tanh(sp / 0.92) / sp))
    pos, _, _, _ = dive_path_at(t)
    x = clamp(norm(pos) - 2.5, 0.0, 1.0)
    v * (x * x * (3.0 - 2.0 * x))   # smoothstep handoff to the free-faller
end
