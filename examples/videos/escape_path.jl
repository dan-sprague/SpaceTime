# Camera path for the "porthole escape" video.
# Keyframes: (t, pos, target, up, fisheye_deg). Catmull-Rom interpolation.
using StaticArrays, LinearAlgebra

# Retimed 2026-09-01: the periapsis pass (12.5M) formerly snapped from inbound
# to outbound across one key — under observer-frame relativity the velocity
# reversal reads as an abrupt aberration "wobble" in the hole's apparent size.
# Same waypoints and framing; the turn now gets ~4x more timeline, with an
# explicit early-outbound key rounding the corner.
const KEYS = [
    (0.00, SVector(2.5, 0.0, 0.0),   SVector(32.0, 0.0, 0.0),  SVector(0.0, 1.0, 0.0), 100.0),
    (0.15, SVector(7.0, 0.3, -0.8),  SVector(10.0, 0.5, -1.0), SVector(0.0, 1.0, 0.0),  95.0),
    (0.35, SVector(16.0, 1.0, -2.5), SVector(22.0, 2.5, -2.0), SVector(0.0, 1.0, 0.0),  80.0),
    (0.50, SVector(24.0, 2.0, -3.2), SVector(12.0, 1.0, -1.5), SVector(0.0, 1.0, 0.0),  60.0),
    (0.60, SVector(25.0, 1.8, -2.6), SVector(0.0, 0.0, 0.0),   SVector(0.0, 1.0, 0.0),  50.0),
    (0.72, SVector(15.0, 1.2, -0.6), SVector(0.0, 0.0, 0.0),   SVector(0.0, 0.9, 0.2),  42.0),
    (0.80, SVector(12.5, 1.1, 0.3),  SVector(0.0, 0.0, 0.0),   SVector(0.0, 0.7, 0.6),  40.0),
    (0.88, SVector(16.0, 1.1, 0.7),  SVector(0.0, 0.0, 0.0),   SVector(0.0, 0.55, 0.84), 35.0),
    (1.00, SVector(30.0, 1.1, 1.6),  SVector(0.0, 0.0, 0.0),   SVector(0.0, 0.342, 0.940), 29.0),
]

# Catmull-Rom on non-uniform keys, clamped ends.
function _cr(p0, p1, p2, p3, s)
    0.5 * ((2.0 * p1) + (-p0 + p2) * s + (2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * s^2 +
           (-p0 + 3.0*p1 - 3.0*p2 + p3) * s^3)
end

function path_at(t)
    t = clamp(t, 0.0, 1.0)
    # Gentle global ease-in/out.
    t = t * t * (3.0 - 2.0 * t)
    n = length(KEYS)
    k = findlast(K -> K[1] <= t, KEYS)
    k = min(k, n - 1)
    t1, t2 = KEYS[k][1], KEYS[k+1][1]
    s = (t - t1) / (t2 - t1)
    i0, i3 = max(k - 1, 1), min(k + 2, n)
    interp(f) = _cr(f(KEYS[i0]), f(KEYS[k]), f(KEYS[k+1]), f(KEYS[i3]), s)
    pos = interp(K -> K[2])
    tgt = interp(K -> K[3])
    up  = normalize(interp(K -> K[4]))
    fe  = interp(K -> K[5])
    # Keep the camera outside the GPU static-observer band and off the axis.
    r = norm(pos)
    r < 2.5 && (pos = pos * (2.5 / r))
    return pos, tgt, up, fe
end
