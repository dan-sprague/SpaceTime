# "Ship-cam" orbital path + jitter for the documentary orbit video.
# Realistic rate: r=30M circular orbit of a 1e5 Msun hole has T ≈ 8.4 min,
# so 30 s of footage sweeps ≈ 21 degrees.
using StaticArrays, LinearAlgebra, Random

const ARC_DEG = 21.0
const P0 = SVector(30.0, 1.1, 1.6)
const UP0 = SVector(0.0, sind(20.0), cosd(20.0))

_rz(φ) = SMatrix{3,3}(cos(φ), sin(φ), 0.0, -sin(φ), cos(φ), 0.0, 0.0, 0.0, 1.0)

# Ornstein-Uhlenbeck mount jitter: 3 channels (right, up, roll).
mutable struct Jitter
    x::SVector{3, Float64}
    rng::Xoshiro
end
Jitter(seed) = Jitter(SVector(0.0, 0.0, 0.0), Xoshiro(seed))

function step!(J::Jitter, dt; tau=0.4, rms=SVector(0.08, 0.08, deg2rad(0.15)))
    a = exp(-dt / tau)
    b = sqrt(1.0 - a * a)
    J.x = SVector(ntuple(i -> J.x[i] * a + rms[i] * b * randn(J.rng), 3))
    return J.x
end

"""Camera basis for frame fraction t ∈ [0,1] with jitter state applied."""
function orbit_cam(t, jx)
    φ = deg2rad(ARC_DEG) * t
    R = _rz(φ)
    pos = R * P0
    up = R * UP0
    fwd = normalize(-pos)
    right = normalize(cross(fwd, up))
    upl = cross(right, fwd)
    # Pointing jitter: offset the look target transverse to the view.
    tgt = SVector(0.0, 0.0, 0.0) + jx[1] * right + jx[2] * upl
    # Roll jitter: rotate up about fwd.
    c, s = cos(jx[3]), sin(jx[3])
    upj = normalize(c * upl + s * right)
    return pos, tgt, upj
end
