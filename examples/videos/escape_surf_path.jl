# Rectilinear "surf" variant of the porthole escape:
#   A [0, 0.065]      hold the tilted marble (draft_render.png framing)
#   B [0.065, 0.165]  roll/pitch until the cone edge is a vertical wall
#                     (event_horizon.png framing), shadow to the right
#   C [0.165, 0.30]   surf: constant r = 2.5M, tangential flight through SURF_DPHI
#   D [0.30, 0.40]    pitch out to point radially outward
#   E [0.40, 1.0]     the original escape (escape_path.jl KEYS, rotated to the
#                     surf's exit longitude), fisheye angles mapped to fov_factor
# All segments rectilinear; returns (pos, tgt, up, fov_factor).
include(joinpath(@__DIR__, "escape_path.jl"))   # provides KEYS, _cr, path_at

const TA, TB, TC, TD = 0.065, 0.165, 0.30, 0.40
const SURF_R = 2.5
const SURF_DPHI = deg2rad(35.0)

# Start framing (tunable by eye against draft_render.png): marble up-left,
# rim sweeping the lower-left, mild roll.
const A_FWD = normalize(SVector(1.0, -0.38, -0.34))
const A_UP = normalize(SVector(0.15, 0.25, 1.0))
const WIDE_FOV = 1.30                      # ~105 deg horizontal, rectilinear

# Surf trims (tunable by eye against event_horizon.png).
const C_PITCH_OUT = deg2rad(6.0)           # lean slightly away from the hole
const C_UP = SVector(0.0, 0.0, -1.0)       # right-hand side faces the shadow

_rzm(φ) = SMatrix{3,3}(cos(φ), sin(φ), 0.0, -sin(φ), cos(φ), 0.0, 0.0, 0.0, 1.0)
_ss(x) = (x = clamp(x, 0.0, 1.0); x * x * x * (x * (6.0 * x - 15.0) + 10.0))  # smootherstep

function _slerp(a, b, s)
    d = clamp(dot(a, b), -1.0, 1.0)
    θ = acos(d)
    θ < 1.0e-6 && return normalize(a)
    normalize((sin((1.0 - s) * θ) * a + sin(s * θ) * b) / sin(θ))
end

# Surf orientation at azimuth φ: tangential look with a slight outward lean.
function _surf_frame(φ)
    r̂ = SVector(cos(φ), sin(φ), 0.0)
    t̂ = SVector(-sin(φ), cos(φ), 0.0)
    fwd = normalize(cos(C_PITCH_OUT) * t̂ + sin(C_PITCH_OUT) * r̂)
    return fwd, C_UP
end

function surf_path_at(t)
    t = clamp(t, 0.0, 1.0)
    if t < TD
        if t < TA                            # A: hold
            φ = 0.0
            fwd, up = A_FWD, A_UP
        elseif t < TB                        # B: roll marble -> wall
            φ = 0.0
            s = _ss((t - TA) / (TB - TA))
            fs, us = _surf_frame(0.0)
            fwd = _slerp(A_FWD, fs, s)
            up = _slerp(A_UP, us, s)
        elseif t < TC                        # C: surf at constant r
            φ = SURF_DPHI * _ss((t - TB) / (TC - TB) * 0.999)
            fwd, up = _surf_frame(φ)
        else                                 # D: pitch out to radial
            φ = SURF_DPHI
            s = _ss((t - TC) / (TD - TC))
            fs, us = _surf_frame(φ)
            r̂ = SVector(cos(φ), sin(φ), 0.0)
            e1 = _rzm(φ) * SVector(0.0, 1.0, 0.0)   # escape-start up, rotated
            fwd = _slerp(fs, r̂, s)
            up = _slerp(us, e1, s)
        end
        pos = SURF_R * SVector(cos(φ), sin(φ), 0.0)
        return pos, pos + 10.0 * fwd, up, WIDE_FOV
    end
    # E: the original escape, rotated to the surf's exit longitude.
    R = _rzm(SURF_DPHI)
    p, tg, u, fe = path_at((t - TD) / (1.0 - TD))
    fov = 0.55 + (fe - 29.0) * (WIDE_FOV - 0.55) / (100.0 - 29.0)
    return R * p, R * tg, R * u, fov
end

"""Ship 3-velocity for the surf path (same smoothing/limiting as dive_beta)."""
function surf_beta(t; T_M=233.0, h=8.0e-3)
    ta, tb = clamp(t - h, 0.0, 1.0), clamp(t + h, 0.0, 1.0)
    pa, _, _, _ = surf_path_at(ta)
    pb, _, _, _ = surf_path_at(tb)
    v = (pb - pa) / (max(tb - ta, 1.0e-9) * T_M)
    sp = norm(v)
    sp > 1.0e-9 && (v = v * (0.92 * tanh(sp / 0.92) / sp))
    return v
end
