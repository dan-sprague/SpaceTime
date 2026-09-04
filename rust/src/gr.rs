//! Spacetimes and the camera tetrad, in Cartesian Kerr-Schild coordinates.
//!
//! Port of the host-side half of src/gr.jl and `ks_camera_tetrad` in
//! src/viewfinder.jl. Everything here is f64: it runs once per frame on one
//! worldline, not a million times on the GPU, and the tetrad's orthonormality
//! is what every per-ray quantity downstream is normalised against.

pub type V3 = [f64; 3];
pub type V4 = [f64; 4];

pub fn dot3(a: V3, b: V3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub fn norm3(a: V3) -> f64 { dot3(a, a).sqrt() }
pub fn scale3(a: V3, s: f64) -> V3 { [a[0] * s, a[1] * s, a[2] * s] }
pub fn add3(a: V3, b: V3) -> V3 { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
pub fn sub3(a: V3, b: V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
pub fn cross3(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn normalize3(a: V3) -> V3 {
    let n = norm3(a);
    if n > 0.0 { scale3(a, 1.0 / n) } else { a }
}

/// A stationary, axisymmetric black hole. `a = 0` is Schwarzschild; the
/// renderer specialises the kernel on that so the spherically symmetric case
/// never pays for Kerr's extra arithmetic.
#[derive(Clone, Copy, Debug)]
pub struct Spacetime {
    pub m: f64,
    pub a: f64,
}

impl Spacetime {
    pub fn schwarzschild(m: f64) -> Self { Self { m, a: 0.0 } }
    pub fn kerr(m: f64, a: f64) -> Self {
        assert!(a.abs() < m, "|a| must be < M");
        Self { m, a }
    }
    pub fn is_kerr(&self) -> bool { self.a != 0.0 }

    /// Radius of the prograde equatorial photon orbit,
    /// `r_ph = 2M[1 + cos(2/3 arccos(-a/M))]` -- 3M at a = 0, falling to M at
    /// a = M. This is the smallest periapsis any null geodesic reaching
    /// infinity can have, so anything below it is captured. It is a *global*
    /// bound (off-equatorial spherical photon orbits all sit higher), which is
    /// why the kernel needs no direction test alongside it.
    pub fn photon_orbit_min(&self) -> f64 {
        if self.a == 0.0 {
            3.0 * self.m
        } else {
            2.0 * self.m * (1.0 + ((2.0 / 3.0) * (-self.a / self.m).acos()).cos())
        }
    }

    /// Outer horizon `r+ = M + sqrt(M^2 - a^2)` in Kerr-Schild r.
    pub fn horizon(&self) -> f64 {
        self.m + (self.m * self.m - self.a * self.a).max(0.0).sqrt()
    }

    /// Spin and the padded squared capture radius for `spacetime_params[6..7]`.
    /// The 0.5% margin is slack for f32 error right at the boundary; without
    /// it, near-critical rays graze the coordinate ridge and escape as phantom
    /// sky inside the shadow.
    pub fn spin_horizon(&self) -> (f32, f32) {
        let rk = 0.995 * self.photon_orbit_min();
        (self.a as f32, (rk * rk) as f32)
    }
}

/// The camera observer freezes into a static frame outside this radius and is
/// a radial free-faller (dropped from rest here) inside it, so the exterior
/// view is unchanged and the frame stays physical through the horizon, where
/// no static observers exist.
const KS_FREEZE_R: f64 = 2.5;
const KS_FREEZE_E2: f64 = 1.0 - 2.0 / KS_FREEZE_R;

/// Orthonormal camera tetrad in Cartesian Kerr-Schild coordinates, as four
/// contravariant 4-vectors `(t, x, y, z)`: the observer `u`, then the camera's
/// forward / right / up axes Gram-Schmidt orthonormalised under the KS metric
/// **with forward first**, so the look direction is exact rather than
/// approximately preserved.
///
/// At `a = 0` the observer is the Schwarzschild static/freeze-faller. For
/// `a != 0` it is the exact Kerr observer: static (u proportional to d_t) where
/// f <= 0.72, smoothstep-blended to the ZAMO (zero angular momentum, corotating
/// at omega, no radial fall) by f = 0.88. The ZAMO is the natural hovering
/// frame inside the ergosphere and exists down to r+, where its lapse -> 0 and
/// the sky blueshift diverges -- physically what hovering at the horizon costs.
/// Port of `ks_camera_tetrad` in src/viewfinder.jl.
///
/// `beta` is the camera's 3-velocity resolved on its own (fwd, right, up)
/// axes. When it is non-zero the whole tetrad is Lorentz-boosted, so
/// aberration, motion Doppler and beaming all follow from the standard
/// machinery downstream -- p_t carries the full shift, and no shading code
/// needs to know the camera is moving.
pub fn ks_camera_tetrad(pos: V3, fwd: V3, right: V3, up: V3, m: f64, a: f64, beta: V3)
    -> (V4, V4, V4, V4)
{
    // `lvec` is the KS null 3-direction (position-unit at a = 0), `f` the KS
    // scalar, `u` the observer 4-velocity. Everything downstream is written in
    // terms of the general `ldot(A) = A_t + lvec . A_xyz` and `f`.
    let lvec: V3;
    let f: f64;
    let u: V4;
    if a == 0.0 {
        let r = norm3(pos);
        let xh = scale3(pos, 1.0 / r);
        f = 2.0 * m / r;
        // Radial-geodesic observer: E^2 = max(E_freeze^2, 1 - f) gives
        // dr/dtau = 0 exactly for r >= 2.5M, and the freeze-radius faller
        // inside.
        let e = KS_FREEZE_E2.max(1.0 - f).sqrt();
        let v = -(e * e - (1.0 - f)).max(0.0).sqrt();
        let w = (1.0 - e * (e - v)) / (e - v);
        let lu = e + w;
        u = [e + f * lu, (w - f * lu) * xh[0], (w - f * lu) * xh[1], (w - f * lu) * xh[2]];
        lvec = xh;
    } else {
        // Kerr-Schild null direction and scalar (same convention as kerr_rhs):
        // l_mu = (1, (rx+ay)/(r^2+a^2), (ry-ax)/(r^2+a^2), z/r), with r the KS
        // radius from the implicit quartic; f = 2 M r^3 / (r^4 + a^2 z^2).
        let (x, y, z) = (pos[0], pos[1], pos[2]);
        let a2 = a * a;
        let wq = x * x + y * y + z * z - a2;
        let rk2 = 0.5 * (wq + (wq * wq + 4.0 * a2 * z * z).sqrt());
        let rk = rk2.max(1.0e-12).sqrt();
        let ira = 1.0 / (rk2 + a2);
        lvec = [(rk * x + a * y) * ira, (rk * y - a * x) * ira, z / rk];
        let sig = rk2 * rk2 + a2 * z * z;
        f = 2.0 * m * rk2 * rk / sig;
        // ZAMO from t_BL = t_KS - A(r), A'(r) = 2 M r / Delta:
        // u_mu ~ -(dt - A' dr), raised with g^-1 = eta - f l(x)l.
        let delta = (rk2 - 2.0 * m * rk + a2).max(1.0e-6);
        let k = 2.0 * m * rk / delta;
        let grad_r: V3 = [x * rk2 * rk / sig, y * rk2 * rk / sig, z * rk * (rk2 + a2) / sig];
        let luc = 1.0 + k * dot3(lvec, grad_r);
        let uz: V4 = [
            1.0 + f * luc,
            k * grad_r[0] - f * luc * lvec[0],
            k * grad_r[1] - f * luc * lvec[1],
            k * grad_r[2] - f * luc * lvec[2],
        ];
        let lu_z = uz[0] + lvec[0] * uz[1] + lvec[1] * uz[2] + lvec[2] * uz[3];
        let nrm2 = -uz[0] * uz[0] + uz[1] * uz[1] + uz[2] * uz[2] + uz[3] * uz[3]
                   + f * lu_z * lu_z;
        let u_zamo: V4 = {
            let s = 1.0 / (-nrm2).sqrt();
            [uz[0] * s, uz[1] * s, uz[2] * s, uz[3] * s]
        };
        if f >= 0.88 {
            u = u_zamo;
        } else {
            let u_stat: V4 = [1.0 / (1.0 - f).sqrt(), 0.0, 0.0, 0.0];
            let mut wb = ((f - 0.72) / 0.16).clamp(0.0, 1.0);
            wb = wb * wb * (3.0 - 2.0 * wb);
            if wb == 0.0 {
                u = u_stat;
            } else {
                let um: V4 = [
                    (1.0 - wb) * u_stat[0] + wb * u_zamo[0],
                    wb * u_zamo[1],
                    wb * u_zamo[2],
                    wb * u_zamo[3],
                ];
                let lu = um[0] + lvec[0] * um[1] + lvec[1] * um[2] + lvec[2] * um[3];
                let nn = -um[0] * um[0] + um[1] * um[1] + um[2] * um[2] + um[3] * um[3]
                         + f * lu * lu;
                let s = 1.0 / (-nn).sqrt();
                u = [um[0] * s, um[1] * s, um[2] * s, um[3] * s];
            }
        }
    }

    let ldot = |a: V4| a[0] + lvec[0] * a[1] + lvec[1] * a[2] + lvec[2] * a[3];
    let gdot = |a: V4, b: V4| {
        -a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + f * ldot(a) * ldot(b)
    };
    let axpy = |a: V4, s: f64, b: V4| -> V4 {
        [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2], a[3] + s * b[3]]
    };
    let scal = |a: V4, s: f64| -> V4 { [a[0] * s, a[1] * s, a[2] * s, a[3] * s] };

    let mut ef: V4 = [0.0, fwd[0], fwd[1], fwd[2]];
    let mut er: V4 = [0.0, right[0], right[1], right[2]];
    let mut eu: V4 = [0.0, up[0], up[1], up[2]];

    ef = axpy(ef, gdot(ef, u), u);            // project out u (g(u,u) = -1)
    ef = scal(ef, 1.0 / gdot(ef, ef).sqrt());
    er = axpy(er, gdot(er, u), u);
    er = axpy(er, -gdot(er, ef), ef);
    er = scal(er, 1.0 / gdot(er, er).sqrt());
    eu = axpy(eu, gdot(eu, u), u);
    eu = axpy(eu, -gdot(eu, ef), ef);
    eu = axpy(eu, -gdot(eu, er), er);
    eu = scal(eu, 1.0 / gdot(eu, eu).sqrt());

    let b2raw = dot3(beta, beta);
    if b2raw > 1.0e-12 {
        let b2 = b2raw.min(0.9801);           // clamp |beta| <= 0.99
        let s = (b2 / b2raw).sqrt();
        let bv = scale3(beta, s);
        let gamma = 1.0 / (1.0 - b2).sqrt();
        // beta^i e_i as a 4-vector
        let be: V4 = [
            bv[0] * ef[0] + bv[1] * er[0] + bv[2] * eu[0],
            bv[0] * ef[1] + bv[1] * er[1] + bv[2] * eu[1],
            bv[0] * ef[2] + bv[1] * er[2] + bv[2] * eu[2],
            bv[0] * ef[3] + bv[1] * er[3] + bv[2] * eu[3],
        ];
        let k = (gamma - 1.0) / b2;
        let mix = |c: f64, e: V4| -> V4 {
            [e[0] + c * (k * be[0] + gamma * u[0]),
             e[1] + c * (k * be[1] + gamma * u[1]),
             e[2] + c * (k * be[2] + gamma * u[2]),
             e[3] + c * (k * be[3] + gamma * u[3])]
        };
        let ub = scal(axpy(u, 1.0, be), gamma);
        return (ub, mix(bv[0], ef), mix(bv[1], er), mix(bv[2], eu));
    }
    (u, ef, er, eu)
}
