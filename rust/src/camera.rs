//! The camera: a pose plus a field of view, packed into the GPU's parameter
//! block as an orthonormal tetrad. Port of src/camera.jl's `Camera` and
//! `_ks_cam_params!` in src/metal.jl.

use crate::gr::*;

/// Length of the camera parameter buffer actually consumed by the ported
/// kernel: pose (position, fov, tetrad) plus the projection selector.
/// The Julia buffer is 49 long because it also carries the thin-lens stratum
/// and the motion-blur end pose, neither of which the real-time path uses.
pub const CAM_PARAMS_N: usize = 22;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub pos: V3,
    pub fwd: V3,
    pub right: V3,
    pub up_local: V3,
    /// `tan(half_fov)`: the half-height of the sensor plane at unit distance.
    pub fov_factor: f64,
    /// World-frame 3-velocity in units of c. Stored in world coordinates
    /// because that is what it physically is -- how the ship is moving,
    /// independent of where it is pointing.
    pub velocity: V3,
}

impl Camera {
    pub fn look_at(pos: V3, target: V3, up: V3, fov_factor: f64) -> Self {
        let fwd = normalize3(sub3(target, pos));
        let right = normalize3(cross3(fwd, up));
        let up_local = cross3(right, fwd);
        Self { pos, fwd, right, up_local, fov_factor, velocity: [0.0; 3] }
    }

    /// Build the pose from yaw/pitch/roll about the world +z axis, matching
    /// the flight loop in src/native_shell.jl.
    pub fn set_orientation(&mut self, yaw: f64, pitch: f64, roll: f64) {
        let (cy, sy) = (yaw.cos(), yaw.sin());
        let (cp, sp) = (pitch.cos(), pitch.sin());
        let fwd = [cp * cy, cp * sy, sp];
        let world_z = [0.0, 0.0, 1.0];
        let mut right = normalize3(cross3(fwd, world_z));
        let mut up = cross3(right, fwd);
        if roll != 0.0 {
            let (cr, sr) = (roll.cos(), roll.sin());
            let r2 = add3(scale3(right, cr), scale3(up, sr));
            let u2 = sub3(scale3(up, cr), scale3(right, sr));
            right = r2;
            up = u2;
        }
        self.fwd = fwd;
        self.right = right;
        self.up_local = up;
    }

    /// The 3-velocity resolved onto the camera's own (forward, right, up)
    /// axes, which is the basis `ks_camera_tetrad` boosts in.
    pub fn beta(&self) -> V3 {
        [dot3(self.velocity, self.fwd),
         dot3(self.velocity, self.right),
         dot3(self.velocity, self.up_local)]
    }

    /// Fill the GPU camera block. `fisheye_deg > 0` selects an equidistant
    /// fisheye with that vertical half-angle at the top edge of the image, so
    /// fields wider than 180 degrees render cleanly -- a rectilinear pinhole
    /// cannot reach 180 at any focal length.
    pub fn write_params(&self, dest: &mut [f32], m: f64, a: f64, fisheye_deg: f64) {
        let (u4, ef, er, eu) =
            ks_camera_tetrad(self.pos, self.fwd, self.right, self.up_local, m, a, self.beta());
        dest[0] = self.pos[0] as f32;
        dest[1] = self.pos[1] as f32;
        dest[2] = self.pos[2] as f32;
        dest[3] = self.fov_factor as f32;
        for k in 0..4 {
            dest[4 + k] = ef[k] as f32;
            dest[8 + k] = er[k] as f32;
            dest[12 + k] = eu[k] as f32;
            dest[16 + k] = u4[k] as f32;
        }
        dest[20] = if fisheye_deg > 0.0 { 1.0 } else { 0.0 };
        dest[21] = fisheye_deg.max(0.0).to_radians() as f32;
    }
}

/// A lens named by focal length in mm on a 36 mm sensor, as in src/raytrace.jl.
pub fn lens(focal_length_mm: f64) -> f64 { 18.0 / focal_length_mm }
