//! A precomputed camera track: a geodesic exported from the Julia solver, flown
//! back as a cinematic path. Each sample is a time, a world position, and the
//! world 3-velocity (in units of c) at that point; the look direction is the
//! velocity, and the camera's relativistic tetrad is boosted by it, so playing
//! the track shows the true aberration of riding the geodesic.
//!
//! File format (whitespace floats, `#` lines are metadata):
//!   # schwarzschild M=1.0 samples=... duration=...s
//!   # up ux uy uz
//!   t  x y z  vx vy vz
//!   ...

use crate::gr::{normalize3, V3};
use std::fs;

#[derive(Clone, Copy)]
struct Sample {
    t: f64,
    pos: V3,
    vel: V3,
}

pub struct Trajectory {
    samples: Vec<Sample>,
    /// Orbital-plane normal, used as the camera's up reference.
    pub up: V3,
    pub duration: f64,
    /// Holes named in the track: `(center, mass)`. A slingshot track declares
    /// both black holes here (hole 1 at the origin, hole 2 out in the gap), so
    /// the app can render the Binary scene with hole 2 where the geodesic
    /// actually flew past it, rather than the app's default binary layout.
    pub holes: Vec<(V3, f64)>,
}

impl Trajectory {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut samples = Vec::new();
        let mut up = [0.0, 0.0, 1.0];
        let mut holes: Vec<(V3, f64)> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix('#') {
                // metadata: "up ux uy uz" and any number of "hole x y z m".
                let mut it = rest.split_whitespace();
                match it.next() {
                    Some("up") => {
                        let v: Vec<f64> = it.filter_map(|s| s.parse().ok()).collect();
                        if v.len() == 3 {
                            up = normalize3([v[0], v[1], v[2]]);
                        }
                    }
                    Some("hole") => {
                        let v: Vec<f64> = it.filter_map(|s| s.parse().ok()).collect();
                        if v.len() == 4 {
                            holes.push(([v[0], v[1], v[2]], v[3]));
                        }
                    }
                    _ => {}
                }
                continue;
            }
            let f: Vec<f64> = line.split_whitespace().filter_map(|s| s.parse().ok()).collect();
            if f.len() >= 7 {
                samples.push(Sample {
                    t: f[0],
                    pos: [f[1], f[2], f[3]],
                    vel: [f[4], f[5], f[6]],
                });
            }
        }
        if samples.len() < 2 {
            return Err(format!("{path}: need at least 2 samples, got {}", samples.len()));
        }
        let duration = samples.last().unwrap().t;
        Ok(Self { samples, up, duration, holes })
    }

    pub fn len(&self) -> usize { self.samples.len() }

    /// The track's positions, for drawing the flight path on the system map.
    pub fn positions(&self) -> Vec<V3> { self.samples.iter().map(|s| s.pos).collect() }

    /// Position and look direction (unit) at playback time `t` (seconds),
    /// clamped to the track's ends. Returns `(pos, forward, velocity)`.
    pub fn at(&self, t: f64) -> (V3, V3, V3) {
        let t = t.clamp(self.samples[0].t, self.samples.last().unwrap().t);
        // linear scan is fine: a few hundred samples, called once per frame.
        let mut i = 0;
        while i + 1 < self.samples.len() && self.samples[i + 1].t < t {
            i += 1;
        }
        let a = &self.samples[i];
        let b = &self.samples[(i + 1).min(self.samples.len() - 1)];
        let span = (b.t - a.t).max(1e-9);
        let f = ((t - a.t) / span).clamp(0.0, 1.0);
        let lerp = |p: V3, q: V3| [p[0] + (q[0] - p[0]) * f,
                                   p[1] + (q[1] - p[1]) * f,
                                   p[2] + (q[2] - p[2]) * f];
        let pos = lerp(a.pos, b.pos);
        let vel = lerp(a.vel, b.vel);
        let fwd = normalize3(vel);
        (pos, fwd, vel)
    }
}
