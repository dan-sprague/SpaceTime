//! A little 3-D system map, software-rasterised on the CPU into a BGRA panel
//! that the renderer blits into a corner of the drawable. It orbits slowly so
//! the equatorial disc and the flight path read as 3-D, and it marks where the
//! camera is in the system right now.
//!
//! Deliberately simple: an orthographic orbit projection and hand-drawn lines.
//! It is redrawn every frame (a few thousand pixels), which is free.

use crate::gr::{dot3, V3};

pub struct Minimap {
    pub w: usize,
    pub h: usize,
    /// BGRA8, row-major, top-left origin -- the drawable's own format.
    pub buf: Vec<u8>,
}

type Bgr = [u8; 3];
const BG: Bgr = [20, 14, 10];
const BORDER: Bgr = [70, 58, 46];
const GRID: Bgr = [46, 38, 30];
const AXIS: Bgr = [90, 74, 58];
const DISC: Bgr = [40, 120, 235]; // amber (B,G,R)
const PATH: Bgr = [235, 165, 95]; // pale blue-cyan
const HOLE_RIM: Bgr = [255, 200, 94]; // cyan #5ec8ff
const HOLE_FILL: Bgr = [12, 7, 5];
const CAM: Bgr = [120, 240, 255]; // bright warm-white

impl Minimap {
    pub fn new(w: usize, h: usize) -> Self {
        Self { w, h, buf: vec![0; w * h * 4] }
    }

    fn px(&mut self, x: i32, y: i32, c: Bgr) {
        if x < 0 || y < 0 || x as usize >= self.w || y as usize >= self.h {
            return;
        }
        let i = (y as usize * self.w + x as usize) * 4;
        self.buf[i] = c[0];
        self.buf[i + 1] = c[1];
        self.buf[i + 2] = c[2];
        self.buf[i + 3] = 255;
    }

    fn dot(&mut self, x: i32, y: i32, r: i32, c: Bgr) {
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    self.px(x + dx, y + dy, c);
                }
            }
        }
    }

    fn line(&mut self, mut x0: i32, mut y0: i32, x1: i32, y1: i32, c: Bgr) {
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.px(x0, y0, c);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x0 += sx;
            }
            if e2 <= dx {
                err += dx;
                y0 += sy;
            }
        }
    }

    /// Render the whole panel. `az` is the orbit angle (radians); `holes` are
    /// `(centre, mass)`; `disc` is `(inner, outer)` radius in M; `path` is the
    /// flight track (or none); `cam_*` mark the current pose.
    #[allow(clippy::too_many_arguments)]
    pub fn render(&mut self, az: f64, holes: &[(V3, f64)], disc: (f64, f64),
                  path: Option<&[V3]>, cam_pos: V3, cam_fwd: V3, fit: f64,
                  center: V3) {
        // clear + border
        for i in (0..self.buf.len()).step_by(4) {
            self.buf[i] = BG[0]; self.buf[i + 1] = BG[1];
            self.buf[i + 2] = BG[2]; self.buf[i + 3] = 255;
        }
        let (w, h) = (self.w as i32, self.h as i32);
        for x in 0..w { self.px(x, 0, BORDER); self.px(x, h - 1, BORDER); }
        for y in 0..h { self.px(0, y, BORDER); self.px(w - 1, y, BORDER); }

        // orthographic orbit projection
        let el = 0.52_f64; // elevation
        let (sa, ca) = (az.sin(), az.cos());
        let (se, ce) = (el.sin(), el.cos());
        let rx: V3 = [-sa, ca, 0.0];
        let ry: V3 = [-se * ca, -se * sa, ce];
        let cx = self.w as f64 * 0.5;
        let cy = self.h as f64 * 0.52;
        let scale = (self.w.min(self.h) as f64 * 0.5 - 10.0) / fit;
        // Everything is drawn relative to `center`, so a widely-separated binary
        // sits centred rather than shoved to one edge. Single-hole scenes pass
        // the origin and are unaffected.
        let proj = |p: V3| -> (i32, i32) {
            let q = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
            ((cx + dot3(q, rx) * scale).round() as i32,
             (cy - dot3(q, ry) * scale).round() as i32)
        };
        let ring = |rad: f64, zc: f64| -> Vec<V3> {
            (0..=72).map(|k| {
                let t = k as f64 / 72.0 * std::f64::consts::TAU;
                [rad * t.cos(), rad * t.sin(), zc]
            }).collect()
        };
        let mut polyline = |mm: &mut Self, pts: &[V3], c: Bgr, thick: bool| {
            for k in 1..pts.len() {
                let (a, b) = (proj(pts[k - 1]), proj(pts[k]));
                mm.line(a.0, a.1, b.0, b.1, c);
                if thick { mm.line(a.0, a.1 + 1, b.0, b.1 + 1, c); }
            }
        };

        // faint ground grid (equatorial plane) + vertical axis
        for r in [10.0, 20.0, 30.0] { polyline(self, &ring(r, 0.0), GRID, false); }
        let o = proj([0.0, 0.0, 0.0]);
        let up = proj([0.0, 0.0, 8.0]);
        self.line(o.0, o.1, up.0, up.1, AXIS);

        // accretion disc (inner + outer rim)
        if disc.1 > disc.0 {
            polyline(self, &ring(disc.0, 0.0), DISC, false);
            polyline(self, &ring(disc.1, 0.0), DISC, false);
        }

        // flight path
        if let Some(p) = path { polyline(self, p, PATH, true); }

        // holes: a black disc with a bright double rim
        for (c, m) in holes {
            let (hx, hy) = proj(*c);
            let rad = ((2.0 * m) * scale).round().max(5.0) as i32;
            self.dot(hx, hy, rad + 1, HOLE_FILL);
            for ring_r in [rad, rad - 1] {
                for k in 0..64 {
                    let t = k as f64 / 64.0 * std::f64::consts::TAU;
                    self.px(hx + (ring_r as f64 * t.cos()).round() as i32,
                            hy + (ring_r as f64 * t.sin()).round() as i32, HOLE_RIM);
                }
            }
        }

        // camera position + heading tick
        let (px, py) = proj(cam_pos);
        let tip = proj([cam_pos[0] + cam_fwd[0] * 5.0,
                        cam_pos[1] + cam_fwd[1] * 5.0,
                        cam_pos[2] + cam_fwd[2] * 5.0]);
        self.line(px, py, tip.0, tip.1, CAM);
        self.dot(px, py, 3, CAM);
        self.dot(px, py, 1, [255, 255, 255]);
    }
}
