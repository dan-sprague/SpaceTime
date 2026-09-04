//! Executable physics checks, in the spirit of test/runtests.jl: the port has
//! to reproduce numbers that are known in closed form, not merely produce a
//! picture that looks plausible.

use crate::camera::{lens, Camera};
use crate::gr::{ks_camera_tetrad, Spacetime, V3, V4};
use crate::renderer::{Renderer, RendererDesc};

/// g(A, B) in the Cartesian Kerr-Schild metric g = eta + f l(x)l at `pos`,
/// spin `a` -- the same chart the tetrad and kernel use. This is the
/// independent yardstick the tetrad's orthonormality is measured against.
fn kerr_gdot(pos: V3, a: f64, m: f64, av: V4, bv: V4) -> f64 {
    let (x, y, z) = (pos[0], pos[1], pos[2]);
    let a2 = a * a;
    let w = x * x + y * y + z * z - a2;
    let rk2 = 0.5 * (w + (w * w + 4.0 * a2 * z * z).sqrt());
    let rk = rk2.max(1.0e-12).sqrt();
    let ira = 1.0 / (rk2 + a2);
    let l: V4 = [1.0, (rk * x + a * y) * ira, (rk * y - a * x) * ira, z / rk];
    let f = 2.0 * m * rk2 * rk / (rk2 * rk2 + a2 * z * z);
    let la = l[0] * av[0] + l[1] * av[1] + l[2] * av[2] + l[3] * av[3];
    let lb = l[0] * bv[0] + l[1] * bv[1] + l[2] * bv[2] + l[3] * bv[3];
    -av[0] * bv[0] + av[1] * bv[1] + av[2] * bv[2] + av[3] * bv[3] + f * la * lb
}

/// The Kerr camera tetrad must be orthonormal under the Kerr metric -- g(u,u)
/// = -1, the three spatial legs unit and mutually orthogonal, all cross terms
/// zero -- at every pose, including deep inside the ergosphere where the
/// observer has blended to the ZAMO. A slip in the frame-dragging algebra
/// shows up here as an O(1) error long before it is visible in a frame.
fn kerr_tetrad_check() -> u32 {
    let (m, a) = (1.0, 0.998);
    let poses: [(&str, V3, V3); 4] = [
        ("hero r=30",     [30.0, 1.1, 1.6],   [0.0, 0.0, 0.0]),
        ("periapsis 3.2", [2.2, 2.2, 0.5],    [0.85, 0.2, -0.1]),
        ("ergo r=1.34",   [0.95, 0.95, -0.25],[0.55, -0.4, 0.1]),
        ("skim r=1.10",   [0.78, 0.78, 0.0],  [0.0, 0.6, 0.0]),
    ];
    let mut fails = 0;
    for (name, pos, beta) in poses {
        let fwd = crate::gr::normalize3([-pos[0], -pos[1], -pos[2]]);
        let right = crate::gr::normalize3(crate::gr::cross3(fwd, [0.0, 0.0, 1.0]));
        let up = crate::gr::cross3(right, fwd);
        let (u, ef, er, eu) = ks_camera_tetrad(pos, fwd, right, up, m, a, beta);
        let g = |x: V4, y: V4| kerr_gdot(pos, a, m, x, y);
        let errs = [
            g(u, u) + 1.0, g(ef, ef) - 1.0, g(er, er) - 1.0, g(eu, eu) - 1.0,
            g(u, ef), g(u, er), g(u, eu), g(ef, er), g(ef, eu), g(er, eu),
        ];
        let worst = errs.iter().fold(0.0f64, |m, &e| m.max(e.abs()));
        // 1e-9 is the f64 Gram-Schmidt floor; a real algebra error is O(1).
        let ok = worst < 1e-9;
        println!("{} Kerr tetrad {name:<14}: max|orthonorm err| = {:.2e}",
                 if ok { "ok  " } else { "FAIL" }, worst);
        if !ok { fails += 1; }
    }
    fails
}

/// Angular radius of the shadow for a static observer at `r`:
/// `sin(psi) = b_crit * sqrt(1 - 2M/r) / r` with the critical impact
/// parameter `b_crit = 3*sqrt(3)*M`.
fn shadow_angle(m: f64, r: f64) -> f64 {
    let b_crit = 3.0 * 3.0f64.sqrt() * m;
    (b_crit * (1.0 - 2.0 * m / r).sqrt() / r).asin()
}

pub fn run() {
    let mut failures = 0;

    // --- Kerr camera tetrad orthonormality -----------------------------
    failures += kerr_tetrad_check();
    println!();

    // --- Schwarzschild shadow -----------------------------------------
    // A white sky and no disc: every black pixel is a captured ray, so the
    // shadow's edge can be read straight off the centre row.
    let m = 1.0;
    let r_cam = 30.0;
    let st = Spacetime::schwarzschild(m);
    let (w, h) = (1201usize, 1201usize);
    let fov = lens(24.0);              // fov_factor = 0.75
    let sky = [1.0f32, 1.0, 1.0];

    let mut r = Renderer::new(&RendererDesc {
        width: w, height: h,
        background: (&sky, 1, 1),
        spacetime: st,
        disc: None,
        volume: None,
        dt: 0.05,
        nmax: 2000,
        r_escape_factor: 2.0,
        order: 4,
        tol: 1e-4,
        gas_stride: 0.32,
        binary: None,
    });
    let cam = Camera::look_at([r_cam, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], fov);
    let t0 = std::time::Instant::now();
    let img = r.trace_blocking(&cam, &st);
    let trace_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Walk the centre row outward from the middle; the first lit pixel is the
    // shadow edge.
    let jc = h / 2;
    let ic = w / 2;
    let mut edge = None;
    for i in ic..w {
        if img[3 * (i + w * jc)] > 0.5 {
            edge = Some(i);
            break;
        }
    }
    match edge {
        None => { println!("FAIL shadow: no sky found on the centre row"); failures += 1; }
        Some(i) => {
            // Pixel centres, matching the kernel's sensor coordinate.
            let u = (i as f64 + 0.5 - w as f64 / 2.0) / (h as f64 / 2.0);
            let measured = (u * fov).atan();
            let expected = shadow_angle(m, r_cam);
            // One pixel of angular width is the resolution floor of this test.
            let px = (fov * 2.0 / h as f64) / (1.0 + (u * fov).powi(2));
            let err = (measured - expected).abs();
            let ok = err < 2.0 * px;
            println!("{} shadow at r={r_cam}M: measured {:.6} rad, exact {:.6} rad, \
                      err {:.2e} ({:.2} px)",
                     if ok { "ok  " } else { "FAIL" }, measured, expected, err, err / px);
            if !ok { failures += 1; }
        }
    }

    // --- Kerr reduces to Schwarzschild at a = 0 ------------------------
    // The Kerr RHS is a separate code path; at a = 0 it must reproduce the
    // Schwarzschild kernel, which is the cheapest way to catch an algebra slip
    // in the frame-dragging terms.
    let st_k = Spacetime { m, a: 1e-9 };
    let mut rk = Renderer::new(&RendererDesc {
        width: w, height: h,
        background: (&sky, 1, 1),
        spacetime: st_k,
        disc: None,
        volume: None,
        dt: 0.05,
        nmax: 2000,
        r_escape_factor: 2.0,
        order: 4,
        tol: 1e-4,
        gas_stride: 0.32,
        binary: None,
    });
    let img_k = rk.trace_blocking(&cam, &st_k);
    let mut worst = 0.0f32;
    let mut diff_px = 0usize;
    for k in 0..img.len() {
        let d = (img[k] - img_k[k]).abs();
        if d > worst { worst = d; }
        if d > 0.5 { diff_px += 1; }
    }
    let frac = diff_px as f64 / (w * h) as f64;
    let ok = frac < 1e-4;
    println!("{} Kerr(a=1e-9) vs Schwarzschild: {diff_px} of {} pixels differ ({:.2e})",
             if ok { "ok  " } else { "FAIL" }, w * h, frac);
    if !ok { failures += 1; }

    // --- shadow grows as the camera falls in ---------------------------
    // sin(psi) = b_crit*sqrt(1-2M/r)/r is monotonically decreasing in r above
    // 3M, so the shadow must swell as the camera approaches.
    let mut last = 0.0;
    let mut monotone = true;
    for &rr in &[40.0, 30.0, 20.0, 12.0, 8.0] {
        let c = Camera::look_at([rr, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], fov);
        let im = r.trace_blocking(&c, &st);
        let mut e = w;
        for i in ic..w {
            if im[3 * (i + w * jc)] > 0.5 { e = i; break; }
        }
        let u = (e as f64 + 0.5 - w as f64 / 2.0) / (h as f64 / 2.0);
        let psi = (u * fov).atan();
        let exact = shadow_angle(m, rr);
        println!("      r={rr:>5}M  measured {:.5}  exact {:.5}", psi, exact);
        if psi <= last { monotone = false; }
        last = psi;
    }
    println!("{} shadow grows monotonically as r falls",
             if monotone { "ok  " } else { "FAIL" });
    if !monotone { failures += 1; }

    // --- what uncapped adaptive stepping costs -------------------------
    // The RK45 controller bounds the ODE's local truncation error, which is
    // NOT the same thing as bounding the renderer's error: the shadow-kill
    // test is sampled once per step, so a long step jumps the band it
    // watches and a captured ray escapes as phantom sky. This isolates that
    // from the disc and gas sampling -- sky only, no disc, no volume.
    println!();
    for (name, order, tol) in [("RK45 capped  ", 45, 1e-4f32),
                               ("RK45 uncapped", 46, 1e-3),
                               ("RK45 uncapped", 46, 1e-7)] {
        let mut ra = Renderer::new(&RendererDesc {
            width: w, height: h,
            background: (&sky, 1, 1),
            spacetime: st,
            disc: None,
            volume: None,
            dt: 0.05,
            nmax: 2000,
            r_escape_factor: 2.0,
            order,
            tol,
        gas_stride: 0.32,
        binary: None,
        });
        let im = ra.trace_blocking(&cam, &st);
        let mut e = w;
        for i in ic..w {
            if im[3 * (i + w * jc)] > 0.5 { e = i; break; }
        }
        let u = (e as f64 + 0.5 - w as f64 / 2.0) / (h as f64 / 2.0);
        let psi = (u * fov).atan();
        let exact = shadow_angle(m, r_cam);
        let px = (fov * 2.0 / h as f64) / (1.0 + (u * fov).powi(2));
        // How many pixels inside the shadow leaked sky: the phantom-sky
        // failure mode, which a shadow-edge measurement alone would miss.
        let mut leaked = 0usize;
        for j in 0..h {
            for i in 0..w {
                let du = (i as f64 + 0.5 - w as f64 / 2.0) / (h as f64 / 2.0);
                let dv = (j as f64 + 0.5 - h as f64 / 2.0) / (h as f64 / 2.0);
                // well inside the analytic shadow
                if ((du * du + dv * dv).sqrt() * fov).atan() < 0.9 * exact
                    && im[3 * (i + w * j)] > 0.5 { leaked += 1; }
            }
        }
        println!("{name} tol {tol:>7.0e}: shadow {:.5} rad (exact {:.5}, err {:.1} px), \
                  {leaked} px of phantom sky inside the shadow",
                 psi, exact, (psi - exact).abs() / px);
    }

    println!("\n{w}x{h} Schwarzschild trace, dt=0.05: {trace_ms:.1} ms");
    if failures == 0 {
        println!("all checks passed");
    } else {
        println!("{failures} check(s) FAILED");
        std::process::exit(1);
    }
}
