//! Where the frame time actually goes, measured rather than assumed.
//!
//! Every configuration renders the same scene as examples/kerr_arcade.jl --
//! Kerr a = 0.9, volumetric gas, procedural stars -- so the numbers are
//! comparable with the Julia renderer's own HUD.

use std::time::Instant;

use crate::camera::{lens, Camera};
use crate::disc::{AccretionDisc, DiscVolume, VolumeSettings};
use crate::gr::Spacetime;
use crate::renderer::{Renderer, RendererDesc, StarSettings};

struct Cfg {
    w: usize,
    h: usize,
    dt: f32,
    order: i32,
    tol: f32,
}

fn build(vol: &DiscVolume, disc: AccretionDisc, st: Spacetime, sky: &[f32], c: &Cfg)
    -> Renderer
{
    let mut r = Renderer::new(&RendererDesc {
        width: c.w, height: c.h,
        background: (sky, 1, 1),
        spacetime: st,
        disc: Some(disc),
        volume: Some(vol),
        dt: c.dt,
        nmax: 1000,
        r_escape_factor: 2.0,
        order: c.order,
        tol: c.tol,
    });
    r.set_starfield(&StarSettings {
        density: 110.0, psf_pixels: 0.9, texture_weight: 0.0, ..Default::default()
    }, lens(24.0));
    r
}

/// Best of `reps` after a warm-up, in ms. Best-of rather than mean: we want
/// the cost of the work, not the cost of whatever else the machine was doing.
fn time_ms(r: &mut Renderer, cam: &Camera, st: &Spacetime, reps: usize) -> f64 {
    r.trace_blocking(cam, st);
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        r.trace_blocking(cam, st);
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    best
}

pub fn run() {
    let st = Spacetime::kerr(1.0, 0.9);
    let disc = AccretionDisc { inner_radius: 3.0, outer_radius: 20.0, density_falloff: 0.8 };
    println!("baking the gas volume...");
    let t0 = Instant::now();
    let vol = DiscVolume::bake(&disc, st.m, &VolumeSettings::default());
    println!("  {:.2}s\n", t0.elapsed().as_secs_f64());

    let sky = [0.0f32; 3];
    let cam = Camera::look_at([15.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], lens(24.0));

    println!("Kerr a=0.9, volumetric gas, procedural stars, RK4, dt=0.1");
    println!("{:<12} {:>10} {:>9} {:>12}", "resolution", "trace ms", "fps", "Mray/s");
    for &(w, h) in &[(256usize, 144usize), (640, 360), (1280, 720), (1920, 1080), (2560, 1440)] {
        let c = Cfg { w, h, dt: 0.1, order: 4, tol: 1e-4 };
        let mut r = build(&vol, disc, st, &sky, &c);
        let ms = time_ms(&mut r, &cam, &st, 20);
        println!("{:<12} {:>10.2} {:>9.0} {:>12.1}",
                 format!("{w}x{h}"), ms, 1000.0 / ms, (w * h) as f64 / (ms * 1000.0));
    }

    println!("\nIntegrator, at 1280x720 (same scene, dt=0.1)");
    println!("{:<28} {:>10} {:>9} {:>16}", "integrator", "trace ms", "fps", "vs RK4 (RMS)");

    // RK4 is the reference image: a faster integrator that changes the picture
    // has not made anything faster, it has made something else.
    let cref = Cfg { w: 1280, h: 720, dt: 0.1, order: 4, tol: 1e-4 };
    let mut rref = build(&vol, disc, st, &sky, &cref);
    let ref_img = rref.trace_blocking(&cam, &st);
    let ms_ref = time_ms(&mut rref, &cam, &st, 20);
    println!("{:<28} {:>10.2} {:>9.0} {:>16}", "RK4 (4 evals/step)", ms_ref, 1000.0 / ms_ref, "-");

    for (name, order, tol) in [
        ("midpoint (2 evals/step)", 2, 1e-4f32),
        ("adaptive RK45, tol 1e-3", 45, 1e-3),
        ("adaptive RK45, tol 1e-4", 45, 1e-4),
        ("adaptive RK45, tol 1e-5", 45, 1e-5),
        ("RK45 uncapped, tol 1e-3", 46, 1e-3),
        ("RK45 uncapped, tol 1e-5", 46, 1e-5),
        ("RK45 uncapped, tol 1e-7", 46, 1e-7),
    ] {
        let c = Cfg { w: 1280, h: 720, dt: 0.1, order, tol };
        let mut r = build(&vol, disc, st, &sky, &c);
        let img = r.trace_blocking(&cam, &st);
        let ms = time_ms(&mut r, &cam, &st, 20);
        let mut se = 0.0f64;
        for k in 0..img.len() {
            let d = (img[k] - ref_img[k]) as f64;
            se += d * d;
        }
        let rms = (se / img.len() as f64).sqrt();
        println!("{:<28} {:>10.2} {:>9.0} {:>16.5}", name, ms, 1000.0 / ms, rms);
    }

    // Sky only -- no disc, no gas -- to separate the geodesic integration
    // from everything that piggybacks on its step. If adaptive stepping is
    // exact here and wrong above, the geodesics were never the problem.
    println!("\nSky only (no disc, no gas), 1280x720, dt=0.1");
    println!("{:<28} {:>10} {:>9} {:>16}", "integrator", "trace ms", "fps", "vs RK4 (RMS)");
    let bare = |order: i32, tol: f32| RendererDesc {
        width: 1280, height: 720,
        background: (&sky[..], 1, 1),
        spacetime: st,
        disc: None,
        volume: None,
        dt: 0.1,
        nmax: 1000,
        r_escape_factor: 2.0,
        order,
        tol,
    };
    let mut b4 = Renderer::new(&bare(4, 1e-4));
    b4.set_starfield(&StarSettings {
        density: 110.0, psf_pixels: 0.9, texture_weight: 0.0, ..Default::default()
    }, lens(24.0));
    let bref = b4.trace_blocking(&cam, &st);
    let bms = time_ms(&mut b4, &cam, &st, 20);
    println!("{:<28} {:>10.2} {:>9.0} {:>16}", "RK4", bms, 1000.0 / bms, "-");
    for (name, order, tol) in [("adaptive RK45 (capped)", 45, 1e-4f32),
                               ("adaptive RK45 (uncapped)", 46, 1e-3),
                               ("adaptive RK45 (uncapped)", 46, 1e-5)] {
        let mut r = Renderer::new(&bare(order, tol));
        r.set_starfield(&StarSettings {
            density: 110.0, psf_pixels: 0.9, texture_weight: 0.0, ..Default::default()
        }, lens(24.0));
        let img = r.trace_blocking(&cam, &st);
        let ms = time_ms(&mut r, &cam, &st, 20);
        let mut se = 0.0f64;
        for k in 0..img.len() {
            let d = (img[k] - bref[k]) as f64;
            se += d * d;
        }
        println!("{:<28} {:>10.2} {:>9.0} {:>16.5}",
                 format!("{name} {tol:.0e}"), ms, 1000.0 / ms,
                 (se / img.len() as f64).sqrt());
    }

    println!("\nStep size, at 1280x720 (RK4)");
    println!("{:<12} {:>10} {:>9} {:>16}", "dt", "trace ms", "fps", "vs dt=0.05 (RMS)");
    let cfine = Cfg { w: 1280, h: 720, dt: 0.05, order: 4, tol: 1e-4 };
    let mut rfine = build(&vol, disc, st, &sky, &cfine);
    let fine = rfine.trace_blocking(&cam, &st);
    for dt in [0.05f32, 0.1, 0.2, 0.4] {
        let c = Cfg { w: 1280, h: 720, dt, order: 4, tol: 1e-4 };
        let mut r = build(&vol, disc, st, &sky, &c);
        let img = r.trace_blocking(&cam, &st);
        let ms = time_ms(&mut r, &cam, &st, 20);
        let mut se = 0.0f64;
        for k in 0..img.len() {
            let d = (img[k] - fine[k]) as f64;
            se += d * d;
        }
        println!("{:<12} {:>10.2} {:>9.0} {:>16.5}", dt, ms, 1000.0 / ms,
                 (se / img.len() as f64).sqrt());
    }
}
