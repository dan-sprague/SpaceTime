//! Real-time general-relativistic black-hole explorer on Metal.
//!
//! A spinning accretion disc traced at a low internal resolution and upscaled,
//! with an Escape menu for the scene (Schwarzschild / Kerr), colour treatment,
//! resolution, upscaling, and lens. The geodesics are exact Kerr in Cartesian
//! Kerr-Schild -- the same equations the Julia renderer integrates -- and the
//! camera rides the exact Kerr observer tetrad (static blending to ZAMO).

mod bench;
mod camera;
mod disc;
mod font;
mod gr;
mod menu;
mod metalfx;
mod minimap;
mod palette;
mod renderer;
mod selftest;
mod trajectory;

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::MetalLayer;
use objc::{msg_send, sel, sel_impl, runtime::YES};
use std::time::Instant;
use winit::{
    dpi::LogicalSize,
    event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::WindowBuilder,
};

use camera::{lens, Camera};
use disc::{AccretionDisc, DiscVolume, VolumeSettings};
use gr::Spacetime;
use menu::{Color, Menu, Scene, Upscale, GAS_LEVELS, LENSES, RES_LEVELS};
use palette::arcade_triad;
use renderer::{Overlay, Renderer, RendererDesc, StarSettings};

const WIN_W: u32 = 1280;
const WIN_H: u32 = 720;

/// Disc rotation rate, in M-time per wall-clock second. Slower than the film's
/// ~7.5 because the film only ran a bounded sweep; a viewer that holds on one
/// frame would watch an unbounded rate wind the frozen texture into streaks.
/// The sampler already caps the differential (see DIFF in trace.metal); this
/// sets the overall pace.
const DISC_RATE: f64 = 3.0;

/// Kerr scene spin. 0.9 resolves cleanly at the arcade step; near-extremal
/// would need a finer dt than real time allows.
const KERR_A: f64 = 0.9;

/// The Binary scene: hole 1 stays at the origin (mass 1), hole 2 sits here.
/// Both Schwarzschild; the disc and gas are off in this scene.
const BINARY_C2: [f64; 3] = [0.0, -20.0, 0.0];
const BINARY_M2: f64 = 1.0;

struct Input {
    keys: [bool; 256],
    dragging: bool,
    dyaw: f64,
    dpitch: f64,
}
impl Input {
    fn new() -> Self { Self { keys: [false; 256], dragging: false, dyaw: 0.0, dpitch: 0.0 } }
    fn down(&self, k: KeyCode) -> bool { self.keys[k as usize & 255] }
    fn set(&mut self, k: KeyCode, v: bool) { self.keys[k as usize & 255] = v; }
}

/// Internal render size for a resolution level: the chosen height at the
/// window's 16:9, width forced even.
fn render_dims(res_idx: usize) -> (usize, usize) {
    let h = RES_LEVELS[res_idx].1;
    let w = ((h * 16 / 9) + 1) & !1;
    (w, h)
}

/// The spacetime for a scene, given the primary hole's mass. Kerr keeps mass 1
/// (its spin scene is fixed); the Schwarzschild and Binary primaries take `m`,
/// so a track can fly a heavier hole (a heavy "booster" bends 45deg+ outside
/// its photon sphere, where the aim is smooth rather than a chaotic whirl).
fn scene_spacetime(scene: Scene, m: f64) -> Spacetime {
    match scene {
        Scene::Schwarzschild | Scene::Binary => Spacetime::schwarzschild(m),
        Scene::Kerr => Spacetime::kerr(1.0, KERR_A),
    }
}

/// Build a renderer for the menu's current scene and resolution. The gas
/// volume is baked once and shared, so this is cheap enough to call on a
/// scene or resolution change.
fn build_renderer(menu: &Menu, disc: AccretionDisc, volume: &DiscVolume,
                  sky: &[f32], bh2: ([f64; 3], f64), bh1_m: f64) -> Renderer {
    let (w, h) = render_dims(menu.res);
    // Gas fidelity: a 0.0 stride is the thin-disc sentinel -- pass volume: None
    // so the shader's VOL constant is off and the thin-plane crossing renders
    // instead of the volumetric march. Otherwise the value IS the march stride.
    let gas_stride = GAS_LEVELS[menu.gas].1;
    // The Binary scene is two Schwarzschild holes with the disc and gas off.
    let is_binary = matches!(menu.scene, Scene::Binary);
    Renderer::new(&RendererDesc {
        width: w, height: h,
        background: (sky, 1, 1),
        spacetime: scene_spacetime(menu.scene, bh1_m),
        disc: if is_binary { None } else { Some(disc) },
        volume: if !is_binary && gas_stride > 0.0 { Some(volume) } else { None },
        dt: 0.1,
        nmax: 1000,
        r_escape_factor: 2.0,
        order: 46,
        tol: 1e-7,
        gas_stride,
        binary: if is_binary { Some(bh2) } else { None },
    })
}

/// Apply the non-structural menu choices (colour, upscale, lens, stars) to an
/// existing renderer and camera. Returns the fisheye half-angle in degrees.
fn apply_soft(r: &mut Renderer, menu: &Menu, cam: &mut Camera) -> f64 {
    // Colour treatment. Every mode sets the whole grade explicitly, since the
    // fields persist between calls.
    match menu.color {
        // Physically faithful: neutral white balance, no contrast crush, no
        // vignette/grain, and a hue-preserving highlight roll-off (1.0) so the
        // hot inner disc stays blue-white and the cool/redshifted gas stays
        // orange instead of clipping to white. This is "what colour is the
        // light, really."
        Color::TrueColor => {
            r.set_palette(None); r.set_dither(0.0); r.set_filmic(true);
            r.set_hue_preserve(1.0); r.set_saturation(1.0); r.set_crush(1.0);
            r.set_white_balance(1.0, 1.0, 1.0); r.set_vignette(0.0); r.set_grain(0.0);
        }
        // A deliberate movie grade, clearly distinct from True Color: a warm
        // white balance, punchy saturation, filmic per-channel highlight bloom
        // (hue 0.3 -> highlights desaturate toward white like film stock),
        // added contrast, a vignette, and a whisper of grain.
        Color::Cinematic => {
            r.set_palette(None); r.set_dither(0.0); r.set_filmic(true);
            r.set_hue_preserve(0.3); r.set_saturation(1.35); r.set_crush(1.15);
            r.set_white_balance(1.07, 1.0, 0.9); r.set_vignette(0.4); r.set_grain(0.02);
        }
        // Retro: a hand-built pixel-art ramp -- three hues (indigo/magenta/
        // amber), three shades each -- collapsed by luminance with ordered
        // dither. Crisp pixels are forced below, since smoothing would erase
        // the pixel-art look this palette is going for.
        Color::Arcade => {
            r.set_palette(Some(&arcade_triad()));
            r.set_dither(1.0); r.set_filmic(false); r.set_hue_preserve(0.75);
            r.set_saturation(1.0); r.set_crush(1.0);
            r.set_white_balance(1.0, 1.0, 1.0); r.set_vignette(0.0); r.set_grain(0.0);
        }
    }
    r.set_upscale_smooth(menu.upscale == Upscale::Smooth);
    r.upscale_metalfx = menu.upscale == Upscale::MetalFX;
    // Pixel-art mode keeps hard pixels regardless of the upscale row: MetalFX
    // and bilinear both dissolve the chunky look the arcade palette wants.
    if menu.color == Color::Arcade {
        r.set_upscale_smooth(false);
        r.upscale_metalfx = false;
    }

    // Lens: focal length -> fov, or the fisheye projection.
    let (_, mm) = LENSES[menu.lens];
    let fisheye_deg = if mm == 0.0 { 110.0 } else { 0.0 };
    cam.fov_factor = if mm > 0.0 { lens(mm) } else { 1.0 };

    // Stars: chunky field for arcade, dense desaturated field otherwise, sized
    // to the render height. The PSF and saturation follow the film's recipe.
    let h = r.height as f32;
    let stars = if menu.color == Color::Arcade {
        StarSettings { density: 110.0, psf_pixels: 0.9, saturation: 1.0,
                       strength: 1.0, flux: 0.011, fill: 0.5, concentration: 3.0,
                       ..Default::default() }
    } else {
        StarSettings { density: (h * 0.8).clamp(120.0, 900.0), psf_pixels: 0.7,
                       saturation: 0.35, strength: 1.4, flux: 0.02, fill: 0.7,
                       concentration: 4.0, ..Default::default() }
    };
    r.set_starfield(&stars, cam.fov_factor);
    fisheye_deg
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--selftest") { selftest::run(); return; }
    if args.iter().any(|a| a == "--bench") { bench::run(); return; }
    if args.iter().any(|a| a == "--fontdump") { font::dump(); return; }
    if args.iter().any(|a| a == "--mapdump") {
        // Render one system-map frame to a PPM for offline inspection.
        let t = trajectory::Trajectory::load("whirl_path.txt").ok();
        let path = t.as_ref().map(|t| t.positions());
        let (pos, fwd) = t.as_ref().map(|t| { let (p, f, _) = t.at(13.0); (p, f) })
            .unwrap_or(([18.0, 0.0, 6.0], [-1.0, 0.0, 0.0]));
        let mut m = minimap::Minimap::new(300, 224);
        m.render(0.7, &[([0.0, 0.0, 0.0], 1.0)], (3.0, 20.0), path.as_deref(), pos, fwd, 40.0,
                 [0.0, 0.0, 0.0]);
        let mut out = format!("P6\n{} {}\n255\n", m.w, m.h).into_bytes();
        for i in (0..m.buf.len()).step_by(4) {
            out.push(m.buf[i + 2]); out.push(m.buf[i + 1]); out.push(m.buf[i]);
        }
        std::fs::write("map.ppm", out).unwrap();
        println!("wrote map.ppm"); return;
    }

    // Cinematic playback: `--play <track>` flies a precomputed geodesic camera
    // track (exported from the Julia solver). The track's own header picks the
    // scene -- two declared holes fly the Binary slingshot, one flies the
    // single-hole whirl -- so the lensing matches the path (see below).
    let track = args.iter().position(|a| a == "--play")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| match trajectory::Trajectory::load(p) {
            Ok(t) => { println!("track: {:.1}s, {} samples -> {}", t.duration, t.len(), p); Some(t) }
            Err(e) => { eprintln!("trajectory load failed: {e}"); None }
        });

    // Headless verification of a slingshot track: `--trackshot <file>` traces a
    // filmstrip of frames along the flight (camera riding the geodesic, both
    // holes lensing), writing trackshot_00..N.ppm. No window, so it can run in
    // CI or over ssh.
    if let Some(i) = args.iter().position(|a| a == "--trackshot") {
        let path = args.get(i + 1).map(|s| s.as_str()).unwrap_or("slingshot_path.txt");
        let tr = match trajectory::Trajectory::load(path) {
            Ok(t) => t, Err(e) => { eprintln!("track load failed: {e}"); return; }
        };
        let bh2 = *tr.holes.get(1).unwrap_or(&(BINARY_C2, BINARY_M2));
        let bh1_m = tr.holes.first().map(|h| h.1).unwrap_or(1.0);
        let sky = [0.0f32; 3];
        let (w, h) = (960usize, 540usize);
        let mut r = Renderer::new(&RendererDesc {
            width: w, height: h, background: (&sky, 1, 1),
            spacetime: Spacetime::schwarzschild(bh1_m),
            disc: None, volume: None, dt: 0.1, nmax: 1000, r_escape_factor: 2.0,
            order: 46, tol: 1e-7, gas_stride: 0.32,
            binary: Some(bh2),
        });
        r.set_starfield(&StarSettings {
            density: 520.0, psf_pixels: 0.7, saturation: 0.35, strength: 1.4,
            flux: 0.02, fill: 0.7, concentration: 4.0, ..Default::default()
        }, lens(24.0));
        let st = Spacetime::schwarzschild(bh1_m);
        // Map framing, identical to the live loop: centre on the system midpoint
        // and fit the whole flight so both holes stay in view.
        let map_holes: Vec<(gr::V3, f64)> = if tr.holes.is_empty() {
            vec![([0.0, 0.0, 0.0], 1.0)]
        } else { tr.holes.clone() };
        let nh = map_holes.len() as f64;
        let center = [map_holes.iter().map(|h| h.0[0]).sum::<f64>() / nh,
                      map_holes.iter().map(|h| h.0[1]).sum::<f64>() / nh,
                      map_holes.iter().map(|h| h.0[2]).sum::<f64>() / nh];
        let path = tr.positions();
        let fit = path.iter().chain(map_holes.iter().map(|h| &h.0))
            .map(|q| gr::norm3([q[0] - center[0], q[1] - center[1], q[2] - center[2]]))
            .fold(0.0f64, f64::max) * 1.12;
        let mut mm = minimap::Minimap::new(320, 200);
        let frames = 6usize;
        for k in 0..frames {
            let t = tr.duration * k as f64 / (frames - 1) as f64;
            let (pos, fwd, vel) = tr.at(t);
            let mut cam = Camera::look_at(pos, [pos[0] + fwd[0], pos[1] + fwd[1],
                                               pos[2] + fwd[2]], tr.up, lens(24.0));
            cam.velocity = vel;
            let img = r.trace_blocking(&cam, &st);
            // Tonemap into an 8-bit RGB buffer.
            let mut rgb = vec![0u8; w * h * 3];
            for i in 0..w * h {
                for c in 0..3 {
                    let v = img[i * 3 + c].max(0.0);
                    rgb[i * 3 + c] =
                        ((v / (1.0 + v)).powf(1.0 / 2.2) * 255.0).clamp(0.0, 255.0) as u8;
                }
            }
            // Composite the system map into the top-right corner (BGRA -> RGB).
            mm.render(0.7, &map_holes, (0.0, 0.0), Some(&path), pos, fwd, fit, center);
            let (mw, mh, margin) = (mm.w, mm.h, 12usize);
            for my in 0..mh {
                for mx in 0..mw {
                    let dx = w - mw - margin + mx;
                    let dy = margin + my;
                    let s = (my * mw + mx) * 4;
                    let d = (dy * w + dx) * 3;
                    rgb[d] = mm.buf[s + 2]; rgb[d + 1] = mm.buf[s + 1]; rgb[d + 2] = mm.buf[s];
                }
            }
            let mut out = format!("P6\n{} {}\n255\n", w, h).into_bytes();
            out.extend_from_slice(&rgb);
            let name = format!("trackshot_{k}.ppm");
            std::fs::write(&name, out).unwrap();
            println!("wrote {name}  t={t:.1}s  pos=({:.1},{:.1},{:.1})  |v|={:.3}c",
                     pos[0], pos[1], pos[2], gr::norm3(vel));
        }
        return;
    }

    if args.iter().any(|a| a == "--binaryshot") {
        let sky = [0.0f32; 3];
        let (w, h) = (960usize, 540usize);
        let mut r = Renderer::new(&RendererDesc {
            width: w, height: h, background: (&sky, 1, 1),
            spacetime: Spacetime::schwarzschild(1.0),
            disc: None, volume: None, dt: 0.1, nmax: 1000, r_escape_factor: 2.0,
            order: 46, tol: 1e-7, gas_stride: 0.32,
            binary: Some((BINARY_C2, BINARY_M2)),
        });
        r.set_starfield(&StarSettings {
            density: 520.0, psf_pixels: 0.7, saturation: 0.35, strength: 1.4,
            flux: 0.02, fill: 0.7, concentration: 4.0, ..Default::default()
        }, lens(24.0));
        let mid = [0.5 * BINARY_C2[0], 0.5 * BINARY_C2[1], 0.5 * BINARY_C2[2]];
        let cam = Camera::look_at([mid[0] + 52.0, mid[1], mid[2] + 14.0], mid,
                                  [0.0, 0.0, 1.0], lens(24.0));
        let img = r.trace_blocking(&cam, &Spacetime::schwarzschild(1.0));
        let mut out = format!("P6\n{} {}\n255\n", w, h).into_bytes();
        for i in 0..w * h {
            for c in 0..3 {
                let v = img[i * 3 + c].max(0.0);
                out.push(((v / (1.0 + v)).powf(1.0 / 2.2) * 255.0).clamp(0.0, 255.0) as u8);
            }
        }
        std::fs::write("binaryshot.ppm", out).unwrap();
        println!("wrote binaryshot.ppm"); return;
    }

    let disc = AccretionDisc { inner_radius: 3.0, outer_radius: 20.0, density_falloff: 0.8 };
    println!("baking the gas volume...");
    let t0 = Instant::now();
    let volume = DiscVolume::bake(&disc, 1.0, &VolumeSettings::default());
    println!("  {} cells in {:.2}s", volume.density.len(), t0.elapsed().as_secs_f64());

    let sky = [0.0f32; 3];

    // Where hole 2 sits and how heavy it is. The default is the menu Binary
    // scene's layout; a slingshot track overrides it with the geometry the
    // geodesic was actually solved in (hole 2 out in the gap), so the render
    // and the map agree with the flight.
    let (mut binary_c2, mut binary_m2) = (BINARY_C2, BINARY_M2);
    // Primary (origin) hole mass. A track can make it heavier than the default 1.
    let mut bh1_m = 1.0f64;

    let mut menu = Menu::default();
    // A track picks the scene: two declared holes -> the Binary slingshot, one
    // (or none) -> the single-hole whirl. The track's own hole layout drives
    // the renderer so the lensing matches the path -- including the primary
    // hole's mass, so a heavy-booster level renders the right shadow size.
    if let Some(tr) = &track {
        if let Some(&(_, m1)) = tr.holes.first() { bh1_m = m1; }
        if tr.holes.len() >= 2 {
            binary_c2 = tr.holes[1].0;
            binary_m2 = tr.holes[1].1;
            menu.scene = Scene::Binary;
        } else {
            menu.scene = Scene::Schwarzschild;
        }
    }
    let mut spacetime = scene_spacetime(menu.scene, bh1_m);

    let mut cam = Camera::look_at([15.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                                  lens(24.0));
    let mut yaw = cam.fwd[1].atan2(cam.fwd[0]);
    let mut pitch = cam.fwd[2].asin();
    let mut roll = 0.0f64;

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Spacetime")
        .with_inner_size(LogicalSize::new(WIN_W, WIN_H))
        .build(&event_loop)
        .unwrap();

    let layer = MetalLayer::new();
    let mut renderer = build_renderer(&menu, disc, &volume, &sky, (binary_c2, binary_m2), bh1_m);
    let mut fisheye = apply_soft(&mut renderer, &menu, &mut cam);
    let mut built_scene = menu.scene;
    let mut built_res = menu.res;
    let mut built_gas = menu.gas;
    let mut last_rev = menu.revision;

    layer.set_device(&renderer.device);
    layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
    layer.set_framebuffer_only(false);
    layer.set_display_sync_enabled(false);
    let size = window.inner_size();
    layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

    unsafe {
        let handle = window.window_handle().unwrap();
        let RawWindowHandle::AppKit(h) = handle.as_raw() else { panic!("not an AppKit window"); };
        let view = h.ns_view.as_ptr() as cocoa_id;
        view.setWantsLayer(YES);
        let _: () = msg_send![view, setLayer: layer.as_ref()];
    }

    let mut input = Input::new();
    let mut speed = 2.0f64;
    let mut exposure = 1.0f32;
    let mut relativistic = track.is_some();   // whirl looks best boosted; toggle with R
    let mut play_t = 0.0f64;                   // playback clock along the track
    let mut paused = false;
    // System map: a slowly-orbiting 3-D overview in the corner.
    let mut map = minimap::Minimap::new(300, 224);
    let mut map_az = 0.7f64;
    let track_path: Option<Vec<gr::V3>> = track.as_ref().map(|t| t.positions());
    let mut disc_t = 0.0f64;
    let mut last = Instant::now();
    let mut frames = 0u32;
    let mut fps_clock = Instant::now();

    event_loop.run(move |event, elwt| {
        elwt.set_control_flow(ControlFlow::Poll);
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(s) => {
                    layer.set_drawable_size(CGSize::new(s.width as f64, s.height as f64));
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        let pressed = event.state == ElementState::Pressed;
                        if pressed && !event.repeat {
                            match code {
                                KeyCode::Escape => menu.toggle(),
                                _ if menu.open => match code {
                                    KeyCode::ArrowUp => menu.move_cursor(-1),
                                    KeyCode::ArrowDown => menu.move_cursor(1),
                                    KeyCode::ArrowLeft => { menu.adjust(-1); }
                                    KeyCode::ArrowRight | KeyCode::Enter => { menu.adjust(1); }
                                    _ => {}
                                },
                                KeyCode::KeyR => relativistic = !relativistic,
                                KeyCode::KeyP if track.is_some() => paused = !paused,
                                _ => {}
                            }
                        }
                        input.set(code, pressed);
                    }
                }
                WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                    input.dragging = state == ElementState::Pressed;
                }
                _ => {}
            },
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                if input.dragging && !menu.open {
                    input.dyaw -= delta.0 * 0.004;
                    input.dpitch -= delta.1 * 0.004;
                }
            }
            Event::AboutToWait => {
                let now = Instant::now();
                let dt = (now - last).as_secs_f64().min(0.1);
                last = now;

                // Re-apply menu choices when anything changed, rebuilding the
                // renderer only for the structural ones (scene, resolution).
                if menu.revision != last_rev {
                    last_rev = menu.revision;
                    if menu.scene != built_scene || menu.res != built_res
                        || menu.gas != built_gas
                    {
                        let to_binary = matches!(menu.scene, Scene::Binary)
                            && built_scene != menu.scene;
                        renderer = build_renderer(&menu, disc, &volume, &sky,
                                                  (binary_c2, binary_m2), bh1_m);
                        layer.set_device(&renderer.device);
                        spacetime = scene_spacetime(menu.scene, bh1_m);
                        built_scene = menu.scene;
                        built_res = menu.res;
                        built_gas = menu.gas;
                        renderer.set_exposure(exposure);
                        // Frame both holes when entering the Binary scene.
                        if to_binary && track.is_none() {
                            let mid = [0.5 * binary_c2[0], 0.5 * binary_c2[1], 0.5 * binary_c2[2]];
                            cam = Camera::look_at([mid[0] + 52.0, mid[1], mid[2] + 14.0],
                                                  mid, [0.0, 0.0, 1.0], cam.fov_factor);
                            yaw = cam.fwd[1].atan2(cam.fwd[0]);
                            pitch = cam.fwd[2].asin();
                            roll = 0.0;
                        }
                    }
                    fisheye = apply_soft(&mut renderer, &menu, &mut cam);
                }

                // Cinematic track playback takes over the camera when a track
                // is loaded; the geodesic's own velocity drives the relativistic
                // tetrad, so the whirl aberration is physically correct.
                if let (Some(tr), false) = (&track, menu.open) {
                    if !paused {
                        play_t += dt;
                        if play_t > tr.duration { play_t = 0.0; }   // loop
                    }
                    let (pos, fwd, vel) = tr.at(play_t);
                    cam.pos = pos;
                    cam.fwd = fwd;
                    cam.right = gr::normalize3(gr::cross3(fwd, tr.up));
                    cam.up_local = gr::cross3(cam.right, fwd);
                    cam.velocity = vel;
                    if input.down(KeyCode::KeyT) {
                        exposure = (exposure * 1.04).min(20.0); renderer.set_exposure(exposure);
                    }
                    if input.down(KeyCode::KeyG) {
                        exposure = (exposure / 1.04).max(0.05); renderer.set_exposure(exposure);
                    }
                    input.dyaw = 0.0; input.dpitch = 0.0;
                }
                // Free flight, disabled while the menu is up or a track is playing.
                else if !menu.open {
                    yaw += input.dyaw;
                    pitch = (pitch + input.dpitch).clamp(-1.5533, 1.5533);
                    roll += ((input.down(KeyCode::KeyC) as i32
                              - input.down(KeyCode::KeyZ) as i32) as f64) * 1.5 * dt;
                    cam.set_orientation(yaw, pitch, roll);

                    if input.down(KeyCode::BracketLeft) { speed = (speed / 1.03).max(0.1); }
                    if input.down(KeyCode::BracketRight) { speed = (speed * 1.03).min(50.0); }
                    if input.down(KeyCode::KeyT) {
                        exposure = (exposure * 1.04).min(20.0); renderer.set_exposure(exposure);
                    }
                    if input.down(KeyCode::KeyG) {
                        exposure = (exposure / 1.04).max(0.05); renderer.set_exposure(exposure);
                    }

                    let v = speed * dt * if input.down(KeyCode::ShiftLeft) { 5.0 } else { 1.0 };
                    let wz = [0.0, 0.0, 1.0];
                    let mut step = |d: gr::V3, s: f64| { cam.pos = gr::add3(cam.pos, gr::scale3(d, s)); };
                    if input.down(KeyCode::KeyW) { step(cam.fwd, v); }
                    if input.down(KeyCode::KeyS) { step(cam.fwd, -v); }
                    if input.down(KeyCode::KeyD) { step(cam.right, v); }
                    if input.down(KeyCode::KeyA) { step(cam.right, -v); }
                    if input.down(KeyCode::KeyE) { step(wz, v); }
                    if input.down(KeyCode::KeyQ) { step(wz, -v); }

                    let r = gr::norm3(cam.pos);
                    if r < 0.35 { cam.pos = gr::scale3(gr::normalize3(cam.pos), 0.35); }
                }
                input.dyaw = 0.0;
                input.dpitch = 0.0;

                // Spin the disc in real time (paused while the menu is open, so
                // the frame the user is grading holds still).
                if !menu.open { disc_t += dt * DISC_RATE; }
                renderer.set_disc_time(disc_t as f32);

                // System map: orbit slowly, keep the camera in frame.
                if !menu.open { map_az += dt * 0.15; }
                let is_binary = matches!(menu.scene, Scene::Binary);
                let holes: Vec<(gr::V3, f64)> = if is_binary {
                    vec![([0.0, 0.0, 0.0], bh1_m), (binary_c2, binary_m2)]
                } else {
                    vec![([0.0, 0.0, 0.0], bh1_m)]
                };
                let map_disc = if is_binary { (0.0, 0.0) }
                               else { (disc.inner_radius, disc.outer_radius) };
                // Centre the map on the system's midpoint (origin for a single
                // hole), and for a slingshot track frame the whole flight so
                // both holes and the path stay in view the entire way across.
                let n = holes.len() as f64;
                let center = [holes.iter().map(|h| h.0[0]).sum::<f64>() / n,
                              holes.iter().map(|h| h.0[1]).sum::<f64>() / n,
                              holes.iter().map(|h| h.0[2]).sum::<f64>() / n];
                let fit = if let Some(p) = track_path.as_deref() {
                    p.iter().chain(holes.iter().map(|h| &h.0))
                        .map(|q| gr::norm3([q[0] - center[0], q[1] - center[1], q[2] - center[2]]))
                        .fold(0.0f64, f64::max) * 1.12
                } else {
                    gr::norm3(cam.pos).max(38.0) * 1.06
                };
                map.render(map_az, &holes, map_disc,
                           track_path.as_deref(), cam.pos, cam.fwd, fit, center);
                renderer.set_minimap(map.w as u32, map.h as u32, &map.buf);

                if let Some(drawable) = layer.next_drawable() {
                    let (cells, attr, dims) = if menu.open {
                        let g = menu.render_grid();
                        (g.cells.iter().map(|&c| c as i32).collect::<Vec<_>>(),
                         g.attr.iter().map(|&a| a as i32).collect::<Vec<_>>(),
                         Some((g.cols, g.rows)))
                    } else {
                        (Vec::new(), Vec::new(), None)
                    };
                    let ov = dims.map(|(cols, rows)| Overlay { cols, rows, cells: &cells, attr: &attr });
                    renderer.render_and_present(&cam, &spacetime, drawable, fisheye, relativistic, ov);
                }

                frames += 1;
                if fps_clock.elapsed().as_secs_f64() >= 1.0 {
                    let secs = fps_clock.elapsed().as_secs_f64();
                    println!("r {:.2}M  {:.1} fps  ({:.2} ms)",
                             gr::norm3(cam.pos), frames as f64 / secs,
                             1000.0 * secs / frames as f64);
                    frames = 0;
                    fps_clock = Instant::now();
                }
            }
            _ => {}
        }
    }).unwrap();
}
