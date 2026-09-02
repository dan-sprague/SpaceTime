//! Real-time general-relativistic flight, arcade mode.
//!
//! The Rust/Metal twin of `examples/kerr_arcade.jl`: a spinning black hole
//! traced at a low internal resolution, blown up nearest-neighbour, and
//! collapsed to a handful of tones. The pixels are not hiding sloppy physics
//! -- the geodesics are exact Kerr in Cartesian Kerr-Schild, the same
//! equations the Julia renderer integrates.

mod bench;
mod camera;
mod disc;
mod gr;
mod palette;
mod renderer;
mod selftest;

use cocoa::{appkit::NSView, base::id as cocoa_id};
use core_graphics_types::geometry::CGSize;
use metal::MetalLayer;
use objc::{msg_send, sel, sel_impl, runtime::YES};
use std::time::Instant;
use winit::{
    dpi::LogicalSize,
    event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasWindowHandle, RawWindowHandle},
    window::WindowBuilder,
};

use camera::{lens, Camera};
use disc::{AccretionDisc, DiscVolume, VolumeSettings};
use gr::Spacetime;
use palette::{arcade_palette, Ramp};
use renderer::{Renderer, RendererDesc, StarSettings};

/// Internal render resolution. Pick one that integer-divides the window or
/// the upscale gives uneven pixels: 256x144 x10 and 320x180 x8 both land on
/// 2560x1440.
const RENDER_W: usize = 256;
const RENDER_H: usize = 144;
const WIN_W: u32 = 1280;
const WIN_H: u32 = 720;

struct Input {
    keys: [bool; 256],
    dragging: bool,
    last_mouse: Option<(f64, f64)>,
    dyaw: f64,
    dpitch: f64,
}

impl Input {
    fn new() -> Self {
        Self { keys: [false; 256], dragging: false, last_mouse: None, dyaw: 0.0, dpitch: 0.0 }
    }
    fn down(&self, k: KeyCode) -> bool { self.keys[k as usize & 255] }
    fn set(&mut self, k: KeyCode, v: bool) { self.keys[k as usize & 255] = v; }
}

fn main() {
    if std::env::args().any(|a| a == "--selftest") {
        selftest::run();
        return;
    }
    if std::env::args().any(|a| a == "--bench") {
        bench::run();
        return;
    }

    // a = 0.9: strong frame dragging -- the shadow sits visibly off-centre and
    // is flattened on the prograde side -- while still resolving cleanly at
    // dt = 0.1. Near-extremal (a > ~0.95) needs a finer step than arcade mode
    // uses: the capture radius and the horizon converge and the shadow starts
    // to leak sky.
    let spacetime = Spacetime::kerr(1.0, 0.9);

    let disc = AccretionDisc {
        inner_radius: 3.0,
        outer_radius: 20.0,
        density_falloff: 0.8,
    };

    println!("baking the gas volume...");
    let t0 = Instant::now();
    let volume = DiscVolume::bake(&disc, spacetime.m, &VolumeSettings::default());
    println!("  {} cells in {:.2}s", volume.density.len(), t0.elapsed().as_secs_f64());

    // No sky texture: arcade mode runs on the procedural starfield alone.
    // The starmap's broad nebulosity is a low-level glow, which under a
    // luminance palette lifts the whole sky off black.
    let sky = [0.0f32; 3];

    let mut cam = Camera::look_at([15.0, 0.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
                                  lens(24.0));
    let mut yaw = (cam.fwd[1]).atan2(cam.fwd[0]);
    let mut pitch = cam.fwd[2].asin();
    let mut roll = 0.0f64;

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Spacetime - Kerr arcade")
        .with_inner_size(LogicalSize::new(WIN_W, WIN_H))
        .build(&event_loop)
        .unwrap();

    let layer = MetalLayer::new();
    let mut renderer = Renderer::new(&RendererDesc {
        width: RENDER_W,
        height: RENDER_H,
        background: (&sky, 1, 1),
        spacetime,
        disc: Some(disc),
        volume: Some(&volume),
        dt: 0.1,
        nmax: 1000,
        r_escape_factor: 2.0,
        order: 4,
        tol: 1e-4,
    });

    layer.set_device(&renderer.device);
    layer.set_pixel_format(metal::MTLPixelFormat::BGRA8Unorm);
    // The pack kernel writes the drawable directly, which needs shaderWrite
    // usage; framebufferOnly would deny it.
    layer.set_framebuffer_only(false);
    // CAMetalLayer syncs to the display by default, so an 11 ms frame still
    // presents on a refresh boundary -- that reads as exactly 60 fps and every
    // renderer saving vanishes. Off runs at whatever the GPU can actually do.
    layer.set_display_sync_enabled(false);
    let size = window.inner_size();
    layer.set_drawable_size(CGSize::new(size.width as f64, size.height as f64));

    unsafe {
        let handle = window.window_handle().unwrap();
        let RawWindowHandle::AppKit(h) = handle.as_raw() else {
            panic!("not an AppKit window");
        };
        let view = h.ns_view.as_ptr() as cocoa_id;
        view.setWantsLayer(YES);
        let _: () = msg_send![view, setLayer: layer.as_ref()];
    }

    // Black plus five magma steps. The mapping is on LUMINANCE, so the frame
    // collapses to exactly this many tones and empty sky lands on entry 0,
    // which is true black. `lo`/`hi` trim the muddiest low end and the
    // blown-out top, which is where these ramps are hardest to look at.
    renderer.set_palette(Some(&arcade_palette(6, Ramp::Magma, true, 0.12, 0.92)));
    renderer.set_dither(1.0);   // 4x4 Bayer, one palette step

    // A 3x3 cell scan only holds while a cell covers 5 sigma: at 144 rows the
    // 1440p default of 384 would truncate every star into a clipped square.
    // Fewer, bigger stars is also just what 256x144 wants.
    renderer.set_starfield(&StarSettings {
        density: 110.0,
        psf_pixels: 0.9,
        texture_weight: 0.0,
        ..Default::default()
    }, cam.fov_factor);

    let mut input = Input::new();
    let mut speed = 2.0f64;
    let mut exposure = 1.0f32;
    let mut fisheye = 0.0f64;
    let mut relativistic = false;
    let mut last = Instant::now();
    let mut frames = 0u32;
    let mut fps_clock = Instant::now();

    event_loop.run(move |event, elwt| {
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
                                KeyCode::Escape => elwt.exit(),
                                KeyCode::KeyR => relativistic = !relativistic,
                                KeyCode::KeyL => fisheye = if fisheye > 0.0 { 0.0 } else { 100.0 },
                                _ => {}
                            }
                        }
                        input.set(code, pressed);
                    }
                }
                WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                    input.dragging = state == ElementState::Pressed;
                    if !input.dragging { input.last_mouse = None; }
                }
                WindowEvent::RedrawRequested => {}
                _ => {}
            },
            Event::DeviceEvent { event: DeviceEvent::MouseMotion { delta }, .. } => {
                if input.dragging {
                    input.dyaw -= delta.0 * 0.004;
                    input.dpitch -= delta.1 * 0.004;
                }
            }
            Event::AboutToWait => {
                let now = Instant::now();
                let dt = (now - last).as_secs_f64().min(0.1);
                last = now;

                yaw += input.dyaw;
                pitch = (pitch + input.dpitch).clamp(-1.5533, 1.5533);
                input.dyaw = 0.0;
                input.dpitch = 0.0;
                roll += ((input.down(KeyCode::KeyC) as i32 - input.down(KeyCode::KeyZ) as i32)
                         as f64) * 1.5 * dt;
                cam.set_orientation(yaw, pitch, roll);

                if input.down(KeyCode::BracketLeft) { speed = (speed / 1.03).max(0.1); }
                if input.down(KeyCode::BracketRight) { speed = (speed * 1.03).min(50.0); }
                if input.down(KeyCode::KeyT) {
                    exposure = (exposure * 1.04).min(20.0);
                    renderer.set_exposure(exposure);
                }
                if input.down(KeyCode::KeyG) {
                    exposure = (exposure / 1.04).max(0.05);
                    renderer.set_exposure(exposure);
                }

                let v = speed * dt * if input.down(KeyCode::ShiftLeft) { 5.0 } else { 1.0 };
                let world_z = [0.0, 0.0, 1.0];
                let mut step = |d: gr::V3, s: f64| {
                    cam.pos = gr::add3(cam.pos, gr::scale3(d, s));
                };
                if input.down(KeyCode::KeyW) { step(cam.fwd, v); }
                if input.down(KeyCode::KeyS) { step(cam.fwd, -v); }
                if input.down(KeyCode::KeyD) { step(cam.right, v); }
                if input.down(KeyCode::KeyA) { step(cam.right, -v); }
                if input.down(KeyCode::KeyE) { step(world_z, v); }
                if input.down(KeyCode::KeyQ) { step(world_z, -v); }

                // Never let the camera reach the singularity: the tetrad is
                // regular through the horizon but not at r = 0.
                let r = gr::norm3(cam.pos);
                if r < 0.35 * spacetime.m {
                    cam.pos = gr::scale3(gr::normalize3(cam.pos), 0.35 * spacetime.m);
                }

                if let Some(drawable) = layer.next_drawable() {
                    renderer.render(&cam, &spacetime, drawable.texture(),
                                    fisheye, relativistic);
                    renderer.commit_present(drawable);
                }

                frames += 1;
                if fps_clock.elapsed().as_secs_f64() >= 1.0 {
                    let secs = fps_clock.elapsed().as_secs_f64();
                    println!("r {:.2}M  {:.1} fps  ({:.2} ms)",
                             gr::norm3(cam.pos) / spacetime.m,
                             frames as f64 / secs,
                             1000.0 * secs / frames as f64);
                    frames = 0;
                    fps_clock = Instant::now();
                }
            }
            _ => {}
        }
    }).unwrap();
}
