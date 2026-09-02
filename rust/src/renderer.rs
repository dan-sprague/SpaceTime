//! Metal plumbing: pipelines, buffers, and the two dispatches that make a
//! frame. Nothing here touches the CPU copy of a pixel -- the trace writes an
//! HDR buffer, the pack kernel grades it straight into the drawable.

use metal::*;
use std::mem;

use crate::camera::{Camera, CAM_PARAMS_N};
use crate::disc::{AccretionDisc, Blackbody, DiscVolume, STAR_WB_TEMPERATURE};
use crate::gr::Spacetime;
use crate::palette::PALETTE_MAX;

include!(concat!(env!("OUT_DIR"), "/shaders.rs"));

/// Load the shader library, preferring the precompiled .metallib when the
/// build had `xcrun metal` available and falling back to the Metal
/// framework's runtime compiler when it did not.
fn load_library(device: &Device) -> Library {
    match SHADER_LIB {
        Some(bytes) => device
            .new_library_with_data(bytes)
            .expect("failed to load shaders.metallib"),
        None => {
            let opts = CompileOptions::new();
            opts.set_fast_math_enabled(true);
            device
                .new_library_with_source(SHADER_SRC, &opts)
                .expect("failed to compile shaders at runtime")
        }
    }
}

/// Shared-storage buffer from a slice. Apple silicon has unified memory, so a
/// "shared" buffer is one allocation both sides read -- there is no staging
/// copy to elide because there was never a transfer.
fn shared_buffer<T: Copy>(device: &Device, data: &[T]) -> Buffer {
    device.new_buffer_with_data(
        data.as_ptr() as *const _,
        mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn empty_buffer(device: &Device, bytes: u64) -> Buffer {
    device.new_buffer(bytes.max(4), MTLResourceOptions::StorageModeShared)
}

/// Host view of a shared buffer, so per-frame parameter updates are plain
/// writes rather than a copy the GPU could already see.
unsafe fn host_slice<'a>(b: &'a Buffer, n: usize) -> &'a mut [f32] {
    std::slice::from_raw_parts_mut(b.contents() as *mut f32, n)
}

/// Step-size coefficient and cap: `h = dt * clamp(coef * r / M, 1, cap)`.
/// Curvature goes as M/r^3, so scaling with r holds the per-step bending error
/// roughly uniform while collapsing the nearly-flat travel legs.
const HSTEP_COEF: f32 = 0.16;
const HSTEP_CAP: f32 = 8.0;

pub struct StarSettings {
    pub strength: f32,
    pub texture_weight: f32,
    pub density: f32,
    pub fill: f32,
    pub flux: f32,
    pub psf_pixels: f32,
    pub galactic: [f32; 3],
    pub concentration: f32,
    pub temp_min: f32,
    pub temp_max: f32,
    pub seed: i32,
}

impl Default for StarSettings {
    fn default() -> Self {
        Self {
            strength: 1.0, texture_weight: 0.0, density: 384.0, fill: 0.5,
            flux: 0.011, psf_pixels: 0.5, galactic: [0.0, 0.0, 1.0],
            concentration: 3.0, temp_min: 3000.0, temp_max: 16000.0, seed: 12345,
        }
    }
}

pub struct Renderer {
    pub device: Device,
    queue: CommandQueue,
    trace_pipeline: ComputePipelineState,
    pack_pipeline: ComputePipelineState,

    out: Buffer,
    bg: Buffer,
    bb_lut: Buffer,
    star_lut: Buffer,
    vol: Buffer,
    vol_params: Buffer,
    star_params: Buffer,
    cam_params: Buffer,
    st_params: Buffer,
    disc_params: Buffer,
    grade: Buffer,
    palette: Buffer,

    pub width: usize,
    pub height: usize,
    bg_w: usize,
    bg_h: usize,
    pub dt: f32,
    pub tol: f32,
    pub nmax_floor: u32,
    pub r_escape_factor: f32,
    grade_host: [f32; 20],
}

pub struct RendererDesc<'a> {
    pub width: usize,
    pub height: usize,
    /// Equirectangular sky as a flat (3, W, H) channel-fastest array.
    pub background: (&'a [f32], usize, usize),
    pub spacetime: Spacetime,
    pub disc: Option<AccretionDisc>,
    pub volume: Option<&'a DiscVolume>,
    pub dt: f32,
    pub nmax: u32,
    pub r_escape_factor: f32,
    /// 4 = classical RK4, 2 = explicit midpoint, 45 = adaptive Cash-Karp.
    pub order: i32,
    /// RK45 error tolerance; ignored by the fixed-order paths.
    pub tol: f32,
}

impl Renderer {
    pub fn new(desc: &RendererDesc) -> Self {
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        let library = load_library(&device);

        // Compile-time specialisation, the MSL analogue of the Julia kernel's
        // `Val{...}` dispatch: the Schwarzschild pipeline never pays Kerr's
        // register budget, and the no-gas pipeline stays lean.
        let consts = FunctionConstantValues::new();
        let kerr = desc.spacetime.is_kerr();
        let vol_on = desc.volume.is_some();
        consts.set_constant_value_at_index(
            &kerr as *const bool as *const _, MTLDataType::Bool, 0);
        consts.set_constant_value_at_index(
            &vol_on as *const bool as *const _, MTLDataType::Bool, 1);
        consts.set_constant_value_at_index(
            &desc.order as *const i32 as *const _, MTLDataType::Int, 2);

        let trace_fn = library
            .get_function("trace", Some(consts))
            .expect("missing kernel `trace`");
        let trace_pipeline = device
            .new_compute_pipeline_state_with_function(&trace_fn)
            .expect("failed to build trace pipeline");
        let pack_fn = library.get_function("pack", None).expect("missing kernel `pack`");
        let pack_pipeline = device
            .new_compute_pipeline_state_with_function(&pack_fn)
            .expect("failed to build pack pipeline");

        let (w, h) = (desc.width, desc.height);
        let (bg_data, bg_w, bg_h) = desc.background;

        // The disc's blackbody LUT and the STAR LUT are deliberately separate.
        // Sharing one made star colour a function of the disc's white balance,
        // so regrading the disc re-tinted the whole sky.
        let bb = match &desc.disc {
            Some(_) => Blackbody::new(5000.0, 1024),
            None => Blackbody::new(6500.0, 1024),
        };
        let stars = Blackbody::new(STAR_WB_TEMPERATURE, 1024);

        let disc_host: [f32; 6] = match &desc.disc {
            Some(d) => [d.inner_radius as f32, d.outer_radius as f32,
                        d.density_falloff as f32, bb.table_min as f32,
                        bb.table_max as f32, bb.table_size as f32],
            None => [0.0; 6],
        };

        let (vol_buf, vol_host) = match desc.volume {
            Some(v) => (
                shared_buffer(&device, &v.density),
                [v.log_s_in, v.log_s_out, v.z_max, v.nr as f32, v.nphi as f32,
                 v.nz as f32, v.emission_scale, v.opacity_scale, 2.0],
            ),
            None => (empty_buffer(&device, 4), [0.0f32; 9]),
        };

        // Grade defaults: the film look at a neutral exposure.
        let mut grade_host = [0.0f32; 20];
        grade_host[0] = 1.0;    // exposure
        grade_host[1] = 1.0;    // filmic
        grade_host[2] = 1.0;    // crush gamma (1 = off)
        grade_host[3] = 1.0;    // saturation
        grade_host[4] = 0.0;    // vignette
        grade_host[5] = 1.0; grade_host[6] = 1.0; grade_host[7] = 1.0;  // white balance
        grade_host[10] = 0.75;  // hue preservation
        grade_host[11] = 1.0 / 2.2;
        grade_host[12] = 0.0;   // grain

        Self {
            out: empty_buffer(&device, (3 * w * h * 4) as u64),
            bg: shared_buffer(&device, bg_data),
            bb_lut: shared_buffer(&device, &bb.table),
            star_lut: shared_buffer(&device, &stars.table),
            vol: vol_buf,
            vol_params: shared_buffer(&device, &vol_host),
            star_params: shared_buffer(&device, &[0.0f32; 16]),
            cam_params: shared_buffer(&device, &[0.0f32; CAM_PARAMS_N]),
            st_params: shared_buffer(&device, &[0.0f32; 8]),
            disc_params: shared_buffer(&device, &disc_host),
            grade: shared_buffer(&device, &grade_host),
            palette: shared_buffer(&device, &[0.0f32; 3 * PALETTE_MAX]),
            device, queue, trace_pipeline, pack_pipeline,
            width: w, height: h, bg_w, bg_h,
            dt: desc.dt, tol: desc.tol, nmax_floor: desc.nmax,
            r_escape_factor: desc.r_escape_factor,
            grade_host,
        }
    }

    /// Procedural point stars, evaluated at output resolution rather than read
    /// from a texture. A 4k equirectangular starmap is 0.088 deg per texel and
    /// a 1080p frame through a wide lens has pixels several times finer, so
    /// texture stars turn to mush; a procedural field has no such limit.
    pub fn set_starfield(&mut self, s: &StarSettings, fov_factor: f64) {
        let gn = (s.galactic[0].powi(2) + s.galactic[1].powi(2)
                  + s.galactic[2].powi(2)).sqrt();
        // The pixel's angular footprint. Sizing the PSF from this rather than
        // from a fixed angle is what keeps stars ~1 px at every resolution --
        // and setting it in the SOURCE sky means lensing stretches and
        // brightens star images near the critical curve on its own.
        let sigma = s.psf_pixels * 2.0 * fov_factor as f32 / self.height as f32;
        let host = unsafe { host_slice(&self.star_params, 16) };
        host.copy_from_slice(&[
            s.strength, s.density, s.fill, sigma, s.flux,
            s.galactic[0] / gn, s.galactic[1] / gn, s.galactic[2] / gn,
            s.concentration, s.temp_min, s.temp_max - s.temp_min,
            500.0, 30000.0, 1024.0, s.seed as f32, s.texture_weight,
        ]);
    }

    pub fn set_palette(&mut self, pal: Option<&[f32]>) {
        match pal {
            Some(p) => {
                let n = p.len() / 3;
                assert!(n <= PALETTE_MAX);
                let host = unsafe { host_slice(&self.palette, 3 * PALETTE_MAX) };
                host[..p.len()].copy_from_slice(p);
                self.grade_host[18] = n as f32;
            }
            None => self.grade_host[18] = 0.0,
        }
        self.flush_grade();
    }

    pub fn set_dither(&mut self, d: f32) { self.grade_host[17] = d; self.flush_grade(); }
    pub fn set_exposure(&mut self, e: f32) { self.grade_host[0] = e; self.flush_grade(); }

    fn flush_grade(&self) {
        let host = unsafe { host_slice(&self.grade, 20) };
        host.copy_from_slice(&self.grade_host);
    }

    /// Trace one frame and grade it into `drawable`.
    pub fn render(&mut self, cam: &Camera, st: &Spacetime, drawable: &TextureRef,
                  fisheye_deg: f64, relativistic: bool) {
        let m = st.m as f32;
        let pos_r = (cam.pos[0].powi(2) + cam.pos[1].powi(2) + cam.pos[2].powi(2)).sqrt();
        let r_escape = self.r_escape_factor * pos_r.max(15.0 * st.m) as f32;

        // Dynamic step count. With radius-adaptive steps the travel legs are
        // logarithmic in r_escape; the constant is the strong-field winding
        // budget (a few photon-sphere orbits). The cap keeps one dispatch
        // under the macOS GPU watchdog when the camera is very far out.
        let nmax = (((75.0 + 6.5 * (r_escape / m).ln() as f64) * st.m / self.dt as f64)
                    .ceil() as u32)
                   .max(self.nmax_floor)
                   .min(20_000);

        {
            let cp = unsafe { host_slice(&self.cam_params, CAM_PARAMS_N) };
            cam.write_params(cp, st.m, fisheye_deg);
            let (spin_a, rkill2) = st.spin_horizon();
            let sp = unsafe { host_slice(&self.st_params, 8) };
            sp[0] = m;
            sp[1] = 2.05 * m;
            sp[2] = r_escape;
            sp[3] = if relativistic { 1.0 } else { 0.0 };
            sp[4] = HSTEP_COEF;
            sp[5] = HSTEP_CAP;
            sp[6] = spin_a;
            sp[7] = rkill2;
        }

        let cmd = self.queue.new_command_buffer();

        // --- trace -----------------------------------------------------
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.trace_pipeline);
            enc.set_buffer(0, Some(&self.out), 0);
            enc.set_buffer(1, Some(&self.bg), 0);
            enc.set_buffer(2, Some(&self.bb_lut), 0);
            enc.set_buffer(3, Some(&self.star_lut), 0);
            enc.set_buffer(4, Some(&self.vol), 0);
            enc.set_buffer(5, Some(&self.vol_params), 0);
            enc.set_buffer(6, Some(&self.star_params), 0);
            enc.set_buffer(7, Some(&self.cam_params), 0);
            enc.set_buffer(8, Some(&self.st_params), 0);
            enc.set_buffer(9, Some(&self.disc_params), 0);
            let dims = [self.width as u32, self.height as u32,
                        self.bg_w as u32, self.bg_h as u32];
            enc.set_bytes(10, 16, dims.as_ptr() as *const _);
            let limits = [nmax, 0u32];
            enc.set_bytes(11, 8, limits.as_ptr() as *const _);
            let steps = [self.dt, self.tol];
            enc.set_bytes(12, 8, steps.as_ptr() as *const _);

            let n = (self.width * self.height) as u64;
            let tg = self.trace_pipeline.max_total_threads_per_threadgroup().min(n);
            enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
            enc.end_encoding();
        }

        // --- grade + present -------------------------------------------
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pack_pipeline);
            enc.set_texture(0, Some(drawable));
            enc.set_buffer(0, Some(&self.out), 0);
            enc.set_buffer(1, Some(&self.grade), 0);
            enc.set_buffer(2, Some(&self.palette), 0);
            let dw = drawable.width() as u32;
            let dh = drawable.height() as u32;
            let dims = [self.width as u32, self.height as u32, dw, dh];
            enc.set_bytes(3, 16, dims.as_ptr() as *const _);
            let escale = 1.0f32;
            enc.set_bytes(4, 4, &escale as *const f32 as *const _);
            enc.dispatch_threads(
                MTLSize::new(dw as u64, dh as u64, 1),
                MTLSize::new(16, 16, 1));
            enc.end_encoding();
        }

        cmd.commit();
    }

    pub fn commit_present(&self, drawable: &MetalDrawableRef) {
        let cmd = self.queue.new_command_buffer();
        cmd.present_drawable(drawable);
        cmd.commit();
    }
}

impl Renderer {
    /// Trace one frame with no grade and no presentation, blocking until the
    /// GPU is done, then hand back the linear HDR buffer as (3, W, H)
    /// channel-fastest. Used by the self-test; a frame loop never needs it.
    pub fn trace_blocking(&mut self, cam: &Camera, st: &Spacetime) -> Vec<f32> {
        let m = st.m as f32;
        let pos_r = (cam.pos[0].powi(2) + cam.pos[1].powi(2) + cam.pos[2].powi(2)).sqrt();
        let r_escape = self.r_escape_factor * pos_r.max(15.0 * st.m) as f32;
        let nmax = (((75.0 + 6.5 * (r_escape / m).ln() as f64) * st.m / self.dt as f64)
                    .ceil() as u32)
                   .max(self.nmax_floor)
                   .min(20_000);
        {
            let cp = unsafe { host_slice(&self.cam_params, CAM_PARAMS_N) };
            cam.write_params(cp, st.m, 0.0);
            let (spin_a, rkill2) = st.spin_horizon();
            let sp = unsafe { host_slice(&self.st_params, 8) };
            sp[0] = m; sp[1] = 2.05 * m; sp[2] = r_escape; sp[3] = 0.0;
            sp[4] = HSTEP_COEF; sp[5] = HSTEP_CAP; sp[6] = spin_a; sp[7] = rkill2;
        }

        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.trace_pipeline);
        enc.set_buffer(0, Some(&self.out), 0);
        enc.set_buffer(1, Some(&self.bg), 0);
        enc.set_buffer(2, Some(&self.bb_lut), 0);
        enc.set_buffer(3, Some(&self.star_lut), 0);
        enc.set_buffer(4, Some(&self.vol), 0);
        enc.set_buffer(5, Some(&self.vol_params), 0);
        enc.set_buffer(6, Some(&self.star_params), 0);
        enc.set_buffer(7, Some(&self.cam_params), 0);
        enc.set_buffer(8, Some(&self.st_params), 0);
        enc.set_buffer(9, Some(&self.disc_params), 0);
        let dims = [self.width as u32, self.height as u32,
                    self.bg_w as u32, self.bg_h as u32];
        enc.set_bytes(10, 16, dims.as_ptr() as *const _);
        let limits = [nmax, 0u32];
        enc.set_bytes(11, 8, limits.as_ptr() as *const _);
        let steps = [self.dt, self.tol];
        enc.set_bytes(12, 8, steps.as_ptr() as *const _);
        let n = (self.width * self.height) as u64;
        let tg = self.trace_pipeline.max_total_threads_per_threadgroup().min(n);
        enc.dispatch_threads(MTLSize::new(n, 1, 1), MTLSize::new(tg, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let n = 3 * self.width * self.height;
        unsafe { std::slice::from_raw_parts(self.out.contents() as *const f32, n) }.to_vec()
    }
}
