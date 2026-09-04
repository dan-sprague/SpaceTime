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
    /// Chroma scale about Rec.709 luminance in the star colour LUT. Raw
    /// blackbody chroma (1.0) is far more saturated than any star looks;
    /// measured star chromaticities top out at pale blue-white, so production
    /// skies use ~0.35. Port of the `saturation` knob in src/metal.jl.
    pub saturation: f32,
}

impl Default for StarSettings {
    fn default() -> Self {
        Self {
            strength: 1.0, texture_weight: 0.0, density: 384.0, fill: 0.5,
            flux: 0.011, psf_pixels: 0.5, galactic: [0.0, 0.0, 1.0],
            concentration: 3.0, temp_min: 3000.0, temp_max: 16000.0, seed: 12345,
            saturation: 1.0,
        }
    }
}

pub struct Renderer {
    pub device: Device,
    queue: CommandQueue,
    trace_pipeline: ComputePipelineState,
    pack_pipeline: ComputePipelineState,
    overlay_pipeline: ComputePipelineState,

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
    font_atlas: Buffer,
    ov_cells: Buffer,
    ov_attr: Buffer,
    ov_glyph_count: i32,

    pub width: usize,
    pub height: usize,
    bg_w: usize,
    bg_h: usize,
    pub dt: f32,
    pub tol: f32,
    pub nmax_floor: u32,
    pub r_escape_factor: f32,
    /// Distance from the origin to the second hole in the Binary scene (0 when
    /// single). The ray escape radius keys off distance from the origin, so it
    /// must be widened to always contain hole 2, whose neighbourhood is where
    /// rays lens even when the camera is mid-gap and close to the origin.
    bh2_r: f32,
    grade_host: [f32; 20],
    /// Raw (unsaturated) star colour LUT, so `set_starfield` can re-bake the
    /// GPU LUT at any chroma without recomputing blackbody colours.
    star_table_raw: Vec<f32>,
    /// MetalFX spatial upscaling. When on, `pack` writes an internal-res
    /// texture, the scaler upsizes it to the drawable, and a blit presents it.
    pub upscale_metalfx: bool,
    internal_color: Option<Texture>,
    mfx: Option<crate::metalfx::SpatialScaler>,
    mfx_output: Option<Texture>,
    mfx_dims: (u32, u32),
    /// A CPU-rasterised system map (texture, width, height) blitted into the
    /// drawable's corner each frame. `None` until the first `set_minimap`.
    minimap: Option<(Texture, u32, u32)>,
}

/// A private-storage colour texture usable as both a compute target and a
/// MetalFX input/output (and a blit source).
fn make_color_texture(device: &Device, w: u32, h: u32) -> Texture {
    let td = TextureDescriptor::new();
    td.set_texture_type(MTLTextureType::D2);
    td.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
    td.set_width(w as u64);
    td.set_height(h as u64);
    td.set_storage_mode(MTLStorageMode::Private);
    td.set_usage(MTLTextureUsage::ShaderRead | MTLTextureUsage::ShaderWrite
                 | MTLTextureUsage::RenderTarget);
    device.new_texture(&td)
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
    /// Arc length between volumetric gas samples, in units of M. Matches the
    /// old every-2nd-step density at the default step size.
    pub gas_stride: f32,
    /// A second Schwarzschild hole `(centre, mass)`, superposed on the first
    /// (which stays at the origin). `None` for a single hole. Schwarzschild
    /// only -- ignored when the spacetime is Kerr.
    pub binary: Option<([f64; 3], f64)>,
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
        let is_binary = desc.binary.is_some() && !kerr;
        consts.set_constant_value_at_index(
            &is_binary as *const bool as *const _, MTLDataType::Bool, 3);
        if let Some((c, m)) = desc.binary {
            let v = [c[0] as f32, c[1] as f32, c[2] as f32, m as f32];
            for (k, val) in v.iter().enumerate() {
                consts.set_constant_value_at_index(
                    val as *const f32 as *const _, MTLDataType::Float, 4 + k as u64);
            }
        }

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
        let overlay_fn = library.get_function("overlay", None).expect("missing kernel `overlay`");
        let overlay_pipeline = device
            .new_compute_pipeline_state_with_function(&overlay_fn)
            .expect("failed to build overlay pipeline");

        let (w, h) = (desc.width, desc.height);
        let (bg_data, bg_w, bg_h) = desc.background;

        // The disc's blackbody LUT and the STAR LUT are deliberately separate.
        // Sharing one made star colour a function of the disc's white balance,
        // so regrading the disc re-tinted the whole sky.
        let bb = match &desc.disc {
            // 6000 K white point. The disc's hottest, brightest ring (the ISCO
            // edge, ~10000 K) sits well above this, so the bright inner disc
            // reads blue-white; the T ~ R^-3/4 gradient only crosses neutral
            // out past ~R=6, leaving a warm gold fringe on the dim outer disc.
            // A 10000 K white point put the whole disc at or below neutral and
            // it came out uniformly golden.
            Some(_) => Blackbody::new(6000.0, 1024),
            None => Blackbody::new(6500.0, 1024),
        };
        let stars = Blackbody::new(STAR_WB_TEMPERATURE, 1024);

        let disc_host: [f32; 6] = match &desc.disc {
            Some(d) => [d.inner_radius as f32, d.outer_radius as f32,
                        d.density_falloff as f32, bb.table_min as f32,
                        bb.table_max as f32, bb.table_size as f32],
            None => [0.0; 6],
        };

        // vol_params[9] is the disc clock (M-time), written per frame; [10] is
        // the spin, so the sampler can apply the same differential Keplerian
        // shear Omega(s) = 1/(s^1.5 + a) the film bakes with `rotate_volume!`
        // -- done analytically here so a spinning disc costs nothing per frame.
        let (vol_buf, vol_host) = match desc.volume {
            Some(v) => (
                shared_buffer(&device, &v.density),
                [v.log_s_in, v.log_s_out, v.z_max, v.nr as f32, v.nphi as f32,
                 v.nz as f32, v.emission_scale, v.opacity_scale, desc.gas_stride,
                 0.0, desc.spacetime.a as f32],
            ),
            None => (empty_buffer(&device, 4), [0.0f32; 11]),
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
            star_table_raw: stars.table.clone(),
            vol: vol_buf,
            vol_params: shared_buffer(&device, &vol_host),
            star_params: shared_buffer(&device, &[0.0f32; 16]),
            cam_params: shared_buffer(&device, &[0.0f32; CAM_PARAMS_N]),
            st_params: shared_buffer(&device, &[0.0f32; 8]),
            disc_params: shared_buffer(&device, &disc_host),
            grade: shared_buffer(&device, &grade_host),
            palette: shared_buffer(&device, &[0.0f32; 3 * PALETTE_MAX]),
            font_atlas: shared_buffer(&device, &crate::font::atlas()),
            // Sized for a generous console; the menu grid is far smaller.
            ov_cells: shared_buffer(&device, &[0i32; 64 * 32]),
            ov_attr: shared_buffer(&device, &[0i32; 64 * 32]),
            ov_glyph_count: crate::font::glyph_count() as i32,
            upscale_metalfx: false,
            internal_color: None,
            mfx: None,
            mfx_output: None,
            mfx_dims: (0, 0),
            minimap: None,
            device, queue, trace_pipeline, pack_pipeline, overlay_pipeline,
            width: w, height: h, bg_w, bg_h,
            dt: desc.dt, tol: desc.tol, nmax_floor: desc.nmax,
            r_escape_factor: desc.r_escape_factor,
            bh2_r: desc.binary
                .map(|(c, _)| (c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt() as f32)
                .unwrap_or(0.0),
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

        // Bake chroma saturation into the star LUT (scale each entry about its
        // Rec.709 luminance), matching src/metal.jl's `_star_lut_cpu`.
        {
            let lut = unsafe { host_slice(&self.star_lut, self.star_table_raw.len()) };
            let sat = s.saturation;
            for i in 0..self.star_table_raw.len() / 3 {
                let (r, g, b) = (self.star_table_raw[3 * i],
                                 self.star_table_raw[3 * i + 1],
                                 self.star_table_raw[3 * i + 2]);
                let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                lut[3 * i]     = y + sat * (r - y);
                lut[3 * i + 1] = y + sat * (g - y);
                lut[3 * i + 2] = y + sat * (b - y);
            }
        }

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
    pub fn set_filmic(&mut self, on: bool) {
        self.grade_host[1] = if on { 1.0 } else { 0.0 }; self.flush_grade();
    }
    pub fn set_saturation(&mut self, s: f32) { self.grade_host[3] = s; self.flush_grade(); }
    /// Hue preservation in the filmic tonemap: 0 = per-channel ACES (highlights
    /// desaturate toward white, the "film" look), 1 = tonemap luminance only
    /// and keep chromaticity (highlights hold their true blackbody hue).
    pub fn set_hue_preserve(&mut self, h: f32) { self.grade_host[10] = h; self.flush_grade(); }
    /// Contrast/crush gamma applied after tonemap (1.0 = off, >1 deepens mids).
    pub fn set_crush(&mut self, g: f32) { self.grade_host[2] = g; self.flush_grade(); }
    pub fn set_vignette(&mut self, v: f32) { self.grade_host[4] = v; self.flush_grade(); }
    pub fn set_grain(&mut self, g: f32) { self.grade_host[12] = g; self.flush_grade(); }
    /// Per-channel white-balance multiplier applied to the linear scene.
    pub fn set_white_balance(&mut self, r: f32, g: f32, b: f32) {
        self.grade_host[5] = r; self.grade_host[6] = g; self.grade_host[7] = b;
        self.flush_grade();
    }
    /// Upscale filter for the internal target: false = nearest (pixel-art),
    /// true = bilinear (smooth).
    pub fn set_upscale_smooth(&mut self, smooth: bool) {
        self.grade_host[13] = if smooth { 1.0 } else { 0.0 }; self.flush_grade();
    }

    /// Advance the disc's rotation clock (in M-time). The gas sampler shears
    /// azimuth by Omega(s) = 1/(s^1.5 + a) per unit time, so the disc spins
    /// differentially -- inner gas faster -- exactly as in the offline film.
    pub fn set_disc_time(&self, t_m: f32) {
        let vp = unsafe { host_slice(&self.vol_params, 11) };
        vp[9] = t_m;
    }

    /// Lazily build the MetalFX input texture, scaler, and output texture for
    /// the current internal resolution and the given drawable size, rebuilding
    /// the scaler and output when the drawable size changes.
    fn ensure_mfx(&mut self, out_w: u32, out_h: u32) {
        if self.internal_color.is_none() {
            self.internal_color =
                Some(make_color_texture(&self.device, self.width as u32, self.height as u32));
        }
        if self.mfx.is_none() || self.mfx_dims != (out_w, out_h) {
            self.mfx = crate::metalfx::SpatialScaler::new(
                &self.device, self.width as u32, self.height as u32,
                out_w, out_h, crate::metalfx::BGRA8UNORM);
            self.mfx_output = self.mfx.as_ref()
                .map(|_| make_color_texture(&self.device, out_w, out_h));
            self.mfx_dims = (out_w, out_h);
        }
    }

    fn flush_grade(&self) {
        let host = unsafe { host_slice(&self.grade, 20) };
        host.copy_from_slice(&self.grade_host);
    }

    /// Upload a CPU-rasterised BGRA system map; it is blitted into the
    /// drawable corner each `render_and_present`. The texture is (re)allocated
    /// only when the panel size changes.
    pub fn set_minimap(&mut self, w: u32, h: u32, bgra: &[u8]) {
        let need = !matches!(&self.minimap, Some((_, tw, th)) if *tw == w && *th == h);
        if need {
            let td = TextureDescriptor::new();
            td.set_texture_type(MTLTextureType::D2);
            td.set_pixel_format(MTLPixelFormat::BGRA8Unorm);
            td.set_width(w as u64);
            td.set_height(h as u64);
            td.set_storage_mode(MTLStorageMode::Shared);
            td.set_usage(MTLTextureUsage::ShaderRead);
            self.minimap = Some((self.device.new_texture(&td), w, h));
        }
        let tex = &self.minimap.as_ref().unwrap().0;
        tex.replace_region(
            MTLRegion {
                origin: MTLOrigin { x: 0, y: 0, z: 0 },
                size: MTLSize { width: w as u64, height: h as u64, depth: 1 },
            },
            0, bgra.as_ptr() as *const _, (w * 4) as u64);
    }

    /// Trace one frame, grade it into the drawable, optionally blit a menu
    /// overlay on top, and present -- all on a single command buffer. Splitting
    /// the present onto a second buffer costs an extra commit and scheduling
    /// round trip per frame for nothing: the queue already orders them.
    pub fn render_and_present(&mut self, cam: &Camera, st: &Spacetime,
                              drawable: &MetalDrawableRef,
                              fisheye_deg: f64, relativistic: bool,
                              overlay: Option<Overlay>) {
        let texture = drawable.texture();
        let m = st.m as f32;
        let pos_r = (cam.pos[0].powi(2) + cam.pos[1].powi(2) + cam.pos[2].powi(2)).sqrt();
        let r_escape = self.r_escape_factor
            * pos_r.max(15.0 * st.m).max(self.bh2_r as f64 + 25.0) as f32;

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
            cam.write_params(cp, st.m, st.a, fisheye_deg);
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

        // MetalFX spatial upscaling only applies when the drawable is larger
        // than the internal target; otherwise fall back to the pack blit.
        // `ensure_mfx` may fail to build a scaler (older OS), so re-check after.
        let dw = texture.width() as u32;
        let dh = texture.height() as u32;
        let mut use_mfx = self.upscale_metalfx
            && dw > self.width as u32 && dh > self.height as u32;
        if use_mfx { self.ensure_mfx(dw, dh); use_mfx = self.mfx.is_some(); }

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

        // --- grade -----------------------------------------------------
        // Without MetalFX, `pack` writes the drawable directly, upscaling as it
        // goes. With MetalFX, it writes the internal-res texture 1:1 and the
        // scaler does the upscale below.
        let pack_tex: &TextureRef = if use_mfx {
            self.internal_color.as_deref().unwrap()
        } else {
            texture
        };
        let (pdw, pdh) = if use_mfx { (self.width as u32, self.height as u32) } else { (dw, dh) };
        {
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pack_pipeline);
            enc.set_texture(0, Some(pack_tex));
            enc.set_buffer(0, Some(&self.out), 0);
            enc.set_buffer(1, Some(&self.grade), 0);
            enc.set_buffer(2, Some(&self.palette), 0);
            let dims = [self.width as u32, self.height as u32, pdw, pdh];
            enc.set_bytes(3, 16, dims.as_ptr() as *const _);
            let escale = 1.0f32;
            enc.set_bytes(4, 4, &escale as *const f32 as *const _);
            enc.dispatch_threads(
                MTLSize::new(pdw as u64, pdh as u64, 1),
                MTLSize::new(16, 16, 1));
            enc.end_encoding();
        }

        // --- MetalFX upscale + blit to the drawable --------------------
        if use_mfx {
            let scaler = self.mfx.as_ref().unwrap();
            let internal = self.internal_color.as_ref().unwrap();
            let out_tex = self.mfx_output.as_ref().unwrap();
            scaler.encode(cmd, internal, out_tex);
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_texture(
                out_tex, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
                MTLSize { width: dw as u64, height: dh as u64, depth: 1 },
                texture, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 });
            blit.end_encoding();
        }

        // --- menu overlay (optional) -----------------------------------
        if let Some(ov) = overlay {
            let dw = texture.width() as i32;
            let dh = texture.height() as i32;
            // One font pixel spans `ps` device pixels; a cell is the glyph plus
            // one column / two rows of spacing. Scale with the drawable so the
            // menu stays legible at any window size.
            let ps = ((dh / 320).max(2)) as i32;
            let gw = crate::font::GLYPH_W as i32;
            let gh = crate::font::GLYPH_H as i32;
            let cw = (gw + 1) * ps;
            let ch = (gh + 2) * ps;
            let cols = ov.cols as i32;
            let rows = ov.rows as i32;
            let panel_w = cols * cw;
            let panel_h = rows * ch;
            let ox = (dw - panel_w) / 2;
            let oy = (dh - panel_h) / 2;
            let pad = 3 * ps;

            {
                let cells = unsafe { host_slice_i32(&self.ov_cells, ov.cells.len()) };
                cells.copy_from_slice(ov.cells);
                let attr = unsafe { host_slice_i32(&self.ov_attr, ov.attr.len()) };
                attr.copy_from_slice(ov.attr);
            }
            let ip: [i32; 11] = [cols, rows, ox, oy, cw, ch, ps, gw, gh, pad,
                                 self.ov_glyph_count];

            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.overlay_pipeline);
            enc.set_texture(0, Some(texture));
            enc.set_buffer(0, Some(&self.font_atlas), 0);
            enc.set_buffer(1, Some(&self.ov_cells), 0);
            enc.set_buffer(2, Some(&self.ov_attr), 0);
            enc.set_bytes(3, (11 * 4) as u64, ip.as_ptr() as *const _);
            enc.dispatch_threads(
                MTLSize::new(dw as u64, dh as u64, 1),
                MTLSize::new(16, 16, 1));
            enc.end_encoding();
        }

        // --- system map (optional), blitted into the top-right corner -----
        if let Some((mm, mw, mh)) = &self.minimap {
            let (dwid, dhei) = (texture.width(), texture.height());
            let margin = 16u64;
            if dwid > *mw as u64 + margin && dhei > *mh as u64 + margin {
                let blit = cmd.new_blit_command_encoder();
                blit.copy_from_texture(
                    mm, 0, 0, MTLOrigin { x: 0, y: 0, z: 0 },
                    MTLSize { width: *mw as u64, height: *mh as u64, depth: 1 },
                    texture, 0, 0,
                    MTLOrigin { x: dwid - *mw as u64 - margin, y: margin, z: 0 });
                blit.end_encoding();
            }
        }

        cmd.present_drawable(drawable);
        cmd.commit();
    }
}

/// A character grid to blit over the presented frame. `cells` holds font-atlas
/// slots and `attr` the per-cell colour attribute, both row-major `cols x rows`.
pub struct Overlay<'a> {
    pub cols: usize,
    pub rows: usize,
    pub cells: &'a [i32],
    pub attr: &'a [i32],
}

/// Host view of a shared i32 buffer, for uploading the overlay grid.
unsafe fn host_slice_i32(b: &Buffer, n: usize) -> &mut [i32] {
    std::slice::from_raw_parts_mut(b.contents() as *mut i32, n)
}

impl Renderer {
    /// Trace one frame with no grade and no presentation, blocking until the
    /// GPU is done, then hand back the linear HDR buffer as (3, W, H)
    /// channel-fastest. Used by the self-test; a frame loop never needs it.
    pub fn trace_blocking(&mut self, cam: &Camera, st: &Spacetime) -> Vec<f32> {
        let m = st.m as f32;
        let pos_r = (cam.pos[0].powi(2) + cam.pos[1].powi(2) + cam.pos[2].powi(2)).sqrt();
        let r_escape = self.r_escape_factor
            * pos_r.max(15.0 * st.m).max(self.bh2_r as f64 + 25.0) as f32;
        let nmax = (((75.0 + 6.5 * (r_escape / m).ln() as f64) * st.m / self.dt as f64)
                    .ceil() as u32)
                   .max(self.nmax_floor)
                   .min(20_000);
        {
            let cp = unsafe { host_slice(&self.cam_params, CAM_PARAMS_N) };
            cam.write_params(cp, st.m, st.a, 0.0);
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
