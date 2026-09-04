//! A thin objc bridge to `MTLFXSpatialScaler`. The `metal` crate (0.27) has no
//! MetalFX bindings, but the framework ships with macOS, so we message the
//! Objective-C objects directly.
//!
//! Spatial (not temporal) is the right tool for this renderer: it upscales a
//! single frame with no motion vectors. Temporal would need per-pixel motion,
//! and the only cheap estimate here -- camera reprojection -- is wrong exactly
//! at the photon ring, where a small camera move sweeps the lensed images
//! around the shadow at large apparent speed.

use metal::foreign_types::{ForeignType, ForeignTypeRef};
use metal::{CommandBufferRef, Device, Texture};
use objc::runtime::{Class, Object};
use objc::{msg_send, sel, sel_impl};

/// `MTLPixelFormatBGRA8Unorm`, the drawable's format.
pub const BGRA8UNORM: u64 = 80;

/// Owns an `id<MTLFXSpatialScaler>` built for a fixed input/output size.
pub struct SpatialScaler {
    scaler: *mut Object,
    pub in_w: u32,
    pub in_h: u32,
    pub out_w: u32,
    pub out_h: u32,
}

impl SpatialScaler {
    /// Build a scaler, or `None` if MetalFX is unavailable (older OS / device)
    /// so the caller can fall back to bilinear.
    pub fn new(device: &Device, in_w: u32, in_h: u32, out_w: u32, out_h: u32,
               fmt: u64) -> Option<Self> {
        let cls = Class::get("MTLFXSpatialScalerDescriptor")?;
        unsafe {
            let desc: *mut Object = msg_send![cls, alloc];
            let desc: *mut Object = msg_send![desc, init];
            if desc.is_null() { return None; }
            let _: () = msg_send![desc, setInputWidth: in_w as u64];
            let _: () = msg_send![desc, setInputHeight: in_h as u64];
            let _: () = msg_send![desc, setOutputWidth: out_w as u64];
            let _: () = msg_send![desc, setOutputHeight: out_h as u64];
            let _: () = msg_send![desc, setColorTextureFormat: fmt];
            let _: () = msg_send![desc, setOutputTextureFormat: fmt];
            // MTLFXSpatialScalerColorProcessingModePerceptual = 0: the input is
            // gamma-encoded display colour, which is what `pack` writes.
            let _: () = msg_send![desc, setColorProcessingMode: 0i64];
            let scaler: *mut Object = msg_send![desc, newSpatialScalerWithDevice: device.as_ptr()];
            let _: () = msg_send![desc, release];
            if scaler.is_null() { return None; }
            Some(Self { scaler, in_w, in_h, out_w, out_h })
        }
    }

    /// Encode the upscale of `color` (input size) into `output` (output size)
    /// onto `cmd`.
    pub fn encode(&self, cmd: &CommandBufferRef, color: &Texture, output: &Texture) {
        unsafe {
            let _: () = msg_send![self.scaler, setColorTexture: color.as_ptr()];
            let _: () = msg_send![self.scaler, setOutputTexture: output.as_ptr()];
            let _: () = msg_send![self.scaler, encodeToCommandBuffer: cmd.as_ptr()];
        }
    }
}

impl Drop for SpatialScaler {
    fn drop(&mut self) {
        unsafe { let _: () = msg_send![self.scaler, release]; }
    }
}
