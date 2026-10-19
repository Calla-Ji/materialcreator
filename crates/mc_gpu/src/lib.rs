// SPDX-License-Identifier: MIT OR Apache-2.0
//! mc_gpu: shared wgpu device/queue, a texture pool, and a compute pipeline cache.
//!
//! Blender/Unreal analogy: this is the "render engine" plumbing underneath the
//! node graph. Nodes never talk to the GPU directly — they ask a [`Context`]
//! to compile their WGSL compute shader and dispatch it, and get a texture
//! back. `mc_app` shares eframe's own wgpu device with this crate (so the UI
//! and node textures live on one GPU); `mc_cli` and tests create a standalone
//! headless device instead.

use std::collections::HashMap;
use std::future::Future;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

use anyhow::{Context as _, Result};

/// A GPU device/queue pair, plus the compute pipelines compiled from it.
pub struct Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// `None` when wrapping an externally-owned device (`Context::new`,
    /// e.g. eframe's), since only `new_headless` requests its own adapter.
    adapter_info: Option<wgpu::AdapterInfo>,
    pipelines: HashMap<(String, u64), Arc<wgpu::ComputePipeline>>,
}

impl Context {
    /// Wrap an existing device/queue, e.g. eframe's wgpu render state.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Self {
        Self {
            device,
            queue,
            adapter_info: None,
            pipelines: HashMap::new(),
        }
    }

    /// Create a standalone device/queue with no window. Used by `mc_cli` and
    /// tests, which never open an eframe window.
    pub fn new_headless() -> Result<Self> {
        block_on(Self::new_headless_async())
    }

    async fn new_headless_async() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .context("no suitable GPU adapter found")?;
        let adapter_info = adapter.get_info();
        // Ask for the adapter's own real limits rather than
        // `wgpu::Limits::default()`'s conservative 256 MiB buffer cap — see
        // `mc_app::main`'s matching override for why (the content-adaptive
        // mesh can need a bigger single vertex buffer at high Tessellation).
        let base_limits = if adapter_info.backend == wgpu::Backend::Gl {
            wgpu::Limits::downlevel_webgl2_defaults()
        } else {
            adapter.limits()
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("mc_gpu headless device"),
                required_limits: base_limits,
                ..Default::default()
            })
            .await
            .context("failed to create GPU device")?;
        let mut ctx = Self::new(device, queue);
        ctx.adapter_info = Some(adapter_info);
        Ok(ctx)
    }

    /// True when this `Context` is backed by a CPU-emulated adapter (e.g.
    /// Lavapipe/llvmpipe, GitHub's `ubuntu-latest` runners' only Vulkan
    /// fallback — no real GPU is present there at all). No end user's
    /// machine ever renders through this path; it exists purely so
    /// `mc_app`/`mc_cli`'s GPU-touching tests have *some* adapter to run
    /// against in CI. Software Vulkan implementations are real, useful
    /// Vulkan conformance targets, but aren't held to the same shader
    /// compiler quality bar as a shipping GPU driver — see
    /// `mc_app::tess_gpu`'s golden-precision tests for the real bug this
    /// was added to work around (a confirmed Lavapipe JIT miscompile,
    /// independently root-caused, not a bug in this project's own code).
    pub fn is_software_adapter(&self) -> bool {
        self.adapter_info
            .as_ref()
            .is_some_and(|info| info.device_type == wgpu::DeviceType::Cpu)
    }

    /// Get or compile the compute pipeline for a node's shader source.
    ///
    /// The cache key includes a hash of `wgsl_source`, so editing a shader
    /// file on disk (hot-reload, debug builds — see `mc_nodes::shader_source`)
    /// transparently recompiles it the next time this is called.
    pub fn compute_pipeline(
        &mut self,
        node_kind: &str,
        wgsl_source: &str,
        entry_point: &str,
    ) -> Arc<wgpu::ComputePipeline> {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        wgsl_source.hash(&mut hasher);
        let cache_key = (node_kind.to_string(), hasher.finish());

        if let Some(pipeline) = self.pipelines.get(&cache_key) {
            return pipeline.clone();
        }

        // Drop any stale pipeline(s) for this node kind (e.g. the pre-edit shader).
        self.pipelines.retain(|(kind, _), _| kind != node_kind);

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(node_kind),
                source: wgpu::ShaderSource::Wgsl(wgsl_source.into()),
            });
        let pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(node_kind),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            },
        ));

        self.pipelines.insert(cache_key, pipeline.clone());
        pipeline
    }

    /// Fill `key`'s texture in `pool` with a single solid RGBA color (0-255
    /// per channel), creating it at `size x size` if needed. Used for a
    /// node's optional inputs that fall back to a flat default (e.g. Blend's
    /// mask defaulting to solid white) rather than running a shader over
    /// them.
    pub fn fill_rgba8(&self, pool: &mut TexturePool, key: &str, size: u32, color: [u8; 4]) {
        pool.get_or_create(&self.device, key, size, wgpu::TextureFormat::Rgba8Unorm);
        let texture = pool.texture(key).expect("just created above");

        let pixels: Vec<u8> = color.repeat((size * size) as usize);
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Read a square `Rgba8Unorm` or `Rgba16Float` texture back to a
    /// tightly-packed `Vec<u8>` of `resolution * resolution * 4` bytes (one
    /// row after another, no padding) — used for PNG export and for tests
    /// that inspect pixels.
    ///
    /// Every node's own output texture has been `Rgba16Float` since the
    /// banding fix (see `mc_nodes`' node files' own doc comments) — storing
    /// each node's result at 16-bit float instead of 8-bit keeps a chain of
    /// several filters from re-quantizing to 256 levels at *every single
    /// hop*, which is what caused visible banding in smooth gradients after
    /// a few nodes. This function still always hands callers back plain
    /// 8-bit RGBA, by downconverting here, at the one point precision
    /// actually needs to drop to match what a screen or a PNG can hold —
    /// so every existing caller (PNG export, the 2D View's display-encode
    /// pass, every node's own tests) keeps working completely unchanged.
    /// A handful of textures are deliberately still `Rgba8Unorm` (flat
    /// fallback colors, imported images, the bundled blue-noise mask, and
    /// the app's own final-display/3D-viewport surfaces) — none of those
    /// lose anything by staying 8-bit, since they either have no gradient
    /// to begin with or are already the last step before the screen.
    pub fn read_rgba8(&self, texture: &wgpu::Texture, resolution: u32) -> Vec<u8> {
        match texture.format() {
            wgpu::TextureFormat::Rgba16Float => {
                let raw = self.read_raw(texture, resolution, 8);
                raw.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|half_bytes| {
                        let half = u16::from_le_bytes(*half_bytes);
                        (half_to_f32(half).clamp(0.0, 1.0) * 255.0).round() as u8
                    })
                    .collect()
            }
            wgpu::TextureFormat::Rgba32Float => {
                let raw = self.read_raw(texture, resolution, 16);
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|f_bytes| {
                        (f32::from_le_bytes(*f_bytes).clamp(0.0, 1.0) * 255.0).round() as u8
                    })
                    .collect()
            }
            _ => self.read_raw(texture, resolution, 4),
        }
    }

    /// Fill `key`'s texture in `pool` with a single solid RGBA color, as
    /// `Rgba16Float` — same purpose as [`Self::fill_rgba8`] (a node's
    /// optional input defaulting to a flat value when nothing's connected),
    /// but for inputs that need real floating-point values rather than
    /// 0..1 display colors (e.g. Scattered Streaks' `angle_map` input,
    /// which defaults to a flat 0.0 radians — "no extra rotation" — when
    /// unconnected, and must be able to hold a connected upstream node's
    /// genuine `Rgba16Float` output, not just 0..255).
    pub fn fill_rgba16f(&self, pool: &mut TexturePool, key: &str, size: u32, value: [f32; 4]) {
        let pixel = value.map(f32_to_half);
        let pixels: Vec<u16> = pixel.repeat((size * size) as usize);
        self.write_rgba16f_raw(pool, key, size, u16s_to_le_bytes(&pixels));
    }

    /// Write `pixels` (one `[f32; 4]` per texel, row by row) into `key`'s
    /// texture in `pool`, as `Rgba16Float`. Used by tests that need to seed
    /// a node's input with specific float data (e.g. an `.npy`-sourced
    /// golden angle map) rather than a flat fallback.
    pub fn write_rgba16f(&self, pool: &mut TexturePool, key: &str, size: u32, pixels: &[[f32; 4]]) {
        assert_eq!(
            pixels.len(),
            (size * size) as usize,
            "write_rgba16f: expected {} pixels, got {}",
            size * size,
            pixels.len()
        );
        let halves: Vec<u16> = pixels.iter().flat_map(|p| p.map(f32_to_half)).collect();
        self.write_rgba16f_raw(pool, key, size, u16s_to_le_bytes(&halves));
    }

    fn write_rgba16f_raw(&self, pool: &mut TexturePool, key: &str, size: u32, bytes: Vec<u8>) {
        pool.get_or_create(&self.device, key, size, wgpu::TextureFormat::Rgba16Float);
        let texture = pool.texture(key).expect("just created above");
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 8),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Write `pixels` (one `[f32; 4]` per texel, row by row) into `key`'s
    /// texture in `pool`, as full-precision `Rgba32Float` — for tests that
    /// need to feed a node a float input at the same precision its own
    /// golden fixture was generated at, where `Rgba16Float`'s ~3-4
    /// significant decimal digits would introduce error the comparison
    /// itself isn't supposed to be measuring (confirmed concretely for
    /// Scattered Streaks' `angle_map` input: switching from `Rgba16Float`
    /// to this dropped `streaks_flowmap`'s `height` max error from ~0.13,
    /// with 0.29% of pixels outside tolerance, to ~0.002 with none).
    pub fn write_rgba32f(&self, pool: &mut TexturePool, key: &str, size: u32, pixels: &[[f32; 4]]) {
        assert_eq!(
            pixels.len(),
            (size * size) as usize,
            "write_rgba32f: expected {} pixels, got {}",
            size * size,
            pixels.len()
        );
        pool.get_or_create(&self.device, key, size, wgpu::TextureFormat::Rgba32Float);
        let texture = pool.texture(key).expect("just created above");
        let bytes: Vec<u8> = pixels
            .iter()
            .flat_map(|p| p.iter().flat_map(|c| c.to_le_bytes()))
            .collect();
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 16),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Reads a square `Rgba16Float` texture back to `resolution * resolution
    /// * 4` `f32`s (R, G, B, A per pixel, row by row), with no 0..1 clamp —
    /// unlike `read_rgba8`, which is for display/export and intentionally
    /// clamps. Needed for node outputs that are genuinely allowed outside
    /// 0..1 but don't need full `f32` storage precision.
    pub fn read_rgba16f(&self, texture: &wgpu::Texture, resolution: u32) -> Vec<f32> {
        let raw = self.read_raw(texture, resolution, 8);
        raw.as_chunks::<2>()
            .0
            .iter()
            .map(|half_bytes| half_to_f32(u16::from_le_bytes(*half_bytes)))
            .collect()
    }

    /// Reads a square `Rgba32Float` texture back to `resolution * resolution
    /// * 4` `f32`s, unclamped, full precision.
    ///
    /// Needed for node outputs whose required precision exceeds
    /// `Rgba16Float`'s ~3-4 significant decimal digits. Gabor Noise's `raw`
    /// output is the motivating case: a unit-variance float reaching
    /// magnitude ~3-4, where `Rgba16Float`'s mantissa only resolves to
    /// about 2e-3 there, too coarse for the reference package's own 1e-3
    /// tolerance on that output. Confirmed by measuring the exact
    /// quantization step before switching this one output's texture
    /// format.
    pub fn read_rgba32f(&self, texture: &wgpu::Texture, resolution: u32) -> Vec<f32> {
        let raw = self.read_raw(texture, resolution, 16);
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|f_bytes| f32::from_le_bytes(*f_bytes))
            .collect()
    }

    /// Reads a square `r32sint` texture back to `resolution * resolution`
    /// `i32`s, row by row — used by `mc_nodes::distance_spread`'s CPU-side
    /// connected-component labeling (Luminance Jitter), which needs the
    /// seed pass's own packed-coordinate texture on the CPU to run a real
    /// union-find rather than many GPU relaxation rounds.
    pub fn read_r32i(&self, texture: &wgpu::Texture, resolution: u32) -> Vec<i32> {
        let raw = self.read_raw(texture, resolution, 4);
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|i_bytes| i32::from_le_bytes(*i_bytes))
            .collect()
    }

    /// Writes `pixels` (one `i32` per texel, row by row) into `key`'s
    /// texture in `pool`, as `r32sint` — the write-side counterpart to
    /// [`Self::read_r32i`], for uploading a CPU-computed label texture back
    /// to the GPU.
    pub fn write_r32i(&self, pool: &mut TexturePool, key: &str, size: u32, pixels: &[i32]) {
        assert_eq!(
            pixels.len(),
            (size * size) as usize,
            "write_r32i: expected {} pixels, got {}",
            size * size,
            pixels.len()
        );
        pool.get_or_create(&self.device, key, size, wgpu::TextureFormat::R32Sint);
        let texture = pool.texture(key).expect("just created above");
        let bytes: Vec<u8> = pixels.iter().flat_map(|p| p.to_le_bytes()).collect();
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Reads a square texture back to a tightly-packed `Vec<u8>` of
    /// `resolution * resolution * bytes_per_pixel` bytes (one row after
    /// another, no padding), undoing wgpu's own row-alignment padding.
    fn read_raw(&self, texture: &wgpu::Texture, resolution: u32, bytes_per_pixel: u32) -> Vec<u8> {
        let unpadded_bytes_per_row = resolution * bytes_per_pixel;
        let padded_bytes_per_row = unpadded_bytes_per_row
            .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc_gpu readback buffer"),
            size: (padded_bytes_per_row * resolution) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mc_gpu readback encoder"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(resolution),
                },
            },
            wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("GPU poll failed");
        rx.recv()
            .expect("map_async callback never fired")
            .expect("failed to map readback buffer");

        let mapped = slice
            .get_mapped_range()
            .expect("buffer was just mapped successfully");
        let mut pixels = Vec::with_capacity((resolution * unpadded_bytes_per_row) as usize);
        for row in 0..resolution {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + unpadded_bytes_per_row as usize;
            pixels.extend_from_slice(&mapped[start..end]);
        }
        drop(mapped);
        buffer.unmap();
        pixels
    }
}

/// Encodes an `f32` to an IEEE 754 half-precision bit pattern — the inverse
/// of [`half_to_f32`] below, same algorithm as `mc_app::hdri`'s own
/// `f32_to_f16_bits` (truncates rather than rounds the mantissa, same as
/// that function). Duplicated rather than shared across the crate boundary
/// since `mc_app` depends on `mc_gpu`, not the other way around.
fn f32_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x007f_ffff;

    if exponent <= 0 {
        sign
    } else if exponent >= 0x1f {
        sign | 0x7c00
    } else {
        sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
    }
}

fn u16s_to_le_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Decodes an IEEE 754 half-precision float's raw bit pattern to `f32` — the
/// inverse of `mc_app::hdri`'s `f32_to_f16_bits`. Hand-rolled rather than
/// pulling in the `half` crate, same rationale as that function's own doc
/// comment: a one-off ~15-line conversion isn't worth a new dependency.
fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x3ff);

    if exponent == 0 && mantissa == 0 {
        return f32::from_bits(sign);
    }
    if exponent == 0x1f {
        // Inf/NaN.
        return f32::from_bits(sign | (0xff << 23) | (mantissa << 13));
    }
    if exponent == 0 {
        // Subnormal half: normalize by shifting the mantissa's leading 1 up
        // into bit 10, adjusting the resulting float exponent to match.
        let mut e: i32 = 0;
        let mut m = mantissa;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        e += 1;
        m &= 0x3ff;
        let float_exponent = ((e + 127 - 15) as u32) << 23;
        return f32::from_bits(sign | float_exponent | (m << 13));
    }

    let float_exponent = (exponent + (127 - 15)) << 23;
    f32::from_bits(sign | float_exponent | (mantissa << 13))
}

#[cfg(test)]
mod half_to_f32_tests {
    use super::half_to_f32;

    #[test]
    fn matches_known_bit_patterns() {
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0xBC00), -1.0);
        assert_eq!(half_to_f32(0x3800), 0.5);
        assert_eq!(half_to_f32(0x4000), 2.0);
    }

    /// Subnormal halves (exponent field 0, nonzero mantissa) are the one
    /// branch with real arithmetic in it — worth its own check rather than
    /// trusting the normal-case tests to exercise it too.
    #[test]
    fn decodes_the_smallest_subnormal_half() {
        // 0x0001 is the smallest positive half-precision value: 2^-24.
        let value = half_to_f32(0x0001);
        assert!(
            (value - 2f32.powi(-24)).abs() < 1e-10,
            "expected ~2^-24, got {value}"
        );
    }
}

/// Caches node output textures by a caller-chosen key (typically the node id),
/// so re-evaluating the same node reuses its texture instead of reallocating.
#[derive(Default)]
pub struct TexturePool {
    textures: HashMap<String, (wgpu::Texture, wgpu::TextureView)>,
}

impl TexturePool {
    /// Get the texture view for `key`, creating (or recreating, if the
    /// requested size/format changed) a square `size x size` storage+sampled
    /// texture as needed.
    pub fn get_or_create(
        &mut self,
        device: &wgpu::Device,
        key: &str,
        size: u32,
        format: wgpu::TextureFormat,
    ) -> &wgpu::TextureView {
        let needs_create = match self.textures.get(key) {
            Some((texture, _)) => texture.width() != size || texture.format() != format,
            None => true,
        };
        if needs_create {
            // Every pool texture gets a full mip chain allocated, even
            // though only the 6 PBR channel textures the 3D viewport reads
            // (`mc_app::mipgen`) ever actually have mips beyond 0 computed —
            // the extra levels just sit unused (and unsampled, since nothing
            // ever binds a full-mip-range view of an intermediate node's
            // texture) for any other node. Simpler than a second texture
            // path just for "the 6 that matter," at the cost of ~33% more
            // VRAM per texture, which isn't a concern at this project's
            // scale.
            let mip_level_count = size.max(1).ilog2() + 1;
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(key),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            // Pinned to mip 0 only, regardless of how many mips the texture
            // actually has: every existing caller (compute-shader storage
            // write targets, `textureLoad` reads, the 2D View/thumbnails)
            // expects exactly mip 0, and a storage-texture binding requires
            // a single-mip view anyway — wgpu rejects a multi-mip view
            // there. `mc_app::mipgen::full_mip_view` creates the
            // all-mips view the 3D viewport's sampler actually needs,
            // separately, on demand.
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                label: None,
                base_mip_level: 0,
                mip_level_count: Some(1),
                ..Default::default()
            });
            self.textures.insert(key.to_string(), (texture, view));
        }
        &self
            .textures
            .get(key)
            .expect("just inserted or already present")
            .1
    }

    /// The underlying texture for `key`, e.g. for a `copy_texture_to_buffer` readback.
    pub fn texture(&self, key: &str) -> Option<&wgpu::Texture> {
        self.textures.get(key).map(|(texture, _)| texture)
    }

    /// The view for `key`, e.g. to register with egui for display.
    pub fn view(&self, key: &str) -> Option<&wgpu::TextureView> {
        self.textures.get(key).map(|(_, view)| view)
    }
}

/// A tiny single-future executor.
///
/// wgpu's native `request_adapter`/`request_device` resolve immediately on
/// desktop (the `Future` wrapper exists only for API parity with the browser,
/// where it genuinely awaits). This avoids pulling in an async runtime crate
/// just to unwrap that; `Waker::noop()` is stable in std.
fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = Box::pin(fut);
    let waker = std::task::Waker::noop();
    let mut cx = TaskContext::from_waker(waker);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::yield_now(),
        }
    }
}
