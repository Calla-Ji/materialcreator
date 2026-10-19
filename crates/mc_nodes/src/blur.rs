// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Spatial / Blur.
//!
//! Display name: "Blur". Tooltip: "Gaussian blur — softens detail. Wrap-
//! addressed so it stays tileable."
//!
//! `dispatch_pass`'s own separable-Gaussian sampling loop (`blur.wgsl`) only
//! samples every texel — correct — up to [`FAST_PATH_MAX_RADIUS_PX`]; past
//! that, its own step grows beyond 1 texel to keep its tap count capped at
//! 32, which *skip-samples* instead of integrating every pixel. Verified
//! directly (not just suspected): blurring a single bright impulse produces
//! a perfectly smooth falloff at 10px/16px radius, but a literal
//! alternating zero/nonzero comb pattern at 33px — a real aliasing bug, not
//! a quality nitpick (the owner's own report, confirmed by this
//! investigation). `dispatch_pass` itself is kept completely unchanged
//! (same shader, same math) for any radius at or under that threshold — the
//! existing backward-compatibility guarantee for any graph already relying
//! on it. Past the threshold, [`dispatch`] instead runs a standard
//! downsample → blur → upsample pyramid (box-downsample `level` times so
//! the *effective* radius at that reduced resolution is back under the
//! same safe threshold, run the same unmodified `dispatch_pass` there, then
//! one direct bilinear upsample back to full resolution) — the owner's own
//! choice over just raising the tap cap, since this both fixes the
//! aliasing and keeps cost roughly constant regardless of radius (unlike a
//! tap count that grows linearly, or worse, with radius). A standard
//! real-time-graphics technique, not derived from any proprietary
//! implementation — the owner's own call that this doesn't need the
//! same independent-research-package treatment the Substance-specific
//! nodes elsewhere in this codebase got, since a Gaussian pyramid is
//! textbook, not proprietary.

use crate::shaders::full_source;
use crate::tiling::TilingMode;

const NODE_KIND: &str = "blur";
const DOWNSAMPLE_NODE_KIND: &str = "blur_downsample";
const UPSAMPLE_NODE_KIND: &str = "blur_upsample";

/// Where `dispatch_pass`'s own sampling loop is proven correct (see this
/// module's own doc comment) — at or under this radius, [`dispatch`] calls
/// it directly; past it, the large-radius pyramid takes over instead.
const FAST_PATH_MAX_RADIUS_PX: f32 = 16.0;

/// The pyramid never downsamples past this resolution, regardless of how
/// large `radius_px` is — an extremely large Radius value simply stops
/// gaining additional softness past some point, a practical ceiling every
/// pyramid-based blur has.
const MIN_PYRAMID_SIZE: u32 = 8;

pub struct BlurParams {
    /// In UV space (0..1), per CLAUDE.md §6.2 — converted to a pixel radius
    /// at dispatch time since the shader's sampling loop needs texel steps.
    pub radius: f32,
    pub tiling_mode: TilingMode,
}

impl Default for BlurParams {
    fn default() -> Self {
        Self {
            radius: 0.01,
            tiling_mode: TilingMode::default(),
        }
    }
}

impl BlurParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            radius: params
                .get("radius")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.radius),
            tiling_mode: params
                .get("tiling_mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => TilingMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.tiling_mode),
        }
    }
}

/// At or under [`FAST_PATH_MAX_RADIUS_PX`], runs the separable blur pass
/// twice (horizontal into a temp texture, then vertical from the temp
/// texture into `output_key`) directly — unchanged from before the
/// large-radius pyramid existed. Past that threshold, runs the pyramid
/// instead (see this module's own doc comment for why).
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    params: &BlurParams,
) {
    let radius_px = params.radius * resolution as f32;

    if radius_px <= FAST_PATH_MAX_RADIUS_PX {
        let temp_key = format!("{output_key}__blur_temp");
        dispatch_pass(
            gpu,
            pool,
            input_key,
            &temp_key,
            resolution,
            0,
            radius_px,
            params.tiling_mode.wrap_x(),
        );
        dispatch_pass(
            gpu,
            pool,
            &temp_key,
            output_key,
            resolution,
            1,
            radius_px,
            params.tiling_mode.wrap_y(),
        );
        return;
    }

    dispatch_pyramid(
        gpu,
        pool,
        input_key,
        output_key,
        resolution,
        radius_px,
        params.tiling_mode,
    );
}

/// The smallest number of 2x downsample steps so that `radius_px /
/// 2^level` is back under [`FAST_PATH_MAX_RADIUS_PX`] — the same threshold
/// where `dispatch_pass`'s own sampling loop is already proven correct —
/// capped so the downsampled resolution never drops below
/// [`MIN_PYRAMID_SIZE`].
fn pyramid_level(radius_px: f32, resolution: u32) -> u32 {
    let mut level = 0u32;
    let mut effective_radius_px = radius_px;
    while effective_radius_px > FAST_PATH_MAX_RADIUS_PX
        && (resolution >> (level + 1)) >= MIN_PYRAMID_SIZE
    {
        level += 1;
        effective_radius_px = radius_px / (1u32 << level) as f32;
    }
    level
}

/// Box-downsamples `input_key` `level` times (each step its own fresh
/// scratch pool key, not reusing `input_key`'s own pre-allocated mip
/// levels — mutating an *upstream* node's own cached texture risks
/// stepping on whatever else reads it), runs the existing, unmodified
/// `dispatch_pass` twice at that reduced resolution with a correspondingly
/// scaled-down radius (guaranteed at or under [`FAST_PATH_MAX_RADIUS_PX`]
/// by construction), then one direct bilinear upsample back to full
/// resolution into `output_key`.
fn dispatch_pyramid(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    radius_px: f32,
    tiling_mode: TilingMode,
) {
    let level = pyramid_level(radius_px, resolution);

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("blur_downsample_encoder"),
        });
    let mut current_key = input_key.to_string();
    let mut current_resolution = resolution;
    for l in 1..=level {
        current_resolution /= 2;
        let next_key = format!("{output_key}__blur_pyramid_{l}");
        record_downsample_pass(
            gpu,
            pool,
            &mut encoder,
            &current_key,
            &next_key,
            current_resolution,
            tiling_mode.wrap_x(),
            tiling_mode.wrap_y(),
        );
        current_key = next_key;
    }
    gpu.queue.submit(Some(encoder.finish()));

    let small_radius_px = radius_px / (1u32 << level) as f32;
    let blurred_temp_key = format!("{output_key}__blur_pyramid_blurred_temp");
    let blurred_key = format!("{output_key}__blur_pyramid_blurred");
    dispatch_pass(
        gpu,
        pool,
        &current_key,
        &blurred_temp_key,
        current_resolution,
        0,
        small_radius_px,
        tiling_mode.wrap_x(),
    );
    dispatch_pass(
        gpu,
        pool,
        &blurred_temp_key,
        &blurred_key,
        current_resolution,
        1,
        small_radius_px,
        tiling_mode.wrap_y(),
    );

    record_upsample_pass(
        gpu,
        pool,
        &blurred_key,
        output_key,
        current_resolution,
        resolution,
        tiling_mode.wrap_x(),
        tiling_mode.wrap_y(),
    );
}

#[allow(clippy::too_many_arguments)]
fn record_downsample_pass(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    encoder: &mut wgpu::CommandEncoder,
    input_key: &str,
    output_key: &str,
    dest_resolution: u32,
    wrap_x: bool,
    wrap_y: bool,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("blur: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("blur_downsample.wgsl");
    let pipeline = gpu.compute_pipeline(DOWNSAMPLE_NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&dest_resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&(wrap_x as u32).to_le_bytes());
    param_bytes[8..12].copy_from_slice(&(wrap_y as u32).to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("blur_downsample_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let output_view = pool.get_or_create(
        &gpu.device,
        output_key,
        dest_resolution,
        wgpu::TextureFormat::Rgba16Float,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blur_downsample_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&input_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("blur_downsample_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    let workgroups = dest_resolution.div_ceil(8);
    pass.dispatch_workgroups(workgroups, workgroups, 1);
}

#[allow(clippy::too_many_arguments)]
fn record_upsample_pass(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    src_resolution: u32,
    dest_resolution: u32,
    wrap_x: bool,
    wrap_y: bool,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("blur: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("blur_upsample.wgsl");
    let pipeline = gpu.compute_pipeline(UPSAMPLE_NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&dest_resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&src_resolution.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&(wrap_x as u32).to_le_bytes());
    param_bytes[12..16].copy_from_slice(&(wrap_y as u32).to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("blur_upsample_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let output_view = pool.get_or_create(
        &gpu.device,
        output_key,
        dest_resolution,
        wgpu::TextureFormat::Rgba16Float,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blur_upsample_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&input_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("blur_upsample_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("blur_upsample_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = dest_resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

#[allow(clippy::too_many_arguments)]
fn dispatch_pass(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    output_key: &str,
    resolution: u32,
    direction: u32,
    radius_px: f32,
    wrap: bool,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("blur: no texture at input key {input_key:?}"))
        .clone();

    let shader_source = full_source("blur.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&direction.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&radius_px.to_le_bytes());
    param_bytes[12..16].copy_from_slice(&(wrap as u32).to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("blur_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let output_view = pool.get_or_create(
        &gpu.device,
        output_key,
        resolution,
        wgpu::TextureFormat::Rgba16Float,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("blur_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&input_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("blur_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("blur_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

#[cfg(test)]
mod tests {
    use super::{dispatch, BlurParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, max_channel_diff, pixel, RESOLUTION};
    use crate::tiling::TilingMode;

    /// The largest left/right-edge mismatch anywhere along the seam —
    /// `assert_tiles_seamlessly` checks this stays within the interior
    /// baseline; these tests check the opposite, that it doesn't.
    fn left_right_seam_diff(pixels: &[u8]) -> i32 {
        (0..RESOLUTION)
            .map(|y| {
                max_channel_diff(
                    pixel(pixels, RESOLUTION, 0, y),
                    pixel(pixels, RESOLUTION, RESOLUTION - 1, y),
                )
            })
            .max()
            .unwrap()
    }

    fn top_bottom_seam_diff(pixels: &[u8]) -> i32 {
        (0..RESOLUTION)
            .map(|x| {
                max_channel_diff(
                    pixel(pixels, RESOLUTION, x, 0),
                    pixel(pixels, RESOLUTION, x, RESOLUTION - 1),
                )
            })
            .max()
            .unwrap()
    }

    fn render(params: &BlurParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        dispatch(&mut gpu, &mut pool, "input", "output", RESOLUTION, params);
        let texture = pool.texture("output").expect("blur texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    /// Wrap-addressed sampling is what keeps a blur of a tileable input
    /// tileable — without it, edge pixels would blend in black/garbage from
    /// outside the texture.
    #[test]
    fn preserves_tiling_of_its_input() {
        assert_tiles_seamlessly(
            &render(&BlurParams {
                radius: 0.03,
                tiling_mode: TilingMode::Both,
            }),
            RESOLUTION,
        );
    }

    #[test]
    fn is_deterministic() {
        let params = BlurParams {
            radius: 0.02,
            tiling_mode: TilingMode::Both,
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same input + params produced different pixels"
        );
    }

    #[test]
    fn increases_smoothness() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        let before = gpu.read_rgba8(pool.texture("input").unwrap(), RESOLUTION);

        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            RESOLUTION,
            &BlurParams {
                radius: 0.05,
                tiling_mode: TilingMode::Both,
            },
        );
        let after = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        let roughness = |pixels: &[u8]| -> i64 {
            let r = RESOLUTION as usize;
            let mut total = 0i64;
            for y in 0..r {
                for x in 0..r - 1 {
                    let a = pixels[(y * r + x) * 4] as i64;
                    let b = pixels[(y * r + x + 1) * 4] as i64;
                    total += (a - b).abs();
                }
            }
            total
        };
        assert!(
            roughness(&after) < roughness(&before),
            "blurred output should be smoother (less pixel-to-pixel variation) than the input"
        );
    }

    #[test]
    fn horizontal_tiling_off_introduces_a_left_right_seam() {
        let wrapped = render(&BlurParams {
            radius: 0.1,
            tiling_mode: TilingMode::Both,
        });
        let clamped = render(&BlurParams {
            radius: 0.1,
            tiling_mode: TilingMode::VerticalOnly,
        });
        assert!(
            left_right_seam_diff(&clamped) > left_right_seam_diff(&wrapped),
            "disabling horizontal tiling should make the left/right edges blur independently \
             instead of blending across the seam"
        );
    }

    #[test]
    fn vertical_tiling_off_introduces_a_top_bottom_seam() {
        let wrapped = render(&BlurParams {
            radius: 0.1,
            tiling_mode: TilingMode::Both,
        });
        let clamped = render(&BlurParams {
            radius: 0.1,
            tiling_mode: TilingMode::HorizontalOnly,
        });
        assert!(
            top_bottom_seam_diff(&clamped) > top_bottom_seam_diff(&wrapped),
            "disabling vertical tiling should make the top/bottom edges blur independently \
             instead of blending across the seam"
        );
    }

    /// `pyramid_level` is a pure function — cheap to check directly without
    /// a GPU. At or under the threshold, no downsampling at all; past it,
    /// just enough levels to bring the effective radius back under the
    /// threshold, capped by `MIN_PYRAMID_SIZE`.
    #[test]
    fn pyramid_level_picks_the_smallest_safe_level() {
        use super::{pyramid_level, FAST_PATH_MAX_RADIUS_PX, MIN_PYRAMID_SIZE};
        assert_eq!(pyramid_level(10.0, 1024), 0);
        assert_eq!(pyramid_level(FAST_PATH_MAX_RADIUS_PX, 1024), 0);
        // 33px needs one halving to land at 16.5px, still just over the
        // threshold, so a second halving (8.25px) is needed too.
        assert_eq!(pyramid_level(33.0, 1024), 2);
        for level in [1u32, 2, 3, 4] {
            let radius = FAST_PATH_MAX_RADIUS_PX * (1u32 << level) as f32;
            let picked = pyramid_level(radius, 1024);
            let effective = radius / (1u32 << picked) as f32;
            assert!(
                effective <= FAST_PATH_MAX_RADIUS_PX,
                "level {picked} for radius {radius} left effective radius {effective} over \
                 the threshold"
            );
        }
        // An absurdly large radius must still stop once the pyramid would
        // shrink below MIN_PYRAMID_SIZE, not loop forever or underflow.
        let level = pyramid_level(100_000.0, 1024);
        assert!((1024u32 >> level) >= MIN_PYRAMID_SIZE);
    }

    /// The regression test for the owner's own real bug report: blurring a
    /// single bright impulse must stay smooth at radii the old
    /// skip-sampling path combed into an alternating zero/nonzero pattern.
    /// Checked at two radii well past the pyramid threshold (33px, close to
    /// the owner's own working value, and 150px, to prove the pyramid keeps
    /// working rather than just patching one threshold) — every non-edge
    /// output pixel must differ from both its immediate neighbors by less
    /// than a true Gaussian's own worst-case local slope would ever allow
    /// near its peak, which a comb pattern (alternating near-zero and a
    /// real value) blows through by an order of magnitude.
    #[test]
    fn large_radius_impulse_response_is_smooth_not_combed() {
        let resolution = 512u32;
        let center = resolution / 2;

        for &radius_px in &[33.0f32, 150.0] {
            let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
            let mut pool = mc_gpu::TexturePool::default();
            let mut input = vec![[0.0f32, 0.0, 0.0, 1.0]; (resolution * resolution) as usize];
            input[(center * resolution + center) as usize] = [1.0, 1.0, 1.0, 1.0];
            gpu.write_rgba16f(&mut pool, "input", resolution, &input);

            dispatch(
                &mut gpu,
                &mut pool,
                "input",
                "output",
                resolution,
                &BlurParams {
                    radius: radius_px / resolution as f32,
                    tiling_mode: TilingMode::Both,
                },
            );
            let out = gpu.read_rgba16f(pool.texture("output").unwrap(), resolution);
            let at = |x: u32| out[(center * resolution + x) as usize * 4];

            let span = (radius_px as u32).min(resolution / 2 - 1);
            let mut max_value = 0.0f32;
            let mut max_neighbor_jump = 0.0f32;
            for dx in -(span as i32)..(span as i32) {
                let x = (center as i32 + dx) as u32;
                let v = at(x);
                let next = at(x + 1);
                max_value = max_value.max(v);
                max_neighbor_jump = max_neighbor_jump.max((next - v).abs());
            }
            assert!(
                max_value > 0.0,
                "radius_px={radius_px}: impulse should produce some visible response"
            );
            // A real comb pattern alternates between ~0 and the local
            // envelope value every other pixel — its jump is close to
            // `max_value` itself. A genuinely smooth falloff's largest
            // single-step jump is a small fraction of its own peak.
            assert!(
                max_neighbor_jump < max_value * 0.5,
                "radius_px={radius_px}: largest neighbor-to-neighbor jump ({max_neighbor_jump}) \
                 is too close to the peak response ({max_value}) — looks like a comb/aliasing \
                 pattern, not a smooth blur"
            );
        }
    }

    #[test]
    fn large_radius_preserves_tiling() {
        let resolution = 256u32;
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            resolution,
            &PerlinParams::default(),
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "output",
            resolution,
            &BlurParams {
                radius: 40.0 / resolution as f32,
                tiling_mode: TilingMode::Both,
            },
        );
        let out = gpu.read_rgba8(pool.texture("output").unwrap(), resolution);
        assert_tiles_seamlessly(&out, resolution);
    }

    #[test]
    fn large_radius_is_deterministic() {
        let resolution = 256u32;
        let params = BlurParams {
            radius: 40.0 / resolution as f32,
            tiling_mode: TilingMode::Both,
        };
        let render_at = |params: &BlurParams| {
            let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
            let mut pool = mc_gpu::TexturePool::default();
            perlin_dispatch(
                &mut gpu,
                &mut pool,
                "input",
                resolution,
                &PerlinParams::default(),
            );
            dispatch(&mut gpu, &mut pool, "input", "output", resolution, params);
            gpu.read_rgba8(pool.texture("output").unwrap(), resolution)
        };
        assert_eq!(
            render_at(&params),
            render_at(&params),
            "same input + params produced different pixels"
        );
    }
}
