// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Spatial / Flow Blur.
//!
//! Display name: "Flow Blur". Tooltip: "Blur along a control map's local
//! gradient, strength scaled by its steepness — key for organic, eroded
//! looks."

use crate::shaders::full_source;
use crate::tiling::TilingMode;

const NODE_KIND: &str = "slope_blur";

/// Matches Substance's own Slope Blur "Mode": `Blur` is a plain weighted
/// directional average (soft, fades contrast where the kernel crosses a
/// sharp feature); `Min`/`Max` are a directional grayscale erosion/dilation
/// instead — no weighting, just the darkest/lightest sample in the kernel —
/// which is what actually gives the crisp, branching "eroded crack" look
/// artists reach for on rock/terrain, since it preserves the original
/// value rather than blending it away. `Chaos` samples a single
/// pseudo-randomly placed point within the same reach instead of reducing
/// over the whole kernel, for a scattered/torn look.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlopeBlurMode {
    Blur,
    Min,
    Max,
    Chaos,
}

impl SlopeBlurMode {
    pub const NAMES: [&'static str; 4] = ["Blur", "Min", "Max", "Chaos"];

    fn as_u32(self) -> u32 {
        match self {
            SlopeBlurMode::Blur => 0,
            SlopeBlurMode::Min => 1,
            SlopeBlurMode::Max => 2,
            SlopeBlurMode::Chaos => 3,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Blur" => Some(SlopeBlurMode::Blur),
            "Min" => Some(SlopeBlurMode::Min),
            "Max" => Some(SlopeBlurMode::Max),
            "Chaos" => Some(SlopeBlurMode::Chaos),
            _ => None,
        }
    }
}

pub struct SlopeBlurParams {
    /// In UV space (0..1), like `BlurParams::radius` — the blur radius
    /// reached at the control map's steepest point.
    pub radius: f32,
    /// Multiplies the control map's own per-texel gradient before it drives
    /// the blur amount — see `slope_blur.wgsl`'s own doc comment for why
    /// this has to be a real param rather than a built-in constant.
    pub strength: f32,
    pub mode: SlopeBlurMode,
    pub tiling_mode: TilingMode,
}

impl Default for SlopeBlurParams {
    fn default() -> Self {
        Self {
            radius: 0.03,
            strength: 150.0,
            mode: SlopeBlurMode::Blur,
            tiling_mode: TilingMode::default(),
        }
    }
}

impl SlopeBlurParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            radius: params
                .get("radius")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.radius),
            strength: params
                .get("strength")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.strength),
            mode: params
                .get("mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => SlopeBlurMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.mode),
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

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    input_key: &str,
    control_key: &str,
    output_key: &str,
    resolution: u32,
    params: &SlopeBlurParams,
) {
    let input_view = pool
        .view(input_key)
        .unwrap_or_else(|| panic!("slope_blur: no texture at input key {input_key:?}"))
        .clone();
    let control_view = pool
        .view(control_key)
        .unwrap_or_else(|| panic!("slope_blur: no texture at control key {control_key:?}"))
        .clone();

    let shader_source = full_source("slope_blur.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let radius_px = params.radius * resolution as f32;
    let mut param_bytes = [0u8; 32];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&radius_px.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&params.strength.to_le_bytes());
    param_bytes[12..16].copy_from_slice(&params.tiling_mode.as_u32().to_le_bytes());
    param_bytes[16..20].copy_from_slice(&params.mode.as_u32().to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("slope_blur_params"),
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
        label: Some("slope_blur_bind_group"),
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
                resource: wgpu::BindingResource::TextureView(&control_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("slope_blur_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("slope_blur_pass"),
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
    use super::{dispatch, SlopeBlurMode, SlopeBlurParams};
    use crate::perlin::{dispatch as perlin_dispatch, PerlinParams};
    use crate::test_support::{assert_tiles_seamlessly, max_channel_diff, pixel, RESOLUTION};
    use crate::tiling::TilingMode;
    use crate::uniform_color::{dispatch as uniform_color_dispatch, UniformColorParams};

    fn render(params: &SlopeBlurParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams {
                seed: 1,
                ..PerlinParams::default()
            },
        );
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "control",
            RESOLUTION,
            &PerlinParams {
                seed: 2,
                ..PerlinParams::default()
            },
        );
        dispatch(
            &mut gpu, &mut pool, "input", "control", "output", RESOLUTION, params,
        );
        let texture = pool.texture("output").expect("slope_blur texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn preserves_tiling_of_its_inputs() {
        assert_tiles_seamlessly(
            &render(&SlopeBlurParams {
                radius: 0.05,
                ..SlopeBlurParams::default()
            }),
            RESOLUTION,
        );
    }

    #[test]
    fn is_deterministic() {
        let params = SlopeBlurParams::default();
        assert_eq!(render(&params), render(&params));
    }

    /// A perfectly flat control map has zero slope everywhere, so the input
    /// should pass through completely unblurred.
    #[test]
    fn flat_control_is_a_no_op() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            RESOLUTION,
            &PerlinParams::default(),
        );
        uniform_color_dispatch(
            &mut gpu,
            &mut pool,
            "control",
            RESOLUTION,
            &UniformColorParams {
                color: [0.5, 0.5, 0.5, 1.0],
            },
        );
        let before = gpu.read_rgba8(pool.texture("input").unwrap(), RESOLUTION);

        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "control",
            "output",
            RESOLUTION,
            &SlopeBlurParams {
                radius: 0.05,
                ..SlopeBlurParams::default()
            },
        );
        let after = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        assert_eq!(
            before, after,
            "a flat control map must not blur the input at all"
        );
    }

    /// Regression test for the node being all but inert in practice: a
    /// height/noise-style control map's own per-texel gradient is tiny (a
    /// smooth hill a few hundred texels wide differs from its neighbor by a
    /// few thousandths), so the default `strength` has to actually amplify
    /// it into a visible effect, not leave it near-invisible the way the old
    /// fixed `*10.0` scaling did (measured: blurring sharp Voronoi edges
    /// along a Perlin control map's gradient at a realistic 1024px
    /// resolution moved the output by a max of 1 of 255 levels before this
    /// fix, vs. 238 after). Uses Voronoi Edges, not another smooth noise
    /// field, as the thing being blurred — two mutually self-similar smooth
    /// fBm fields barely change when averaged with their own local
    /// neighborhood regardless of blur strength, which isn't this node's
    /// fault; sharp content is what actually shows the streaking "flow"
    /// look Substance's own Slope Blur is known for.
    #[test]
    fn default_params_produce_a_visible_blur_on_a_realistic_control_map() {
        use crate::voronoi::{dispatch as voronoi_dispatch, VoronoiMode, VoronoiParams};
        let res = 1024;
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        voronoi_dispatch(
            &mut gpu,
            &mut pool,
            "input",
            res,
            &VoronoiParams {
                mode: VoronoiMode::Edges,
                ..VoronoiParams::default()
            },
        );
        perlin_dispatch(
            &mut gpu,
            &mut pool,
            "control",
            res,
            &PerlinParams {
                seed: 2,
                ..PerlinParams::default()
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "input",
            "control",
            "output",
            res,
            &SlopeBlurParams {
                radius: 0.1,
                ..SlopeBlurParams::default()
            },
        );
        let before = gpu.read_rgba8(pool.texture("input").unwrap(), res);
        let after = gpu.read_rgba8(pool.texture("output").unwrap(), res);
        let max_diff = before
            .iter()
            .zip(after.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        assert!(
            max_diff > 100,
            "default strength should produce a clearly visible blur on a \
             realistic control map's steepest areas, got max_diff={max_diff}"
        );
    }

    #[test]
    fn tiling_off_introduces_a_seam() {
        // A low, pre-this-fix-era `strength` on purpose: at this test
        // module's own tiny `RESOLUTION` (64px), the new, much higher
        // production default saturates the slope's own `clamp(..., 0.0,
        // 1.0)` ceiling almost everywhere (a 64px-wide test texture has a
        // much steeper per-texel gradient for the same noise pattern than a
        // realistic 1024px+ render does), which washes out the very
        // wrap-vs-clamp distinction this test exists to check. That's a
        // property of this test's own tiny resolution, not a regression —
        // see `default_params_produce_a_visible_blur_on_a_realistic_control_
        // map` for the real strength/resolution this value is tuned for.
        let wrapped = render(&SlopeBlurParams {
            radius: 0.1,
            strength: 10.0,
            mode: SlopeBlurMode::Blur,
            tiling_mode: TilingMode::Both,
        });
        let clamped = render(&SlopeBlurParams {
            radius: 0.1,
            strength: 10.0,
            mode: SlopeBlurMode::Blur,
            tiling_mode: TilingMode::Off,
        });
        let seam = |pixels: &[u8]| -> i32 {
            (0..RESOLUTION)
                .map(|y| {
                    max_channel_diff(
                        pixel(pixels, RESOLUTION, 0, y),
                        pixel(pixels, RESOLUTION, RESOLUTION - 1, y),
                    )
                })
                .max()
                .unwrap()
        };
        assert!(
            seam(&clamped) > seam(&wrapped),
            "disabling tiling should make the left/right edges blur independently"
        );
    }

    /// Max is a directional dilation (no weighting, just the lightest
    /// sample in the footprint, which always includes the pixel's own
    /// value) — it must never make any pixel *darker* than it already was.
    #[test]
    fn max_mode_never_darkens_a_pixel_below_its_own_value() {
        let before = render(&SlopeBlurParams {
            radius: 0.0,
            ..SlopeBlurParams::default()
        });
        let after = render(&SlopeBlurParams {
            radius: 0.1,
            mode: SlopeBlurMode::Max,
            ..SlopeBlurParams::default()
        });
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(a >= b, "Max mode darkened a pixel: before={b} after={a}");
        }
    }

    /// Min is a directional erosion — the mirror image of Max, so it must
    /// never make any pixel *lighter* than it already was.
    #[test]
    fn min_mode_never_lightens_a_pixel_above_its_own_value() {
        let before = render(&SlopeBlurParams {
            radius: 0.0,
            ..SlopeBlurParams::default()
        });
        let after = render(&SlopeBlurParams {
            radius: 0.1,
            mode: SlopeBlurMode::Min,
            ..SlopeBlurParams::default()
        });
        for (b, a) in before.iter().zip(after.iter()) {
            assert!(a <= b, "Min mode lightened a pixel: before={b} after={a}");
        }
    }

    #[test]
    fn min_and_max_mode_preserve_tiling() {
        for mode in [SlopeBlurMode::Min, SlopeBlurMode::Max] {
            assert_tiles_seamlessly(
                &render(&SlopeBlurParams {
                    radius: 0.05,
                    mode,
                    ..SlopeBlurParams::default()
                }),
                RESOLUTION,
            );
        }
    }

    #[test]
    fn chaos_mode_is_deterministic_and_preserves_tiling() {
        let params = SlopeBlurParams {
            mode: SlopeBlurMode::Chaos,
            ..SlopeBlurParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same pixel coordinates must hash to the same random offset every run"
        );
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }
}
