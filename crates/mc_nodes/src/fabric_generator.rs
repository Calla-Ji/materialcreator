// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Fabric Generator.
//!
//! Display name: "Fabric Generator". Tooltip: "Woven and knit base shapes —
//! Plain Weave, Twill, Herringbone, Rib Knit, Stockinette — for building
//! fabric materials from scratch."
//!
//! Each `WeaveType` only uses a subset of the shared params (see each
//! variant's own doc comment below) — the same "every param is always
//! shown, some are inert for the current mode" convention `fractal_sum.rs`
//! and `voronoi.rs` already use for their own mode switches.

use crate::shaders::full_source;

const NODE_KIND: &str = "fabric_generator";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeaveType {
    /// Two perpendicular thread sets alternating over/under in a
    /// checkerboard — uses `thread_group_size` (1 = plain weave, 2+ groups
    /// threads into blocks, the classic Basket Weave look).
    Plain,
    /// Like Plain Weave, but the over/under crossing steps diagonally
    /// instead of alternating — uses `twill_count` for the diagonal's
    /// repeat period (the denim/chino look).
    Twill,
    /// Twill with the diagonal's direction reversed every `twill_count`
    /// columns — the chevron/zigzag tweed look. Shares `twill_count` with
    /// Twill: it both sets the diagonal's own repeat period and, here, the
    /// width of each reversed band.
    Herringbone,
    /// Alternating raised/recessed vertical columns — uses `rib_width` for
    /// how many columns wide each rib is.
    RibKnit,
    /// Interlocking "V"-shaped loops in a staggered brick layout — the
    /// classic T-shirt knit. Uses none of the weave-specific params above.
    Stockinette,
}

impl WeaveType {
    pub const NAMES: [&'static str; 5] = [
        "Plain Weave",
        "Twill",
        "Herringbone",
        "Rib Knit",
        "Stockinette",
    ];

    fn as_u32(self) -> u32 {
        match self {
            WeaveType::Plain => 0,
            WeaveType::Twill => 1,
            WeaveType::Herringbone => 2,
            WeaveType::RibKnit => 3,
            WeaveType::Stockinette => 4,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Plain Weave" => Some(WeaveType::Plain),
            "Twill" => Some(WeaveType::Twill),
            "Herringbone" => Some(WeaveType::Herringbone),
            "Rib Knit" => Some(WeaveType::RibKnit),
            "Stockinette" => Some(WeaveType::Stockinette),
            _ => None,
        }
    }
}

pub struct FabricGeneratorParams {
    pub weave_type: WeaveType,
    pub rows: u32,
    pub columns: u32,
    pub thread_width: f32,
    pub thread_group_size: u32,
    pub twill_count: u32,
    pub rib_width: u32,
    pub size_jitter: f32,
    pub luminance_jitter: f32,
    pub seed: u32,
}

impl Default for FabricGeneratorParams {
    fn default() -> Self {
        Self {
            weave_type: WeaveType::Plain,
            // Even, and a multiple of every weave-specific period below
            // (twill_count=4, rib_width*2=2) — tiles seamlessly out of the
            // box in every weave type; see the shader's own doc comment.
            rows: 24,
            columns: 24,
            thread_width: 0.85,
            thread_group_size: 1,
            twill_count: 4,
            rib_width: 1,
            size_jitter: 0.06,
            luminance_jitter: 0.1,
            seed: 0,
        }
    }
}

impl FabricGeneratorParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        let int = |name: &str, default: u32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(default)
        };
        Self {
            weave_type: params
                .get("weave_type")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => WeaveType::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.weave_type),
            rows: int("rows", defaults.rows),
            columns: int("columns", defaults.columns),
            thread_width: float("thread_width", defaults.thread_width),
            thread_group_size: int("thread_group_size", defaults.thread_group_size),
            twill_count: int("twill_count", defaults.twill_count),
            rib_width: int("rib_width", defaults.rib_width),
            size_jitter: float("size_jitter", defaults.size_jitter),
            luminance_jitter: float("luminance_jitter", defaults.luminance_jitter),
            seed: params
                .get("seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.seed),
        }
    }
}

fn pack_params(resolution: u32, params: &FabricGeneratorParams) -> [u8; 48] {
    let mut bytes = [0u8; 48];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.rows.to_le_bytes());
    bytes[8..12].copy_from_slice(&params.columns.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.seed.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.weave_type.as_u32().to_le_bytes());
    bytes[20..24].copy_from_slice(&params.thread_width.to_le_bytes());
    // bytes[24..28] was `thread_roundness`, retired now that every Thread
    // ply is a genuine round tube (no longer an adjustable flat-vs-round
    // cross-section) — left as explicit padding rather than reflowing
    // every field after it.
    bytes[28..32].copy_from_slice(&params.thread_group_size.to_le_bytes());
    bytes[32..36].copy_from_slice(&params.twill_count.to_le_bytes());
    bytes[36..40].copy_from_slice(&params.rib_width.to_le_bytes());
    bytes[40..44].copy_from_slice(&params.size_jitter.to_le_bytes());
    bytes[44..48].copy_from_slice(&params.luminance_jitter.to_le_bytes());
    bytes
}

/// Where `dispatch` writes its two outputs — mirrors
/// [`crate::thread_shape::ThreadOutputKeys`].
pub struct FabricGeneratorOutputKeys<'a> {
    pub height: &'a str,
    pub ply_id: &'a str,
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    fiber_key: Option<&str>,
    fiber_ply_id_key: Option<&str>,
    outputs: FabricGeneratorOutputKeys,
    resolution: u32,
    params: &FabricGeneratorParams,
) {
    // No Thread wired in: render one internally from this node's own
    // thread_width, so Fabric Generator still works standalone with no
    // setup — same "optional input defaults sensibly" convention as
    // Blend's mask or Scanline Jitter's own mask above. A single straight
    // (untwisted, unplied) round strand filling one row — `threads: 1`
    // makes Thread's own `thickness` mean "fraction of the whole tile
    // height", matching what this fallback always meant. This internal
    // Thread already generates a real Ply ID too, so the fallback fiber
    // path below carries it straight through instead of discarding it.
    let default_fiber_height_key = format!("{}__default_fiber_height", outputs.height);
    let default_fiber_ply_id_key = format!("{}__default_fiber_ply_id", outputs.height);
    let (fiber_key, fallback_ply_id_key) = match fiber_key {
        Some(key) => (key, None),
        None => {
            crate::thread_shape::dispatch(
                gpu,
                pool,
                crate::thread_shape::ThreadOutputKeys {
                    height: &default_fiber_height_key,
                    ply_id: &default_fiber_ply_id_key,
                    fiber: &format!("{}__default_fiber_fiber", outputs.height),
                    mask: &format!("{}__default_fiber_mask", outputs.height),
                },
                resolution,
                &crate::thread_shape::ThreadShapeParams {
                    threads: 1,
                    thickness: params.thread_width,
                    plies: 1,
                    fiber_mode: crate::thread_shape::FiberMode::None,
                    ..crate::thread_shape::ThreadShapeParams::default()
                },
            );
            (
                default_fiber_height_key.as_str(),
                Some(default_fiber_ply_id_key.as_str()),
            )
        }
    };
    let fiber_view = pool
        .view(fiber_key)
        .unwrap_or_else(|| panic!("fabric_generator: no texture at fiber key {fiber_key:?}"))
        .clone();

    // A real Thread is wired in but its Ply ID output wasn't separately
    // connected: fall back to a flat "single ply" value (every strand reads
    // as ply 1 of 1), the same sensible-default convention as every other
    // optional input in this engine.
    let single_ply_key = format!("{}__single_ply_fallback", outputs.height);
    let fiber_ply_id_key = match fiber_ply_id_key.or(fallback_ply_id_key) {
        Some(key) => key.to_string(),
        None => {
            crate::uniform_color::dispatch(
                gpu,
                pool,
                &single_ply_key,
                resolution,
                &crate::uniform_color::UniformColorParams {
                    color: [1.0, 1.0, 1.0, 1.0],
                },
            );
            single_ply_key.clone()
        }
    };
    let fiber_ply_id_view = pool
        .view(&fiber_ply_id_key)
        .unwrap_or_else(|| {
            panic!("fabric_generator: no texture at fiber ply id key {fiber_ply_id_key:?}")
        })
        .clone();

    let shader_source = full_source("fabric_generator.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fabric_generator_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let view = pool
        .get_or_create(
            &gpu.device,
            outputs.height,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let ply_id_view = pool
        .get_or_create(
            &gpu.device,
            outputs.ply_id,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fabric_generator_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&fiber_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&fiber_ply_id_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&ply_id_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fabric_generator_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fabric_generator_pass"),
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
    use super::{dispatch, FabricGeneratorOutputKeys, FabricGeneratorParams, WeaveType};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    fn render(params: &FabricGeneratorParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(
            &mut gpu,
            &mut pool,
            None,
            None,
            FabricGeneratorOutputKeys {
                height: "test",
                ply_id: "test_ply_id",
            },
            RESOLUTION,
            params,
        );
        let texture = pool.texture("test").expect("fabric_generator texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    fn render_ply_id(params: &FabricGeneratorParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(
            &mut gpu,
            &mut pool,
            None,
            None,
            FabricGeneratorOutputKeys {
                height: "test",
                ply_id: "test_ply_id",
            },
            RESOLUTION,
            params,
        );
        let texture = pool
            .texture("test_ply_id")
            .expect("fabric_generator ply_id texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    #[test]
    fn tiles_seamlessly_plain_weave() {
        assert_tiles_seamlessly(&render(&FabricGeneratorParams::default()), RESOLUTION);
    }

    #[test]
    fn tiles_seamlessly_basket_weave() {
        let params = FabricGeneratorParams {
            thread_group_size: 3,
            rows: 24,
            columns: 24,
            ..FabricGeneratorParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    #[test]
    fn tiles_seamlessly_twill() {
        let params = FabricGeneratorParams {
            weave_type: WeaveType::Twill,
            rows: 24,
            columns: 24,
            twill_count: 4,
            ..FabricGeneratorParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    /// Herringbone needs columns a multiple of `twill_count * 2` (an even
    /// number of reversed bands), stricter than Twill's own `twill_count` —
    /// see the shader's own doc comment. 24 is a multiple of 4 * 2 = 8.
    #[test]
    fn tiles_seamlessly_herringbone() {
        let params = FabricGeneratorParams {
            weave_type: WeaveType::Herringbone,
            rows: 24,
            columns: 24,
            twill_count: 4,
            ..FabricGeneratorParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    /// Herringbone is Twill with the diagonal periodically reversed — if the
    /// reversal never actually fired (e.g. a sign error always flipping, or
    /// never flipping), it would silently render identically to Twill.
    #[test]
    fn herringbone_differs_from_twill() {
        let shared = FabricGeneratorParams {
            rows: 24,
            columns: 24,
            twill_count: 4,
            ..FabricGeneratorParams::default()
        };
        let twill = render(&FabricGeneratorParams {
            weave_type: WeaveType::Twill,
            ..shared
        });
        let herringbone = render(&FabricGeneratorParams {
            weave_type: WeaveType::Herringbone,
            ..shared
        });
        assert_ne!(twill, herringbone);
    }

    #[test]
    fn tiles_seamlessly_rib_knit() {
        let params = FabricGeneratorParams {
            weave_type: WeaveType::RibKnit,
            columns: 24,
            rib_width: 2,
            ..FabricGeneratorParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    #[test]
    fn tiles_seamlessly_stockinette() {
        let params = FabricGeneratorParams {
            weave_type: WeaveType::Stockinette,
            rows: 24,
            columns: 24,
            ..FabricGeneratorParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let params = FabricGeneratorParams {
            seed: 7,
            ..FabricGeneratorParams::default()
        };
        assert_eq!(render(&params), render(&params));
    }

    #[test]
    fn seed_changes_output() {
        let a = render(&FabricGeneratorParams {
            seed: 1,
            ..FabricGeneratorParams::default()
        });
        let b = render(&FabricGeneratorParams {
            seed: 2,
            ..FabricGeneratorParams::default()
        });
        assert_ne!(a, b, "different seeds produced identical pixels");
    }

    #[test]
    fn every_weave_type_name_round_trips() {
        for name in WeaveType::NAMES {
            assert!(
                WeaveType::from_name(name).is_some(),
                "{name:?} should parse"
            );
        }
        assert!(WeaveType::from_name("Nonsense").is_none());
    }

    /// Each weave type should actually look different from the others, not
    /// silently fall through to the same code path.
    #[test]
    fn every_weave_type_produces_distinct_output() {
        let mut outputs = Vec::new();
        for weave_type in [
            WeaveType::Plain,
            WeaveType::Twill,
            WeaveType::Herringbone,
            WeaveType::RibKnit,
            WeaveType::Stockinette,
        ] {
            outputs.push(render(&FabricGeneratorParams {
                weave_type,
                ..FabricGeneratorParams::default()
            }));
        }
        for i in 0..outputs.len() {
            for j in (i + 1)..outputs.len() {
                assert_ne!(
                    outputs[i],
                    outputs[j],
                    "{:?} and {:?} produced identical output",
                    WeaveType::NAMES[i],
                    WeaveType::NAMES[j]
                );
            }
        }
    }

    /// Plain Weave vs. the same params with `thread_group_size` > 1 (Basket
    /// Weave) should differ — otherwise the group-size param would be a
    /// silent no-op.
    #[test]
    fn thread_group_size_changes_plain_weave_output() {
        let plain = render(&FabricGeneratorParams {
            rows: 24,
            columns: 24,
            ..FabricGeneratorParams::default()
        });
        let basket = render(&FabricGeneratorParams {
            rows: 24,
            columns: 24,
            thread_group_size: 3,
            ..FabricGeneratorParams::default()
        });
        assert_ne!(plain, basket);
    }

    /// A connected Fiber input should actually change the output, not just
    /// be plumbing that never gets read — distinct from the fallback the
    /// node renders internally when nothing's wired in.
    #[test]
    fn a_connected_fiber_changes_the_output() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let params = FabricGeneratorParams::default();

        dispatch(
            &mut gpu,
            &mut pool,
            None,
            None,
            FabricGeneratorOutputKeys {
                height: "fallback",
                ply_id: "fallback_ply_id",
            },
            RESOLUTION,
            &params,
        );
        let fallback = gpu.read_rgba8(pool.texture("fallback").unwrap(), RESOLUTION);

        crate::thread_shape::dispatch(
            &mut gpu,
            &mut pool,
            crate::thread_shape::ThreadOutputKeys {
                height: "thin_fiber_height",
                ply_id: "thin_fiber_ply_id",
                fiber: "thin_fiber_fiber",
                mask: "thin_fiber_mask",
            },
            RESOLUTION,
            &crate::thread_shape::ThreadShapeParams {
                threads: 1,
                plies: 1,
                thickness: 0.1,
                fiber_mode: crate::thread_shape::FiberMode::None,
                ..crate::thread_shape::ThreadShapeParams::default()
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            Some("thin_fiber_height"),
            None,
            FabricGeneratorOutputKeys {
                height: "with_fiber",
                ply_id: "with_fiber_ply_id",
            },
            RESOLUTION,
            &params,
        );
        let with_fiber = gpu.read_rgba8(pool.texture("with_fiber").unwrap(), RESOLUTION);

        assert_ne!(
            fallback, with_fiber,
            "wiring in a very different fiber shape should change Fabric Generator's output"
        );
    }

    /// A Thread's own Twist Angle should actually show up in the woven
    /// result, not just at the Thread node's own preview — end-to-end proof
    /// the per-cell along-length sampling (`fiber_height`'s `along`
    /// parameter) is wired up correctly. There's no "untwisted" state in
    /// the new tube-geometry model (`turns_from_angle` always returns at
    /// least 1 full turn, same as real yarn always having *some* twist), so
    /// this compares a loose vs. a tight twist angle instead.
    #[test]
    fn a_tighter_twist_angle_changes_the_output() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let params = FabricGeneratorParams {
            rows: 24,
            columns: 24,
            size_jitter: 0.0,
            ..FabricGeneratorParams::default()
        };

        crate::thread_shape::dispatch(
            &mut gpu,
            &mut pool,
            crate::thread_shape::ThreadOutputKeys {
                height: "loose_height",
                ply_id: "loose_ply_id",
                fiber: "loose_fiber",
                mask: "loose_mask",
            },
            RESOLUTION,
            &crate::thread_shape::ThreadShapeParams {
                plies: 3,
                twist_angle: 10.0,
                thickness: 0.7,
                fiber_mode: crate::thread_shape::FiberMode::None,
                ..crate::thread_shape::ThreadShapeParams::default()
            },
        );
        crate::thread_shape::dispatch(
            &mut gpu,
            &mut pool,
            crate::thread_shape::ThreadOutputKeys {
                height: "tight_height",
                ply_id: "tight_ply_id",
                fiber: "tight_fiber",
                mask: "tight_mask",
            },
            RESOLUTION,
            &crate::thread_shape::ThreadShapeParams {
                plies: 3,
                twist_angle: 55.0,
                thickness: 0.7,
                fiber_mode: crate::thread_shape::FiberMode::None,
                ..crate::thread_shape::ThreadShapeParams::default()
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            Some("loose_height"),
            None,
            FabricGeneratorOutputKeys {
                height: "with_straight",
                ply_id: "with_straight_ply_id",
            },
            RESOLUTION,
            &params,
        );
        dispatch(
            &mut gpu,
            &mut pool,
            Some("tight_height"),
            None,
            FabricGeneratorOutputKeys {
                height: "with_twisted",
                ply_id: "with_twisted_ply_id",
            },
            RESOLUTION,
            &params,
        );
        let with_straight = gpu.read_rgba8(pool.texture("with_straight").unwrap(), RESOLUTION);
        let with_twisted = gpu.read_rgba8(pool.texture("with_twisted").unwrap(), RESOLUTION);

        assert_ne!(
            with_straight, with_twisted,
            "a twisted 3-ply fiber should weave differently than the same fiber untwisted"
        );
    }

    #[test]
    fn ply_id_tiles_seamlessly() {
        assert_tiles_seamlessly(
            &render_ply_id(&FabricGeneratorParams::default()),
            RESOLUTION,
        );
    }

    #[test]
    fn ply_id_is_deterministic() {
        let params = FabricGeneratorParams {
            seed: 7,
            ..FabricGeneratorParams::default()
        };
        assert_eq!(render_ply_id(&params), render_ply_id(&params));
    }

    /// Stockinette never samples a fiber cross-section at all — its Ply ID
    /// output should read as flat 0, not leftover/undefined data.
    #[test]
    fn stockinette_ply_id_is_flat_zero() {
        let params = FabricGeneratorParams {
            weave_type: WeaveType::Stockinette,
            rows: 24,
            columns: 24,
            ..FabricGeneratorParams::default()
        };
        let pixels = render_ply_id(&params);
        assert!(
            pixels.iter().step_by(4).all(|&r| r == 0),
            "Stockinette's Ply ID output should be uniformly zero"
        );
    }

    /// Wiring a real multi-ply Thread's Ply ID output in should actually
    /// change Fabric Generator's own Ply ID output, not just be plumbing
    /// that's silently discarded (the bug this whole feature started from —
    /// the internal fallback Thread already generated a real Ply ID that
    /// used to be thrown away).
    #[test]
    fn a_connected_fiber_ply_id_changes_the_ply_id_output() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let params = FabricGeneratorParams {
            rows: 24,
            columns: 24,
            size_jitter: 0.0,
            ..FabricGeneratorParams::default()
        };

        crate::thread_shape::dispatch(
            &mut gpu,
            &mut pool,
            crate::thread_shape::ThreadOutputKeys {
                height: "yarn_height",
                ply_id: "yarn_ply_id",
                fiber: "yarn_fiber",
                mask: "yarn_mask",
            },
            RESOLUTION,
            &crate::thread_shape::ThreadShapeParams {
                plies: 3,
                thickness: 0.7,
                fiber_mode: crate::thread_shape::FiberMode::None,
                ..crate::thread_shape::ThreadShapeParams::default()
            },
        );

        // Fiber connected, Ply ID left unconnected: flat single-ply fallback.
        dispatch(
            &mut gpu,
            &mut pool,
            Some("yarn_height"),
            None,
            FabricGeneratorOutputKeys {
                height: "no_ply_id_height",
                ply_id: "no_ply_id_ply_id",
            },
            RESOLUTION,
            &params,
        );
        let flat = gpu.read_rgba8(pool.texture("no_ply_id_ply_id").unwrap(), RESOLUTION);

        // Both Fiber and its matching Ply ID connected: real per-ply data.
        dispatch(
            &mut gpu,
            &mut pool,
            Some("yarn_height"),
            Some("yarn_ply_id"),
            FabricGeneratorOutputKeys {
                height: "with_ply_id_height",
                ply_id: "with_ply_id_ply_id",
            },
            RESOLUTION,
            &params,
        );
        let real = gpu.read_rgba8(pool.texture("with_ply_id_ply_id").unwrap(), RESOLUTION);

        assert_ne!(
            flat, real,
            "connecting a real multi-ply Thread's Ply ID should change Fabric Generator's own \
             Ply ID output, not be silently discarded"
        );
    }
}
