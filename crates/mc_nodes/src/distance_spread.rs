// SPDX-License-Identifier: MIT OR Apache-2.0
//! Filters / Spatial / Distance.
//!
//! Display name: "Distance". Tooltip: "Spreads Source's value outward from
//! Mask, fading to zero past Distance — dirt pooling near seams, decay
//! radiating from damage, that kind of effect."
//!
//! The owner's own request, after confirming this codebase had nothing
//! quite like Substance Designer's "Distance" node: the existing Hard-
//! Surface Nodes package (`distance_field.rs`, `redistance.rs`,
//! `shape_boolean.rs`, `bevel_profile.rs`, `edge_wear.rs`) already covers
//! exact signed-distance-field modeling for hard-surface work, but has no
//! equivalent of Substance's more general "dilate an arbitrary value
//! outward from a mask" grunge/decay effect — and its own `Sdf` port type
//! is deliberately walled off from `Gray`/`Color` ports, so that package
//! can't be chained into one directly either.
//!
//! Unlike `distance_field.rs` (an exact, CPU-side Euclidean transform —
//! deliberately exact because hard-surface modeling needs genuinely clean
//! geometry), this node's core nearest-seed pipeline is a real-time GPU
//! one: three compute passes (`distance_spread_seed.wgsl`,
//! `distance_spread_jfa.wgsl` ×N, `distance_spread_resolve.wgsl`)
//! implementing Jump Flooding (JFA+1) — the standard real-time-graphics
//! technique for an approximate nearest-feature transform, the same
//! technique `distance_field.rs`'s own doc comment already names as its
//! planned (not yet built) GPU replacement. JFA's small, very-rare
//! approximation error is a non-issue for a grunge effect in a way it
//! wouldn't be for exact bevel geometry — a deliberately different
//! precision/speed tradeoff for a deliberately different use case, not a
//! shared implementation between the two nodes. Luminance Jitter adds a
//! CPU-side step in between the GPU passes — see `crate::connected_
//! components`'s own doc comment for why.
//!
//! With no Source connected, the propagated value is a flat white (1.0),
//! not Mask's own grayscale value at the seed — matching Substance's own
//! "no Source" mode exactly ("outputs a gradient... or the color at that
//! position in a source image" — the gradient alone, not a copy of Mask's
//! own brightness there). An earlier version of this node *did* fall back
//! to sampling Mask itself, which looked like barely any effect at all on
//! a mask with its own internal brightness variation (e.g. Pattern
//! Generator's Edge Darkening darkening a brick's edge pixels) — the
//! owner's own bug report, confirmed by dumping actual pixel values: the
//! "spread" was real but propagating an already-dim edge value instead of
//! a clean white, which read as a no-op next to the original mask.

use crate::connected_components::label_components;
use crate::shaders::full_source;
use crate::tiling::TilingMode;

const SEED_NODE_KIND: &str = "distance_spread_seed";
const JFA_NODE_KIND: &str = "distance_spread_jfa";
const RESOLVE_NODE_KIND: &str = "distance_spread_resolve";

/// Substance Designer's own "Combine Source/Distance" toggle — the owner's
/// own follow-up request, after comparing side by side against the real
/// node's UI. `Combine` is this node's existing, only-ever behavior: the
/// propagated value scaled by the smooth 0..1 falloff. `OnlySource` drops
/// that falloff multiplication entirely, showing the propagated value at
/// full, flat strength everywhere within Distance, with a hard cutoff at
/// the edge instead of a gradient — Substance's own "solid color only from
/// the Source input."
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceCombineMode {
    #[default]
    Combine,
    OnlySource,
}

impl DistanceCombineMode {
    pub const NAMES: [&'static str; 2] = ["Combine", "Only Source"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Combine" => Some(Self::Combine),
            "Only Source" => Some(Self::OnlySource),
            _ => None,
        }
    }

    fn as_u32(self) -> u32 {
        match self {
            Self::Combine => 0,
            Self::OnlySource => 1,
        }
    }
}

pub struct DistanceSpreadParams {
    /// In UV space (0..1), per CLAUDE.md §6.2 — converted to a pixel
    /// distance at dispatch time, same reasoning as Blur's own `radius`.
    pub distance: f32,
    /// A plain multiplier on `distance`'s own effective reach — Substance
    /// Designer's own Distance node has an equivalent "Scale" param,
    /// confirmed by the owner's own side-by-side comparison screenshots
    /// (same mask, same `Distance` value, only `Scale` changed: a higher
    /// value spreads far enough to flood right through the gaps between
    /// separate shapes into soft rounded blobs, a lower one keeps the
    /// gaps/seams between them crisp and visible). Kept as its own
    /// multiplier rather than folded into `distance` directly for the
    /// same reason Substance's own two params are separate: quick global
    /// reach adjustments (e.g. driven by an exposed graph variable shared
    /// across several nodes) without re-tuning each node's own
    /// hand-picked `distance` value.
    pub scale: f32,
    /// Which Mask value counts as a seed to spread from.
    pub threshold: f32,
    pub tiling_mode: TilingMode,
    pub combine_mode: DistanceCombineMode,
    /// Per-blob random brightness variation — same idea as Pattern
    /// Generator's own Luminance Jitter, just per disconnected *island* in
    /// Mask instead of per grid tile (Distance has no grid to begin with).
    /// Every seed pixel belonging to the same connected blob gets the exact
    /// same jittered brightness, via a CPU-side connected-component-
    /// labeling pass (`crate::connected_components`) — see its own doc
    /// comment for why plain per-pixel hashing can't give this "one value
    /// per blob" look (it only has each pixel's own nearest-seed
    /// coordinate, which varies continuously across a solid region, not a
    /// stable per-blob identity) and why that labeling runs on the CPU
    /// rather than as more GPU passes.
    pub luminance_jitter: f32,
    /// Independent of nothing else on this node — unlike Pattern
    /// Generator's own Luminance Seed (which exists specifically so it
    /// *doesn't* also reshuffle Size/Rotation Jitter), Distance has no
    /// other per-feature randomness to protect, so this is simply this
    /// node's own Seed.
    pub luminance_seed: u32,
}

impl Default for DistanceSpreadParams {
    fn default() -> Self {
        Self {
            distance: 0.05,
            scale: 1.0,
            threshold: 0.5,
            tiling_mode: TilingMode::default(),
            combine_mode: DistanceCombineMode::default(),
            luminance_jitter: 0.0,
            luminance_seed: 0,
        }
    }
}

impl DistanceSpreadParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            distance: params
                .get("distance")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.distance),
            scale: params
                .get("scale")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.scale),
            threshold: params
                .get("threshold")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.threshold),
            tiling_mode: params
                .get("tiling_mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => TilingMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.tiling_mode),
            combine_mode: params
                .get("combine_mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => DistanceCombineMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.combine_mode),
            luminance_jitter: params
                .get("luminance_jitter")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.luminance_jitter),
            luminance_seed: params
                .get("luminance_seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(defaults.luminance_seed),
        }
    }
}

/// The Jump Flooding step sizes `dispatch` runs, in order: halving from
/// roughly half the resolution down to 1, then 1 again (the "+1"
/// refinement). Works for any resolution, not just powers of two —
/// `next_power_of_two` just means a non-power-of-two size gets a couple of
/// redundant extra steps rather than an incomplete propagation.
fn step_sizes(resolution: u32) -> Vec<i32> {
    let mut step = (resolution.next_power_of_two() / 2).max(1);
    let mut steps = Vec::new();
    loop {
        steps.push(step as i32);
        if step == 1 {
            break;
        }
        step /= 2;
    }
    steps.push(1); // JFA+1
    steps
}

/// Where each of this node's declared output ports should be written —
/// same `<Node>OutputKeys` shape every other multi-output node in this
/// crate already uses (`thread_shape::ThreadOutputKeys`,
/// `fasteners::FastenersOutputKeys`). `output` stays this struct's first
/// field for the same reason it's the first-declared port in `defs.rs`:
/// an existing saved graph's own connection from this node predates Cell
/// ID and must keep resolving to the same texture.
pub struct DistanceSpreadOutputKeys<'a> {
    pub output: &'a str,
    pub cell_id: &'a str,
}

/// Runs the seed pass, then Jump Flooding (one pass per `step_sizes`, for
/// the nearest-seed coordinate), then — only when Luminance Jitter is
/// actually non-zero or Cell ID is connected (`compute_cell_id`) — a
/// CPU-side connected-component labeling pass (see
/// `crate::connected_components`'s own doc comment for why that runs on
/// the CPU rather than as more GPU rounds), then the resolve pass, which
/// combines both. See this module's own doc comment for why three-plus
/// passes instead of one, and `blur::dispatch`'s own doc comment for the
/// same "ping-pong between two scratch textures" shape the JFA pipeline
/// here follows.
///
/// `compute_cell_id` is the caller's own answer to "is anything actually
/// connected to the Cell ID output port" (`mc_nodes::registry`'s own
/// job — this function has no access to the graph to check itself). It
/// matters for more than performance: when `false`, Cell ID's own output
/// texture still gets written every pixel (the resolve shader doesn't
/// know or care whether anyone's listening), but from whatever the
/// Luminance Jitter skip path's dummy label placeholder happens to
/// contain — garbage, but *unobserved* garbage, since nothing downstream
/// ever queries that texture pool key in that case. `true` forces the
/// real labeling pass to run even when Luminance Jitter is left at its
/// default 0, since Cell ID needs genuine converged labels to mean
/// anything.
#[allow(clippy::too_many_arguments)]
pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    mask_key: &str,
    source_key: Option<&str>,
    keys: DistanceSpreadOutputKeys,
    compute_cell_id: bool,
    resolution: u32,
    params: &DistanceSpreadParams,
) {
    let output_key = keys.output;
    let wrap_flags = params.tiling_mode.as_u32();
    let distance_px = (params.distance * params.scale * resolution as f32).max(0.0001);

    let seed_key = format!("{output_key}__distance_seed");
    let jfa_a_key = format!("{output_key}__distance_jfa_a");
    let jfa_b_key = format!("{output_key}__distance_jfa_b");
    let label_key = format!("{output_key}__distance_label");

    // The seed and Jump Flooding passes all record into one shared
    // encoder, submitted once — not one `queue.submit()` per pass (an
    // earlier version of this module did that per label-propagation
    // round too, when that still ran on the GPU; batching barely helped
    // there, since that cost turned out to be real GPU compute time
    // scaling with resolution, not fixed per-submission overhead — see
    // `crate::connected_components`'s own doc comment for the actual fix).
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("distance_spread_encoder"),
        });

    record_seed_pass(
        gpu,
        pool,
        &mut encoder,
        mask_key,
        &seed_key,
        resolution,
        params.threshold,
    );

    let rounds = step_sizes(resolution);

    // Nearest-seed-coordinate pipeline (unchanged from before Luminance
    // Jitter existed). Round 0 reads from `seed_key` and writes into
    // `jfa_a_key`; every round after that ping-pongs between `jfa_a_key`
    // and `jfa_b_key` only — `seed_key` itself is never written to again,
    // since the CPU-side labeling step below needs to read it back
    // untouched (a real bug this fixes: an earlier version of this loop
    // ping-ponged between the seed pass's own output texture and one
    // scratch texture, which overwrote the seed pass's own data by round
    // 2 — the label pipeline was then unknowingly reading corrupted
    // JFA-in-progress data instead of the original per-seed-pixel
    // coordinates it needs, caught by
    // `luminance_jitter_gives_separate_blobs_different_brightness` failing
    // after this file's encoder-batching refactor).
    let mut current = seed_key.clone();
    let mut other = jfa_a_key;
    for (i, &step) in rounds.iter().enumerate() {
        record_jfa_pass(
            gpu,
            pool,
            &mut encoder,
            &current,
            &other,
            resolution,
            step,
            wrap_flags,
        );
        if i == 0 {
            current = other;
            other = jfa_b_key.clone();
        } else {
            std::mem::swap(&mut current, &mut other);
        }
    }

    let need_labels = params.luminance_jitter != 0.0 || compute_cell_id;
    if !need_labels {
        // `distance_spread_resolve.wgsl`'s own `luminance_scale` always
        // evaluates to exactly 1.0 when `luminance_jitter == 0.0`
        // regardless of what the label texture contains, and Cell ID
        // isn't connected to anything either, so there's nothing to
        // compute — `seed_key` stands in as a harmless, already-valid
        // R32Sint texture for the resolve pass to bind, never actually
        // read meaningfully (see `dispatch`'s own doc comment on
        // `compute_cell_id` for why that's fine). One encoder, one
        // submit, same as every other node.
        record_resolve_pass(
            gpu,
            pool,
            &mut encoder,
            source_key.unwrap_or(mask_key),
            source_key.is_some(),
            &current,
            &seed_key,
            keys.output,
            keys.cell_id,
            resolution,
            wrap_flags,
            distance_px,
            params.combine_mode,
            params.luminance_jitter,
            params.luminance_seed,
        );
        gpu.queue.submit(Some(encoder.finish()));
        return;
    }

    // Luminance Jitter needs each disconnected blob's own stable identity
    // — the JFA pipeline above only has each pixel's nearest-seed
    // *coordinate*, which varies continuously across a solid region, not a
    // per-blob identity. Submit what's recorded so far, then read the seed
    // pass's own output back to the CPU to label it (see
    // `crate::connected_components`'s own doc comment for why that's a CPU
    // union-find now, not a GPU relaxation loop) before recording the
    // resolve pass, which needs the label texture as an input.
    gpu.queue.submit(Some(encoder.finish()));

    let seed_texture = pool
        .texture(&seed_key)
        .unwrap_or_else(|| panic!("distance_spread: no texture at seed key {seed_key:?}"))
        .clone();
    let packed = gpu.read_r32i(&seed_texture, resolution);
    let occupied: Vec<bool> = packed.iter().map(|&p| p != -1).collect();
    let labels = label_components(
        &occupied,
        resolution,
        params.tiling_mode.wrap_x(),
        params.tiling_mode.wrap_y(),
    );
    gpu.write_r32i(pool, &label_key, resolution, &labels);

    let mut resolve_encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("distance_spread_resolve_encoder"),
        });
    // `source_key.unwrap_or(mask_key)` only supplies a texture the bind
    // group can legally point at — the shader itself ignores it entirely
    // when `has_source` is false, using a flat white instead (see
    // `distance_spread_resolve.wgsl`'s own doc comment for why that's not
    // the same thing as falling back to Mask's own value here).
    record_resolve_pass(
        gpu,
        pool,
        &mut resolve_encoder,
        source_key.unwrap_or(mask_key),
        source_key.is_some(),
        &current,
        &label_key,
        keys.output,
        keys.cell_id,
        resolution,
        wrap_flags,
        distance_px,
        params.combine_mode,
        params.luminance_jitter,
        params.luminance_seed,
    );
    gpu.queue.submit(Some(resolve_encoder.finish()));
}

fn record_seed_pass(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    encoder: &mut wgpu::CommandEncoder,
    mask_key: &str,
    seed_out_key: &str,
    resolution: u32,
    threshold: f32,
) {
    let mask_view = pool
        .view(mask_key)
        .unwrap_or_else(|| panic!("distance_spread: no texture at mask key {mask_key:?}"))
        .clone();

    let shader_source = full_source("distance_spread_seed.wgsl");
    let pipeline = gpu.compute_pipeline(SEED_NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&threshold.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distance_spread_seed_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let seed_out_view = pool.get_or_create(
        &gpu.device,
        seed_out_key,
        resolution,
        wgpu::TextureFormat::R32Sint,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("distance_spread_seed_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&mask_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(seed_out_view),
            },
        ],
    });

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("distance_spread_seed_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    let workgroups = resolution.div_ceil(8);
    pass.dispatch_workgroups(workgroups, workgroups, 1);
}

#[allow(clippy::too_many_arguments)]
fn record_jfa_pass(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    encoder: &mut wgpu::CommandEncoder,
    seed_in_key: &str,
    seed_out_key: &str,
    resolution: u32,
    step: i32,
    wrap_flags: u32,
) {
    let seed_in_view = pool
        .view(seed_in_key)
        .unwrap_or_else(|| panic!("distance_spread: no texture at seed key {seed_in_key:?}"))
        .clone();

    let shader_source = full_source("distance_spread_jfa.wgsl");
    let pipeline = gpu.compute_pipeline(JFA_NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 16];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&step.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&wrap_flags.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distance_spread_jfa_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let seed_out_view = pool.get_or_create(
        &gpu.device,
        seed_out_key,
        resolution,
        wgpu::TextureFormat::R32Sint,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("distance_spread_jfa_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&seed_in_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(seed_out_view),
            },
        ],
    });

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("distance_spread_jfa_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    let workgroups = resolution.div_ceil(8);
    pass.dispatch_workgroups(workgroups, workgroups, 1);
}

#[allow(clippy::too_many_arguments)]
fn record_resolve_pass(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    encoder: &mut wgpu::CommandEncoder,
    source_key: &str,
    has_source: bool,
    seed_in_key: &str,
    label_in_key: &str,
    output_key: &str,
    cell_id_key: &str,
    resolution: u32,
    wrap_flags: u32,
    distance_px: f32,
    combine_mode: DistanceCombineMode,
    luminance_jitter: f32,
    luminance_seed: u32,
) {
    let source_view = pool
        .view(source_key)
        .unwrap_or_else(|| panic!("distance_spread: no texture at source key {source_key:?}"))
        .clone();
    let seed_in_view = pool
        .view(seed_in_key)
        .unwrap_or_else(|| panic!("distance_spread: no texture at seed key {seed_in_key:?}"))
        .clone();
    let label_in_view = pool
        .view(label_in_key)
        .unwrap_or_else(|| panic!("distance_spread: no texture at label key {label_in_key:?}"))
        .clone();

    let shader_source = full_source("distance_spread_resolve.wgsl");
    let pipeline = gpu.compute_pipeline(RESOLVE_NODE_KIND, &shader_source, "main");

    let mut param_bytes = [0u8; 32];
    param_bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    param_bytes[4..8].copy_from_slice(&wrap_flags.to_le_bytes());
    param_bytes[8..12].copy_from_slice(&distance_px.to_le_bytes());
    param_bytes[12..16].copy_from_slice(&(has_source as u32).to_le_bytes());
    param_bytes[16..20].copy_from_slice(&combine_mode.as_u32().to_le_bytes());
    param_bytes[20..24].copy_from_slice(&luminance_jitter.to_le_bytes());
    param_bytes[24..28].copy_from_slice(&luminance_seed.to_le_bytes());
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("distance_spread_resolve_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let output_view = pool
        .get_or_create(
            &gpu.device,
            output_key,
            resolution,
            wgpu::TextureFormat::Rgba16Float,
        )
        .clone();
    let cell_id_view = pool.get_or_create(
        &gpu.device,
        cell_id_key,
        resolution,
        wgpu::TextureFormat::Rgba16Float,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("distance_spread_resolve_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&source_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&seed_in_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&label_in_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&output_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(cell_id_view),
            },
        ],
    });

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("distance_spread_resolve_pass"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    let workgroups = resolution.div_ceil(8);
    pass.dispatch_workgroups(workgroups, workgroups, 1);
}

#[cfg(test)]
mod tests {
    use super::{dispatch, DistanceCombineMode, DistanceSpreadOutputKeys, DistanceSpreadParams};
    use crate::test_support::{assert_tiles_seamlessly, pixel, RESOLUTION};
    use crate::tiling::TilingMode;

    fn single_seed_mask(resolution: u32, seed_x: u32, seed_y: u32) -> Vec<[f32; 4]> {
        (0..resolution * resolution)
            .map(|i| {
                let x = i % resolution;
                let y = i / resolution;
                if x == seed_x && y == seed_y {
                    [1.0, 1.0, 1.0, 1.0]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                }
            })
            .collect()
    }

    fn headless() -> (mc_gpu::Context, mc_gpu::TexturePool) {
        (
            mc_gpu::Context::new_headless().expect("headless GPU context"),
            mc_gpu::TexturePool::default(),
        )
    }

    #[test]
    fn value_is_full_strength_at_the_seed_and_fades_to_zero_beyond_distance() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        let mask = single_seed_mask(RESOLUTION, 32, 32);
        gpu.write_rgba16f(&mut pool, "mask", RESOLUTION, &mask);

        let params = DistanceSpreadParams {
            distance: 0.15, // 0.15 * 64 ≈ 9.6px
            threshold: 0.5,
            tiling_mode: TilingMode::Both,
            ..DistanceSpreadParams::default()
        };
        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output",
                cell_id: "output__cell_id",
            },
            false,
            RESOLUTION,
            &params,
        );
        let out = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        let at_seed = pixel(&out, RESOLUTION, 32, 32);
        assert!(
            at_seed[0] > 240,
            "seed itself should read full strength, got {at_seed:?}"
        );

        let mid_range = pixel(&out, RESOLUTION, 37, 32); // 5px away
        assert!(
            mid_range[0] > 0 && mid_range[0] < 240,
            "5px away (within the 9.6px distance) should be partially faded, got {mid_range:?}"
        );

        let far = pixel(&out, RESOLUTION, 52, 32); // 20px away, beyond 9.6px
        assert!(
            far[0] < 10,
            "far beyond Distance should read ~0, got {far:?}"
        );
    }

    /// Substance Designer's own "Only Source" mode: full strength
    /// everywhere within Distance (no gradient fade, unlike Combine), a
    /// hard cutoff to 0 beyond it.
    #[test]
    fn only_source_mode_is_a_hard_cutoff_not_a_fade() {
        let mask = single_seed_mask(RESOLUTION, 32, 32);
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        gpu.write_rgba16f(&mut pool, "mask", RESOLUTION, &mask);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output",
                cell_id: "output__cell_id",
            },
            false,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.15, // 0.15 * 64 ≈ 9.6px
                threshold: 0.5,
                tiling_mode: TilingMode::Both,
                combine_mode: DistanceCombineMode::OnlySource,
                ..DistanceSpreadParams::default()
            },
        );
        let out = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);

        let at_seed = pixel(&out, RESOLUTION, 32, 32);
        let mid_range = pixel(&out, RESOLUTION, 37, 32); // 5px away, within distance
        let far = pixel(&out, RESOLUTION, 52, 32); // 20px away, beyond distance
        assert!(
            at_seed[0] > 240,
            "at the seed should be full strength, got {at_seed:?}"
        );
        assert!(
            mid_range[0] > 240,
            "within Distance, Only Source should be full strength too, not faded, got \
             {mid_range:?}"
        );
        assert!(
            far[0] < 10,
            "beyond Distance should still cut off to ~0, got {far:?}"
        );
    }

    /// `combine_mode` defaults to `Combine`, the node's own original,
    /// only-ever behavior before this param existed — an older saved graph
    /// with no `combine_mode` key must keep fading, not suddenly cut off.
    #[test]
    fn combine_mode_defaults_to_combine() {
        assert_eq!(
            DistanceSpreadParams::default().combine_mode,
            DistanceCombineMode::Combine
        );
        let parsed = DistanceSpreadParams::from_graph(&std::collections::BTreeMap::new());
        assert_eq!(parsed.combine_mode, DistanceCombineMode::Combine);
    }

    /// Mono-pixel mask at two coordinates, each its own disconnected blob.
    fn two_seed_mask(resolution: u32, a: (u32, u32), b: (u32, u32)) -> Vec<[f32; 4]> {
        (0..resolution * resolution)
            .map(|i| {
                let x = i % resolution;
                let y = i / resolution;
                if (x, y) == a || (x, y) == b {
                    [1.0, 1.0, 1.0, 1.0]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                }
            })
            .collect()
    }

    /// The owner's own request, matching Pattern Generator's own Luminance
    /// Jitter/Seed pair — except per disconnected *blob* in Mask instead of
    /// per grid tile. Two separate single-pixel seeds, far enough apart
    /// that each stays its own island, should (almost certainly) get
    /// different jittered brightness.
    #[test]
    fn luminance_jitter_gives_separate_blobs_different_brightness() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        let mask = two_seed_mask(RESOLUTION, (10, 10), (50, 50));
        gpu.write_rgba16f(&mut pool, "mask", RESOLUTION, &mask);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output",
                cell_id: "output__cell_id",
            },
            false,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.1,
                luminance_jitter: 1.0,
                luminance_seed: 7,
                ..DistanceSpreadParams::default()
            },
        );
        let out = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        let a = pixel(&out, RESOLUTION, 10, 10);
        let b = pixel(&out, RESOLUTION, 50, 50);
        assert_ne!(
            a, b,
            "two disconnected blobs should get different jittered brightness"
        );
    }

    /// The core claim: every pixel of one *solid, connected* blob shares
    /// the exact same jittered brightness — not per-pixel noise, which a
    /// naive hash of the raw nearest-seed coordinate would produce (every
    /// interior pixel of a solid region is its own nearest seed).
    #[test]
    fn luminance_jitter_is_identical_across_one_connected_blob() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        let resolution = RESOLUTION;
        // A solid 5x5 filled square — every pixel in it is a seed, and
        // every pixel is 4-connected to its neighbors within the square.
        let mask: Vec<[f32; 4]> = (0..resolution * resolution)
            .map(|i| {
                let x = i % resolution;
                let y = i / resolution;
                if (20..25).contains(&x) && (20..25).contains(&y) {
                    [1.0, 1.0, 1.0, 1.0]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                }
            })
            .collect();
        gpu.write_rgba16f(&mut pool, "mask", resolution, &mask);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output",
                cell_id: "output__cell_id",
            },
            false,
            resolution,
            &DistanceSpreadParams {
                distance: 0.2,
                luminance_jitter: 1.0,
                luminance_seed: 3,
                ..DistanceSpreadParams::default()
            },
        );
        let out = gpu.read_rgba8(pool.texture("output").unwrap(), resolution);
        let top_left = pixel(&out, resolution, 20, 20);
        let bottom_right = pixel(&out, resolution, 24, 24);
        assert_eq!(
            top_left, bottom_right,
            "opposite corners of one solid, connected blob should read identical brightness, \
             got {top_left:?} vs {bottom_right:?}"
        );
    }

    /// Two seeds one texel apart (directly 4-connected) must be treated as
    /// one blob, even though they're still two distinct pixels.
    #[test]
    fn luminance_jitter_is_identical_for_two_directly_touching_seeds() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        let mask = two_seed_mask(RESOLUTION, (30, 30), (31, 30));
        gpu.write_rgba16f(&mut pool, "mask", RESOLUTION, &mask);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output",
                cell_id: "output__cell_id",
            },
            false,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.1,
                luminance_jitter: 1.0,
                luminance_seed: 11,
                ..DistanceSpreadParams::default()
            },
        );
        let out = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        let a = pixel(&out, RESOLUTION, 30, 30);
        let b = pixel(&out, RESOLUTION, 31, 30);
        assert_eq!(
            a, b,
            "two directly-touching seed pixels should be one blob, got {a:?} vs {b:?}"
        );
    }

    /// `luminance_jitter` defaults to 0 — an older saved graph with no
    /// Luminance Jitter/Seed keys at all must render exactly as before
    /// this feature existed.
    #[test]
    fn luminance_jitter_zero_is_a_no_op() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        let mask = single_seed_mask(RESOLUTION, 32, 32);
        gpu.write_rgba16f(&mut pool, "mask", RESOLUTION, &mask);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output_a",
                cell_id: "output_a__cell_id",
            },
            false,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.15,
                luminance_jitter: 0.0,
                luminance_seed: 1,
                ..DistanceSpreadParams::default()
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output_b",
                cell_id: "output_b__cell_id",
            },
            false,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.15,
                luminance_jitter: 0.0,
                luminance_seed: 99,
                ..DistanceSpreadParams::default()
            },
        );
        let a = gpu.read_rgba8(pool.texture("output_a").unwrap(), RESOLUTION);
        let b = gpu.read_rgba8(pool.texture("output_b").unwrap(), RESOLUTION);
        assert_eq!(
            a, b,
            "with Luminance Jitter at 0, Luminance Seed should have no effect at all"
        );
    }

    /// Every pixel of one connected blob shares the same Cell ID — same
    /// claim as `luminance_jitter_is_identical_across_one_connected_blob`,
    /// for the independent signal the owner actually wants to feed
    /// downstream (into Edge Detect, for a real stone-to-stone seam) now
    /// that Luminance Jitter's own per-blob labeling is exposed directly.
    #[test]
    fn cell_id_is_identical_across_one_connected_blob() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        let resolution = RESOLUTION;
        let mask: Vec<[f32; 4]> = (0..resolution * resolution)
            .map(|i| {
                let x = i % resolution;
                let y = i / resolution;
                if (20..25).contains(&x) && (20..25).contains(&y) {
                    [1.0, 1.0, 1.0, 1.0]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                }
            })
            .collect();
        gpu.write_rgba16f(&mut pool, "mask", resolution, &mask);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output",
                cell_id: "cell_id",
            },
            true,
            resolution,
            &DistanceSpreadParams {
                distance: 0.2,
                ..DistanceSpreadParams::default()
            },
        );
        let cell_id = gpu.read_rgba8(pool.texture("cell_id").unwrap(), resolution);
        let top_left = pixel(&cell_id, resolution, 20, 20);
        let bottom_right = pixel(&cell_id, resolution, 24, 24);
        assert_eq!(
            top_left, bottom_right,
            "opposite corners of one solid, connected blob should share one Cell ID, got \
             {top_left:?} vs {bottom_right:?}"
        );
    }

    /// Two disconnected blobs get different Cell IDs — same claim as
    /// `luminance_jitter_gives_separate_blobs_different_brightness`, for
    /// the independent Cell ID signal.
    #[test]
    fn cell_id_differs_between_disconnected_blobs() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        let mask = two_seed_mask(RESOLUTION, (10, 10), (50, 50));
        gpu.write_rgba16f(&mut pool, "mask", RESOLUTION, &mask);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output",
                cell_id: "cell_id",
            },
            true,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.1,
                ..DistanceSpreadParams::default()
            },
        );
        let cell_id = gpu.read_rgba8(pool.texture("cell_id").unwrap(), RESOLUTION);
        let a = pixel(&cell_id, RESOLUTION, 10, 10);
        let b = pixel(&cell_id, RESOLUTION, 50, 50);
        assert_ne!(a, b, "two disconnected blobs should get different Cell IDs");
    }

    /// Cell ID and Luminance Jitter are genuinely independent signals, not
    /// the same hash reused under two names — same topology, same Cell ID
    /// regardless of what Luminance Jitter is set to.
    #[test]
    fn cell_id_is_unaffected_by_luminance_jitter() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        let mask = two_seed_mask(RESOLUTION, (10, 10), (50, 50));
        gpu.write_rgba16f(&mut pool, "mask", RESOLUTION, &mask);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output_a",
                cell_id: "cell_id_a",
            },
            true,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.1,
                luminance_jitter: 0.0,
                ..DistanceSpreadParams::default()
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output_b",
                cell_id: "cell_id_b",
            },
            true,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.1,
                luminance_jitter: 1.0,
                luminance_seed: 42,
                ..DistanceSpreadParams::default()
            },
        );
        let cell_id_a = gpu.read_rgba8(pool.texture("cell_id_a").unwrap(), RESOLUTION);
        let cell_id_b = gpu.read_rgba8(pool.texture("cell_id_b").unwrap(), RESOLUTION);
        assert_eq!(
            cell_id_a, cell_id_b,
            "Cell ID should be the same regardless of Luminance Jitter's own value/seed"
        );
    }

    /// `compute_cell_id: false` (nothing connected to Cell ID — the common
    /// case) must still produce a correct main `output`, and must not
    /// panic reading whatever placeholder ends up in the Cell ID texture.
    #[test]
    fn compute_cell_id_false_does_not_affect_main_output() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        let mask = two_seed_mask(RESOLUTION, (10, 10), (50, 50));
        gpu.write_rgba16f(&mut pool, "mask", RESOLUTION, &mask);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output_without",
                cell_id: "cell_id_without",
            },
            false,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.1,
                ..DistanceSpreadParams::default()
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output_with",
                cell_id: "cell_id_with",
            },
            true,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.1,
                ..DistanceSpreadParams::default()
            },
        );
        let without = gpu.read_rgba8(pool.texture("output_without").unwrap(), RESOLUTION);
        let with = gpu.read_rgba8(pool.texture("output_with").unwrap(), RESOLUTION);
        assert_eq!(
            without, with,
            "compute_cell_id shouldn't change the main output at all"
        );
    }

    /// Regression test for the owner's own bug report: a mask whose seed
    /// pixel is only *just* above threshold (not a clean 1.0 — e.g. Pattern
    /// Generator's own Edge Darkening dimming a brick's edge) must still
    /// propagate full white with no Source connected. Reusing the mask's
    /// own sub-1.0 value there (an earlier, wrong version of this node)
    /// made the spread barely visible against the input.
    #[test]
    fn with_no_source_the_spread_is_flat_white_not_the_masks_own_dim_seed_value() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        let resolution = RESOLUTION;
        let seed_x = 32u32;
        let seed_y = 32u32;
        // Every pixel is a dim 0.6 (just above the 0.5 default threshold),
        // not a clean 1.0 — unlike `single_seed_mask`'s binary 0/1 masks.
        let mask: Vec<[f32; 4]> = (0..resolution * resolution)
            .map(|i| {
                let x = i % resolution;
                let y = i / resolution;
                if x == seed_x && y == seed_y {
                    [0.6, 0.6, 0.6, 1.0]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                }
            })
            .collect();
        gpu.write_rgba16f(&mut pool, "mask", resolution, &mask);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output",
                cell_id: "output__cell_id",
            },
            false,
            resolution,
            &DistanceSpreadParams {
                distance: 0.15,
                threshold: 0.5,
                tiling_mode: TilingMode::Both,
                ..DistanceSpreadParams::default()
            },
        );
        let out = gpu.read_rgba8(pool.texture("output").unwrap(), resolution);
        let at_seed = pixel(&out, resolution, seed_x, seed_y);
        assert!(
            at_seed[0] > 250,
            "no Source connected should spread flat white regardless of the mask's own dim \
             seed value (0.6*255≈153), got {at_seed:?}"
        );
    }

    /// With Source connected, the propagated value comes from Source's own
    /// pixel at the nearest seed — not a flat white, and not Mask's value.
    #[test]
    fn with_source_connected_the_spread_uses_sources_own_value() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        let mask = single_seed_mask(RESOLUTION, 32, 32);
        gpu.write_rgba16f(&mut pool, "mask", RESOLUTION, &mask);
        gpu.fill_rgba16f(&mut pool, "source", RESOLUTION, [0.2, 0.4, 0.6, 1.0]);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            Some("source"),
            DistanceSpreadOutputKeys {
                output: "output",
                cell_id: "output__cell_id",
            },
            false,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.15,
                threshold: 0.5,
                tiling_mode: TilingMode::Both,
                ..DistanceSpreadParams::default()
            },
        );
        let out = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        let at_seed = pixel(&out, RESOLUTION, 32, 32);
        assert!(
            (at_seed[0] as i32 - 51).abs() <= 2
                && (at_seed[1] as i32 - 102).abs() <= 2
                && (at_seed[2] as i32 - 153).abs() <= 2,
            "at the seed, output should match Source's own color (0.2,0.4,0.6), got {at_seed:?}"
        );
    }

    #[test]
    fn no_seed_produces_all_zero_output() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        gpu.fill_rgba16f(&mut pool, "mask", RESOLUTION, [0.0, 0.0, 0.0, 1.0]);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output",
                cell_id: "output__cell_id",
            },
            false,
            RESOLUTION,
            &DistanceSpreadParams::default(),
        );
        let out = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        assert!(
            out.as_chunks::<4>()
                .0
                .iter()
                .all(|px| px[0..3] == [0, 0, 0]),
            "no seed anywhere in the mask should produce an all-zero RGB output (alpha stays \
             opaque, same as every other Gray-broadcast node)"
        );
    }

    #[test]
    fn tiles_seamlessly() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        // A seed close to the left edge, with a distance wide enough that
        // its falloff genuinely reaches across the tile seam — wrap-
        // addressing must be correct for the seam to stay as smooth as any
        // other interior step.
        let mask = single_seed_mask(RESOLUTION, 5, 32);
        gpu.write_rgba16f(&mut pool, "mask", RESOLUTION, &mask);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output",
                cell_id: "output__cell_id",
            },
            false,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.3,
                threshold: 0.5,
                tiling_mode: TilingMode::Both,
                ..DistanceSpreadParams::default()
            },
        );
        let out = gpu.read_rgba8(pool.texture("output").unwrap(), RESOLUTION);
        assert_tiles_seamlessly(&out, RESOLUTION);
    }

    #[test]
    fn is_deterministic() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        let mask = single_seed_mask(RESOLUTION, 20, 40);
        gpu.write_rgba16f(&mut pool, "mask", RESOLUTION, &mask);
        let params = DistanceSpreadParams::default();

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output_a",
                cell_id: "output_a__cell_id",
            },
            false,
            RESOLUTION,
            &params,
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output_b",
                cell_id: "output_b__cell_id",
            },
            false,
            RESOLUTION,
            &params,
        );
        let a = gpu.read_rgba8(pool.texture("output_a").unwrap(), RESOLUTION);
        let b = gpu.read_rgba8(pool.texture("output_b").unwrap(), RESOLUTION);
        assert_eq!(a, b, "same input + params produced different pixels");
    }

    #[test]
    fn from_graph_round_trips() {
        let mut params = std::collections::BTreeMap::new();
        params.insert("distance".to_string(), mc_core::ParamValue::Float(0.2));
        params.insert("scale".to_string(), mc_core::ParamValue::Float(2.5));
        params.insert("threshold".to_string(), mc_core::ParamValue::Float(0.3));
        params.insert(
            "tiling_mode".to_string(),
            mc_core::ParamValue::Enum("Off".to_string()),
        );
        params.insert(
            "combine_mode".to_string(),
            mc_core::ParamValue::Enum("Only Source".to_string()),
        );
        params.insert(
            "luminance_jitter".to_string(),
            mc_core::ParamValue::Float(0.4),
        );
        params.insert("luminance_seed".to_string(), mc_core::ParamValue::Seed(9));
        let parsed = DistanceSpreadParams::from_graph(&params);
        assert_eq!(parsed.distance, 0.2);
        assert_eq!(parsed.scale, 2.5);
        assert_eq!(parsed.threshold, 0.3);
        assert_eq!(parsed.tiling_mode, TilingMode::Off);
        assert_eq!(parsed.combine_mode, DistanceCombineMode::OnlySource);
        assert_eq!(parsed.luminance_jitter, 0.4);
        assert_eq!(parsed.luminance_seed, 9);

        let defaults = DistanceSpreadParams::from_graph(&std::collections::BTreeMap::new());
        assert_eq!(defaults.distance, 0.05);
        assert_eq!(defaults.scale, 1.0);
        assert_eq!(defaults.threshold, 0.5);
        assert_eq!(defaults.tiling_mode, TilingMode::Both);
        assert_eq!(defaults.combine_mode, DistanceCombineMode::Combine);
        assert_eq!(defaults.luminance_jitter, 0.0);
        assert_eq!(defaults.luminance_seed, 0);
    }

    /// Scale is a plain multiplier on Distance's own effective reach —
    /// `distance: 0.1, scale: 2.0` should spread exactly as far as
    /// `distance: 0.2, scale: 1.0` (confirmed against the owner's own
    /// side-by-side Substance Designer comparison, see `DistanceSpreadParams
    /// ::scale`'s own doc comment). Checked by comparing actual output
    /// pixels, not just the internal `distance_px` math, since that's what
    /// the owner would actually see change (or not) on screen.
    #[test]
    fn scale_is_a_plain_multiplier_on_distance() {
        let (gpu, mut pool) = headless();
        let mut gpu = gpu;
        let mask = single_seed_mask(RESOLUTION, RESOLUTION / 2, RESOLUTION / 2);
        gpu.write_rgba16f(&mut pool, "mask", RESOLUTION, &mask);

        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output_direct",
                cell_id: "output_direct__cell_id",
            },
            false,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.2,
                scale: 1.0,
                ..DistanceSpreadParams::default()
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            "mask",
            None,
            DistanceSpreadOutputKeys {
                output: "output_scaled",
                cell_id: "output_scaled__cell_id",
            },
            false,
            RESOLUTION,
            &DistanceSpreadParams {
                distance: 0.1,
                scale: 2.0,
                ..DistanceSpreadParams::default()
            },
        );
        let direct = gpu.read_rgba8(pool.texture("output_direct").unwrap(), RESOLUTION);
        let scaled = gpu.read_rgba8(pool.texture("output_scaled").unwrap(), RESOLUTION);
        assert_eq!(
            direct, scaled,
            "distance=0.2,scale=1.0 should match distance=0.1,scale=2.0 pixel for pixel"
        );
    }
}
