// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Pattern Generator.
//!
//! Display name: "Pattern Generator". Tooltip: "A brick or grid pattern —
//! use Offset 0.5 for a classic running-bond brick layout."
//!
//! For the brick stagger to tile seamlessly top-to-bottom, `rows` needs to
//! be even — see the shader's doc comment.

use crate::shaders::full_source;

const NODE_KIND: &str = "tile_generator";

/// Which grid axes actually get a tile drawn — not the shared `tiling::
/// TilingMode` (wrap-vs-clip at a sampled edge), a genuinely different
/// concept: this governs the tile layout itself. `Both` is the standard
/// full `rows` x `columns` grid; `HorizontalOnly`/`VerticalOnly` draw a
/// single centered row/column of tiles at their normal (unstretched) size,
/// leaving the rest of the canvas empty — e.g. distributing a connected
/// Shape across one row or column instead of a full grid. No `Off` variant:
/// "neither axis" has no sensible meaning for a pattern generator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PatternDistribution {
    #[default]
    Both,
    HorizontalOnly,
    VerticalOnly,
}

impl PatternDistribution {
    pub const NAMES: [&'static str; 3] = ["Both (Seamless)", "Horizontal Only", "Vertical Only"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Both (Seamless)" => Some(Self::Both),
            "Horizontal Only" => Some(Self::HorizontalOnly),
            "Vertical Only" => Some(Self::VerticalOnly),
            _ => None,
        }
    }

    fn as_u32(self) -> u32 {
        match self {
            Self::Both => 0,
            Self::HorizontalOnly => 1,
            Self::VerticalOnly => 2,
        }
    }
}

/// Which generation mode the built-in tile shape uses — a real, ongoing mode
/// the shader checks every frame (unlike `LAYOUT_PRESETS` below, a one-shot
/// action that just sets `offset`). Only applies with no Shape connected —
/// same restriction `gap`/`bevel` already have, since Herringbone needs its
/// own specific rotated-plank geometry, not an arbitrary connected shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PatternLayout {
    #[default]
    Grid,
    Herringbone,
}

impl PatternLayout {
    pub const NAMES: [&'static str; 2] = ["Grid", "Herringbone"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Grid" => Some(Self::Grid),
            "Herringbone" => Some(Self::Herringbone),
            _ => None,
        }
    }

    fn as_u32(self) -> u32 {
        match self {
            Self::Grid => 0,
            Self::Herringbone => 1,
        }
    }
}

/// How overlapping tiles combine once a negative Gap grows them past their
/// own cell — the owner's own request, after a negative Gap alone turned
/// out to just saturate each tile to its own cell's full extent once its
/// own half-size passed 0.5 (every pixel only ever tested its own cell's
/// rect, so there was nothing *to* blend with a neighbor). Only has a
/// visible effect once Gap goes negative — with Gap >= 0 no tile ever grows
/// past its own cell, so no neighbor cell can ever reach a given pixel, and
/// `dispatch`'s own neighbor search finds nothing to blend regardless of
/// this mode. Grid layout only (see `herringbone_value`'s own doc comment
/// on why Herringbone's lattice doesn't extend this same way); no effect
/// once something's wired into Shape, same restriction Gap/Bevel already
/// have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OverlapBlendMode {
    Add,
    #[default]
    Max,
    /// Substance Designer's own "Add Sub": values are signed deltas
    /// centered on mid-gray (0.5) rather than a plain additive layer —
    /// above 0.5 raises/lightens the base, below 0.5 lowers/darkens it.
    AddSub,
}

impl OverlapBlendMode {
    pub const NAMES: [&'static str; 3] = ["Add", "Max", "Add Sub"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Add" => Some(Self::Add),
            "Max" => Some(Self::Max),
            "Add Sub" => Some(Self::AddSub),
            _ => None,
        }
    }

    fn as_u32(self) -> u32 {
        match self {
            Self::Add => 0,
            Self::Max => 1,
            Self::AddSub => 2,
        }
    }
}

/// A one-click starting `offset`/`gap` for a common masonry stagger — the
/// same one-shot, no-lasting-identity idea as Thread's own `PRESETS`:
/// picking one just sets those two params, which stay normal editable
/// sliders afterward. Not to be confused with `PatternLayout` above —
/// that's an ongoing generation mode, this is a param shortcut, hence the
/// "Stagger Preset" label in the UI rather than reusing the name "Layout".
pub struct LayoutPreset {
    pub name: &'static str,
    pub offset: f32,
    /// A small nonzero default (rather than 0) so Stack Bond's plain grid
    /// doesn't render visually identical to Grid's own zero-gap default —
    /// the whole point of picking a preset is to see the difference.
    pub gap: f32,
}

pub const LAYOUT_PRESETS: &[LayoutPreset] = &[
    LayoutPreset {
        name: "Running Bond (Brick)",
        offset: 0.5,
        gap: 0.04,
    },
    LayoutPreset {
        name: "Stack Bond (Grid)",
        offset: 0.0,
        gap: 0.04,
    },
];

pub struct TileGeneratorParams {
    pub rows: u32,
    pub columns: u32,
    pub seed: u32,
    pub offset: f32,
    pub gap: f32,
    /// Shown to the artist as "Edge Darkening" (see `defs.rs`) — a flat
    /// per-tile luminance fade toward the gap, not a height-shaping bevel.
    /// Field/param-key name left as `bevel` for `.mcgraph` compatibility.
    pub bevel: f32,
    pub size_jitter: f32,
    pub rotation_jitter: f32,
    pub luminance_jitter: f32,
    /// Independent of `seed` — re-rolling this alone reshuffles per-tile
    /// brightness without also reshuffling size/rotation jitter, unlike the
    /// shared `seed` (whose own dice button randomizes every param on the
    /// node). The owner's own request. `from_graph` falls back to `seed`
    /// itself (not a fixed default) when this key is missing from an older
    /// saved graph, so an existing file's luminance pattern — previously
    /// always derived from `seed` — renders bit-identical to before this
    /// param existed.
    pub luminance_seed: u32,
    pub distribution: PatternDistribution,
    /// Only used when a Shape is connected — see the shader's doc comment.
    pub shape_size: f32,
    /// Only takes effect with no Shape connected — see `PatternLayout`'s own
    /// doc comment. `columns` sets this layout's own baseline density
    /// (`rows` is hidden — see `param_is_relevant` — and ignored, see
    /// `pack_params`); `plank_length`/`plank_width`'s ratio adjusts it
    /// further so pieces meet edge to edge at any proportion, rather than
    /// leaving a gap — see the shader's own doc comment.
    pub layout: PatternLayout,
    /// Herringbone only — the plank's relative length; only the ratio to
    /// `plank_width` matters (clamped to a safe range — see the shader's
    /// own `HERRINGBONE_ASPECT_MIN`/`_MAX`), not either value alone.
    /// Changing it re-tiles the whole pattern with more or fewer,
    /// differently-proportioned pieces, not a fixed-size plank shrinking
    /// within unchanged slot (that was an earlier, discarded version).
    pub plank_length: f32,
    /// Herringbone only — same as `plank_length`, for the plank's width.
    pub plank_width: f32,
    /// Grid layout only — see `OverlapBlendMode`'s own doc comment.
    pub overlap_blend_mode: OverlapBlendMode,
    /// How much of the overlap blend shows, 0..1 — a plain `mix` between
    /// "ignore any overlap, show just this cell's own value" (0) and "the
    /// full blended result" (1). One shared opacity for every mode, rather
    /// than baking a separate strength into each mode's own math.
    pub overlap_opacity: f32,
    /// Per-tile random offset from its own cell center, 0..1. Full-grid
    /// (`PatternDistribution::Both`) only — same scope restriction as
    /// `rotation_jitter`, and the same reason: a coherent "position" only
    /// means the same thing on both axes when they repeat at the same
    /// scale.
    pub position_jitter: f32,
    /// Independent of `seed` — same idea as `luminance_seed`.
    pub position_seed: u32,
    /// How strongly a connected Distribution Map input biases which tiles
    /// survive — 0 ignores the map, 1 makes a tile's own odds of appearing
    /// exactly match the map's brightness at its own cell. Only has any
    /// effect once something's wired into the `distribution_map` input.
    pub distribution_amount: f32,
    /// Independent of `seed` — same idea as `luminance_seed`, for the
    /// Distribution Map gate's own per-tile random draw.
    pub distribution_seed: u32,
    /// Chance each tile is independently mirrored per axis, 0..1 — only
    /// has a visible effect once something's wired into `shape`; mirroring
    /// the built-in rect is a mathematical no-op (see the shader's own
    /// `shape_cell_value` doc comment).
    pub mirror_jitter: f32,
    /// Independent of `seed` — same idea as `luminance_seed`.
    pub mirror_seed: u32,
}

impl Default for TileGeneratorParams {
    // NOTE: these are the *legacy-graph* fallback values for params missing
    // from an older `.mcgraph` file saved before they existed (`from_graph`
    // falls back here per-key) — not the same thing as a freshly-added
    // node's own UI defaults (`defs.rs`'s `ParamSpec::default`, used by
    // `NodeDef::instantiate`, which a new node always gets explicit stored
    // values from). The two happen to have matched historically, but
    // deliberately kept independent now: changing a *new* node's default
    // shouldn't silently change how an existing saved example (e.g.
    // `examples/graphs/fabric.mcgraph`, which predates size_jitter/
    // luminance_jitter/shape_size) already renders. `shape_size` only ever
    // matters once a Shape is connected, which `fabric.mcgraph`'s own
    // tile_generator node never does — so its fallback value here was
    // safe to correct alongside `shape_size`'s own semantic fix (tile-
    // absolute, not cell-relative; see `defs.rs`'s own doc comment) rather
    // than frozen at the old (now-wrong-units) value like the others above.
    fn default() -> Self {
        Self {
            rows: 6,
            columns: 4,
            seed: 0,
            offset: 0.5,
            gap: 0.06,
            bevel: 0.08,
            size_jitter: 0.08,
            rotation_jitter: 0.0,
            luminance_jitter: 0.15,
            luminance_seed: 0,
            distribution: PatternDistribution::default(),
            shape_size: 0.1,
            layout: PatternLayout::default(),
            plank_length: 2.0,
            plank_width: 1.0,
            overlap_blend_mode: OverlapBlendMode::default(),
            overlap_opacity: 1.0,
            position_jitter: 0.0,
            position_seed: 0,
            distribution_amount: 1.0,
            distribution_seed: 0,
            mirror_jitter: 0.0,
            mirror_seed: 0,
        }
    }
}

/// `rows`/`columns` as actually used for a dispatch: unchanged normally (the
/// built-in tile *wants* an asymmetric grid — e.g. the classic 6-row/4-column
/// brick default), but forced square (both set to whichever was larger) once
/// a Shape is connected (matching what the Parameters panel's own linked
/// "Tile Count" slider already implies to the artist — without this, a
/// freshly-wired Shape renders stretched into whatever asymmetric grid the
/// built-in tile happened to default to, before the artist ever touches that
/// slider) or once Herringbone layout is selected (its own seamless-tiling
/// math needs a single integer repeat count, not independent row/column
/// counts — see the shader's own doc comment).
fn effective_grid(rows: u32, columns: u32, force_square: bool) -> (u32, u32) {
    if force_square {
        let n = rows.max(columns);
        (n, n)
    } else {
        (rows, columns)
    }
}

impl TileGeneratorParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        let float = |name: &str, default: f32| {
            params
                .get(name)
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(default)
        };
        let seed = params
            .get("seed")
            .and_then(mc_core::ParamValue::as_seed)
            .unwrap_or(defaults.seed);
        Self {
            rows: params
                .get("rows")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.rows),
            columns: params
                .get("columns")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(1) as u32)
                .unwrap_or(defaults.columns),
            seed,
            offset: float("offset", defaults.offset),
            gap: float("gap", defaults.gap),
            bevel: float("bevel", defaults.bevel),
            size_jitter: float("size_jitter", defaults.size_jitter),
            rotation_jitter: float("rotation_jitter", defaults.rotation_jitter),
            luminance_jitter: float("luminance_jitter", defaults.luminance_jitter),
            // Falls back to `seed` itself (not `defaults.luminance_seed`,
            // always 0) when missing from an older saved graph — see this
            // field's own doc comment on the struct.
            luminance_seed: params
                .get("luminance_seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(seed),
            distribution: params
                .get("tiling_mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => PatternDistribution::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.distribution),
            shape_size: float("shape_size", defaults.shape_size),
            layout: params
                .get("layout")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => PatternLayout::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.layout),
            plank_length: float("plank_length", defaults.plank_length),
            plank_width: float("plank_width", defaults.plank_width),
            overlap_blend_mode: params
                .get("overlap_blend_mode")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => OverlapBlendMode::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.overlap_blend_mode),
            overlap_opacity: float("overlap_opacity", defaults.overlap_opacity),
            position_jitter: float("position_jitter", defaults.position_jitter),
            // Falls back to `seed` itself for an older saved graph missing
            // this key — same convention as `luminance_seed`.
            position_seed: params
                .get("position_seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(seed),
            distribution_amount: float("distribution_amount", defaults.distribution_amount),
            distribution_seed: params
                .get("distribution_seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(seed),
            mirror_jitter: float("mirror_jitter", defaults.mirror_jitter),
            mirror_seed: params
                .get("mirror_seed")
                .and_then(mc_core::ParamValue::as_seed)
                .unwrap_or(seed),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    shape_key: Option<&str>,
    distribution_map_key: Option<&str>,
    output_key: &str,
    resolution: u32,
    params: &TileGeneratorParams,
) {
    // No custom Shape wired in: bind a throwaway 1x1 texture that the
    // shader never actually reads (`has_shape` is 0, so it stays on the
    // built-in rounded-rect path) — wgpu still needs *something* bound at
    // that slot regardless.
    let placeholder_key = format!("{output_key}__unused_shape_placeholder");
    let has_shape = shape_key.is_some();
    let shape_key = shape_key.unwrap_or_else(|| {
        gpu.fill_rgba8(pool, &placeholder_key, 1, [0, 0, 0, 255]);
        &placeholder_key
    });
    let shape_view = pool
        .view(shape_key)
        .unwrap_or_else(|| panic!("tile_generator: no texture at shape key {shape_key:?}"))
        .clone();

    // Same throwaway-texture convention as Shape above, for the same
    // reason — `has_distribution_map` is 0 whenever nothing's connected, so
    // the shader never actually reads this binding in that case.
    let distribution_placeholder_key = format!("{output_key}__unused_distribution_placeholder");
    let has_distribution_map = distribution_map_key.is_some();
    let distribution_map_key = distribution_map_key.unwrap_or_else(|| {
        gpu.fill_rgba8(pool, &distribution_placeholder_key, 1, [255, 255, 255, 255]);
        &distribution_placeholder_key
    });
    let distribution_map_view = pool
        .view(distribution_map_key)
        .unwrap_or_else(|| {
            panic!("tile_generator: no texture at distribution map key {distribution_map_key:?}")
        })
        .clone();

    let shader_source = full_source("tile_generator.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params, has_shape, has_distribution_map);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile_generator_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let view = pool.get_or_create(
        &gpu.device,
        output_key,
        resolution,
        wgpu::TextureFormat::Rgba16Float,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tile_generator_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&shape_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&distribution_map_view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tile_generator_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile_generator_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

fn pack_params(
    resolution: u32,
    params: &TileGeneratorParams,
    has_shape: bool,
    has_distribution_map: bool,
) -> [u8; 112] {
    // Herringbone hides Rows entirely (see `param_is_relevant`) and derives
    // its own density from Columns alone — using `rows.max(columns)` here
    // like the Shape-connected case would let a stale, no-longer-visible
    // Rows value silently affect the density the artist can't see or
    // change.
    let (rows, columns) = if params.layout == PatternLayout::Herringbone {
        (params.columns, params.columns)
    } else {
        effective_grid(params.rows, params.columns, has_shape)
    };
    let mut bytes = [0u8; 112];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&rows.to_le_bytes());
    bytes[8..12].copy_from_slice(&columns.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.seed.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.offset.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.gap.to_le_bytes());
    bytes[24..28].copy_from_slice(&params.bevel.to_le_bytes());
    bytes[28..32].copy_from_slice(&params.size_jitter.to_le_bytes());
    bytes[32..36].copy_from_slice(&params.rotation_jitter.to_le_bytes());
    bytes[36..40].copy_from_slice(&params.luminance_jitter.to_le_bytes());
    bytes[40..44].copy_from_slice(&params.distribution.as_u32().to_le_bytes());
    bytes[44..48].copy_from_slice(&(has_shape as u32).to_le_bytes());
    bytes[48..52].copy_from_slice(&params.shape_size.to_le_bytes());
    bytes[52..56].copy_from_slice(&params.layout.as_u32().to_le_bytes());
    bytes[56..60].copy_from_slice(&params.plank_length.to_le_bytes());
    bytes[60..64].copy_from_slice(&params.plank_width.to_le_bytes());
    bytes[64..68].copy_from_slice(&params.luminance_seed.to_le_bytes());
    bytes[68..72].copy_from_slice(&params.overlap_blend_mode.as_u32().to_le_bytes());
    bytes[72..76].copy_from_slice(&params.overlap_opacity.to_le_bytes());
    bytes[76..80].copy_from_slice(&params.position_jitter.to_le_bytes());
    bytes[80..84].copy_from_slice(&params.position_seed.to_le_bytes());
    bytes[84..88].copy_from_slice(&params.distribution_amount.to_le_bytes());
    bytes[88..92].copy_from_slice(&params.distribution_seed.to_le_bytes());
    bytes[92..96].copy_from_slice(&(has_distribution_map as u32).to_le_bytes());
    bytes[96..100].copy_from_slice(&params.mirror_jitter.to_le_bytes());
    bytes[100..104].copy_from_slice(&params.mirror_seed.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{
        dispatch, effective_grid, OverlapBlendMode, PatternDistribution, PatternLayout,
        TileGeneratorParams, LAYOUT_PRESETS,
    };
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    /// "Running Bond" and "Stack Bond" are just named shortcuts for
    /// `offset` values already proven to tile seamlessly by
    /// `tiles_seamlessly_with_default_even_rows`/`tiles_seamlessly_with_no_stagger`
    /// above — this only guards against a typo'd/duplicated preset.
    #[test]
    fn layout_presets_have_distinct_names_and_offsets() {
        assert_eq!(LAYOUT_PRESETS.len(), 2);
        assert_ne!(LAYOUT_PRESETS[0].name, LAYOUT_PRESETS[1].name);
        assert_ne!(LAYOUT_PRESETS[0].offset, LAYOUT_PRESETS[1].offset);
    }

    #[test]
    fn effective_grid_is_unchanged_with_no_shape_connected() {
        assert_eq!(effective_grid(6, 4, false), (6, 4));
    }

    #[test]
    fn effective_grid_is_forced_square_once_a_shape_is_connected() {
        assert_eq!(effective_grid(6, 4, true), (6, 6));
        assert_eq!(effective_grid(4, 6, true), (6, 6));
        assert_eq!(effective_grid(5, 5, true), (5, 5));
    }

    #[test]
    fn every_distribution_name_round_trips() {
        for name in PatternDistribution::NAMES {
            assert!(
                PatternDistribution::from_name(name).is_some(),
                "{name:?} should parse"
            );
        }
    }

    fn render(params: &TileGeneratorParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, None, None, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("tile_generator texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    /// Default params use an even row count — see the node's own doc comment
    /// on why that's required for the brick stagger to tile vertically.
    #[test]
    fn tiles_seamlessly_with_default_even_rows() {
        assert_tiles_seamlessly(&render(&TileGeneratorParams::default()), RESOLUTION);
    }

    #[test]
    fn tiles_seamlessly_with_no_stagger() {
        let params = TileGeneratorParams {
            offset: 0.0,
            ..TileGeneratorParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    /// Herringbone's own domino lattice is seamless by construction for any
    /// `columns` value (see `herringbone_value`'s own doc comment in the
    /// shader for the exact derivation) — this is the numeric confirmation
    /// of that, not something tuned to pass for this one case only. Offset
    /// below 0.25 (Stack Bond territory) selects the 45\u{b0} diagonal variant.
    #[test]
    fn tiles_seamlessly_herringbone_diagonal() {
        let params = TileGeneratorParams {
            layout: PatternLayout::Herringbone,
            offset: 0.0,
            ..TileGeneratorParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    /// Offset 0.25 and above (Running Bond territory) selects the
    /// axis-aligned variant instead — needs `columns` even for the
    /// checkerboard parity to land back in sync at the seam (see the
    /// shader's own doc comment); the default's forced-square `columns`
    /// (`effective_grid`, `rows.max(columns)` = 6) already satisfies this.
    #[test]
    fn tiles_seamlessly_herringbone_axis_aligned() {
        let params = TileGeneratorParams {
            layout: PatternLayout::Herringbone,
            offset: 0.5,
            ..TileGeneratorParams::default()
        };
        assert_tiles_seamlessly(&render(&params), RESOLUTION);
    }

    /// Herringbone should actually look like herringbone, not silently fall
    /// through to the Grid layout's own built-in rect.
    #[test]
    fn herringbone_differs_from_grid() {
        let grid = render(&TileGeneratorParams::default());
        let herringbone = render(&TileGeneratorParams {
            layout: PatternLayout::Herringbone,
            offset: 0.0,
            ..TileGeneratorParams::default()
        });
        assert_ne!(grid, herringbone);
    }

    /// The two Offset-selected orientations (diagonal vs. axis-aligned)
    /// should actually look different from each other, not silently
    /// collapse to the same geometry.
    #[test]
    fn herringbone_diagonal_differs_from_axis_aligned() {
        let shared = TileGeneratorParams {
            layout: PatternLayout::Herringbone,
            ..TileGeneratorParams::default()
        };
        let diagonal = render(&TileGeneratorParams {
            offset: 0.0,
            ..shared
        });
        let axis_aligned = render(&TileGeneratorParams {
            offset: 0.5,
            ..shared
        });
        assert_ne!(diagonal, axis_aligned);
    }

    /// Rows is hidden for Herringbone (see `param_is_relevant`) and must
    /// have zero effect on the render — otherwise a stale leftover Rows
    /// value (e.g. from before switching Layout) would silently change
    /// density the artist can no longer see or control.
    #[test]
    fn herringbone_ignores_rows() {
        let base = TileGeneratorParams {
            layout: PatternLayout::Herringbone,
            rows: 6,
            columns: 11,
            ..TileGeneratorParams::default()
        };
        let different_rows = TileGeneratorParams { rows: 30, ..base };
        assert_eq!(render(&base), render(&different_rows));
    }

    /// Changing Plank Length/Width's ratio should re-tile with more/fewer,
    /// differently-proportioned pieces that still meet edge to edge — not
    /// leave a large empty gap, which an earlier version (treating them as
    /// fill-fractions of an otherwise-fixed slot) actually did. Coverage
    /// (the fraction of non-background pixels) should stay roughly the same
    /// across very different length:width ratios, not collapse because
    /// pieces got smaller without the slot count increasing to compensate.
    #[test]
    fn plank_ratio_does_not_leave_a_large_gap() {
        let coverage = |plank_length: f32, plank_width: f32| -> f64 {
            let params = TileGeneratorParams {
                layout: PatternLayout::Herringbone,
                plank_length,
                plank_width,
                gap: 0.0,
                bevel: 0.0001,
                size_jitter: 0.0,
                ..TileGeneratorParams::default()
            };
            let pixels = render(&params);
            let filled = pixels.iter().step_by(4).filter(|&&v| v > 10).count();
            filled as f64 / (pixels.len() / 4) as f64
        };

        let classic = coverage(2.0, 1.0); // aspect 2.0, the classic proportion
        let elongated = coverage(4.0, 1.0); // aspect 4.0, much longer planks
        assert!(
            (classic - elongated).abs() < 0.15,
            "coverage should stay roughly consistent across ratios, not collapse into a gap: \
             classic={classic:.3}, elongated={elongated:.3}"
        );
    }

    /// Plank Width should now be a genuinely continuous control (no longer
    /// snapped to ~5 whole-number aspect ratios) — two nearby values should
    /// render differently, not collapse to the same rendered result.
    #[test]
    fn plank_width_is_continuous_not_snapped() {
        let base = TileGeneratorParams {
            layout: PatternLayout::Herringbone,
            plank_width: 0.52,
            ..TileGeneratorParams::default()
        };
        let nearby = TileGeneratorParams {
            plank_width: 0.58,
            ..base
        };
        assert_ne!(render(&base), render(&nearby));
    }

    /// Plank Width should actually reach the rendered pixels (more/narrower
    /// planks per repeat unit), not be plumbing that's silently ignored.
    #[test]
    fn plank_width_changes_herringbone_output() {
        let base = TileGeneratorParams {
            layout: PatternLayout::Herringbone,
            ..TileGeneratorParams::default()
        };
        let narrower = TileGeneratorParams {
            plank_width: 0.25,
            ..base
        };
        assert_ne!(render(&base), render(&narrower));
    }

    #[test]
    fn herringbone_is_deterministic() {
        let params = TileGeneratorParams {
            layout: PatternLayout::Herringbone,
            seed: 3,
            ..TileGeneratorParams::default()
        };
        assert_eq!(render(&params), render(&params));
    }

    #[test]
    fn is_deterministic() {
        let params = TileGeneratorParams {
            seed: 3,
            ..TileGeneratorParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same params produced different pixels"
        );
    }

    #[test]
    fn seed_changes_output() {
        // Needs non-zero jitter — the seed only ever reaches rendered
        // pixels through Size/Rotation/Luminance Jitter, all zero at this
        // node's own defaults.
        let a = render(&TileGeneratorParams {
            seed: 1,
            size_jitter: 0.08,
            luminance_jitter: 0.15,
            ..TileGeneratorParams::default()
        });
        let b = render(&TileGeneratorParams {
            seed: 2,
            size_jitter: 0.08,
            luminance_jitter: 0.15,
            ..TileGeneratorParams::default()
        });
        assert_ne!(a, b, "different seeds produced identical pixels");
    }

    /// The owner's own request: Luminance Seed re-rolls just the per-tile
    /// brightness, independent of the shared Seed.
    #[test]
    fn luminance_seed_changes_output_independently_of_seed() {
        let a = render(&TileGeneratorParams {
            seed: 1,
            luminance_seed: 1,
            luminance_jitter: 0.5,
            ..TileGeneratorParams::default()
        });
        let b = render(&TileGeneratorParams {
            seed: 1,
            luminance_seed: 2,
            luminance_jitter: 0.5,
            ..TileGeneratorParams::default()
        });
        assert_ne!(
            a, b,
            "different Luminance Seeds (same Seed) produced identical pixels"
        );
    }

    /// The flip side: with Luminance Jitter at zero, Luminance Seed has
    /// nothing to drive, so it must have no effect on Size/Rotation Jitter
    /// (still keyed off the shared Seed, untouched here).
    #[test]
    fn luminance_seed_does_not_affect_size_or_rotation_jitter() {
        let a = render(&TileGeneratorParams {
            seed: 1,
            luminance_seed: 1,
            size_jitter: 0.3,
            rotation_jitter: 0.3,
            luminance_jitter: 0.0,
            ..TileGeneratorParams::default()
        });
        let b = render(&TileGeneratorParams {
            seed: 1,
            luminance_seed: 99,
            size_jitter: 0.3,
            rotation_jitter: 0.3,
            luminance_jitter: 0.0,
            ..TileGeneratorParams::default()
        });
        assert_eq!(
            a, b,
            "Luminance Seed alone changed Size/Rotation Jitter's own output"
        );
    }

    /// An older saved graph has no `luminance_seed` key at all (added after
    /// this param existed) — falling back to `seed` itself there keeps its
    /// luminance pattern bit-identical to before this param existed, when
    /// luminance was always derived from `seed + 211u` directly.
    #[test]
    fn from_graph_falls_back_to_seed_for_missing_luminance_seed() {
        let mut params = std::collections::BTreeMap::new();
        params.insert("seed".to_string(), mc_core::ParamValue::Seed(7));
        let parsed = TileGeneratorParams::from_graph(&params);
        assert_eq!(parsed.luminance_seed, 7);

        let mut params = std::collections::BTreeMap::new();
        params.insert("seed".to_string(), mc_core::ParamValue::Seed(7));
        params.insert("luminance_seed".to_string(), mc_core::ParamValue::Seed(42));
        let parsed = TileGeneratorParams::from_graph(&params);
        assert_eq!(
            parsed.luminance_seed, 42,
            "an explicit luminance_seed key should win over the seed fallback"
        );
    }

    /// The owner's own request: a negative Gap grows each tile past its own
    /// cell instead of shrinking it, overlapping neighbors — confirmed by a
    /// large-enough negative value fully closing the mortar seam (no
    /// near-black pixels left) rather than crashing or leaving a blank
    /// output.
    #[test]
    fn negative_gap_overlaps_neighbors_instead_of_crashing() {
        let out = render(&TileGeneratorParams {
            gap: -0.2,
            ..TileGeneratorParams::default()
        });
        assert!(
            out.iter().any(|&v| v > 0),
            "negative gap produced a fully blank output"
        );
        assert!(
            !out.as_chunks::<4>().0.iter().any(|px| px[0] < 10),
            "gap=-0.2 should fully close the mortar seam, found a near-black pixel left"
        );
    }

    /// A heavily negative Gap plus real per-cell luminance variation
    /// (Luminance Jitter > 0) gives overlapping tiles genuinely different
    /// values to combine — enough for Add vs Max to disagree (Add can
    /// exceed either input's own value; Max never does).
    fn overlap_test_params(mode: OverlapBlendMode, opacity: f32) -> TileGeneratorParams {
        TileGeneratorParams {
            gap: -0.3,
            luminance_jitter: 1.0,
            seed: 3,
            overlap_blend_mode: mode,
            overlap_opacity: opacity,
            ..TileGeneratorParams::default()
        }
    }

    #[test]
    fn overlap_blend_mode_changes_output_when_tiles_overlap() {
        let add = render(&overlap_test_params(OverlapBlendMode::Add, 1.0));
        let max = render(&overlap_test_params(OverlapBlendMode::Max, 1.0));
        let add_sub = render(&overlap_test_params(OverlapBlendMode::AddSub, 1.0));
        assert_ne!(add, max, "Add and Max produced identical pixels");
        assert_ne!(add, add_sub, "Add and Add Sub produced identical pixels");
        assert_ne!(max, add_sub, "Max and Add Sub produced identical pixels");
    }

    /// `value = mix(center_v, blended, opacity)` — at opacity 0 the blend
    /// mode is never consulted at all, so every mode must agree exactly.
    #[test]
    fn overlap_opacity_zero_ignores_the_blend_mode_entirely() {
        let add = render(&overlap_test_params(OverlapBlendMode::Add, 0.0));
        let max = render(&overlap_test_params(OverlapBlendMode::Max, 0.0));
        let add_sub = render(&overlap_test_params(OverlapBlendMode::AddSub, 0.0));
        assert_eq!(add, max, "opacity 0 should make Add and Max agree exactly");
        assert_eq!(
            add, add_sub,
            "opacity 0 should make Add and Add Sub agree exactly"
        );
    }

    /// With Gap >= 0 no tile can ever grow past its own cell, so no
    /// neighbor can ever reach a pixel the base cell doesn't already cover
    /// — the new overlap search must be a complete no-op there, regardless
    /// of mode/opacity, matching this node's behavior before it existed.
    #[test]
    fn overlap_blend_has_no_effect_when_gap_is_not_negative() {
        let base_params = TileGeneratorParams {
            gap: 0.06,
            luminance_jitter: 1.0,
            seed: 3,
            ..TileGeneratorParams::default()
        };
        let default_mode = render(&base_params);
        let add_full_opacity = render(&TileGeneratorParams {
            overlap_blend_mode: OverlapBlendMode::Add,
            overlap_opacity: 1.0,
            ..base_params
        });
        assert_eq!(
            default_mode, add_full_opacity,
            "a non-negative Gap should make Overlap Blend/Opacity irrelevant"
        );
    }

    /// Renders a connected disc Shape with the given distribution, no
    /// stagger/jitter (keeps the geometry easy to reason about: a plain
    /// 4x4 grid of discs, centered row/column index 2).
    fn render_distribution(distribution: PatternDistribution) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc",
            RESOLUTION,
            &crate::shape::ShapeParams::default(),
        );
        dispatch(
            &mut gpu,
            &mut pool,
            Some("disc"),
            None,
            "test",
            RESOLUTION,
            &TileGeneratorParams {
                rows: 4,
                columns: 4,
                offset: 0.0,
                distribution,
                ..TileGeneratorParams::default()
            },
        );
        gpu.read_rgba8(pool.texture("test").unwrap(), RESOLUTION)
    }

    /// A disc never quite touches the edge of its own cell, so checking a
    /// single pixel row/column is fragile — this checks the *whole band* of
    /// pixel rows (or columns) a grid row (or column) index covers, 4x4
    /// grid over `RESOLUTION`.
    fn row_band_is_blank(pixels: &[u8], resolution: u32, grid_rows: u32, row_index: u32) -> bool {
        let y0 = row_index * resolution / grid_rows;
        let y1 = (row_index + 1) * resolution / grid_rows;
        (y0..y1).all(|y| (0..resolution).all(|x| pixels[((y * resolution + x) * 4) as usize] == 0))
    }

    fn column_band_is_blank(
        pixels: &[u8],
        resolution: u32,
        grid_columns: u32,
        col_index: u32,
    ) -> bool {
        let x0 = col_index * resolution / grid_columns;
        let x1 = (col_index + 1) * resolution / grid_columns;
        (x0..x1).all(|x| (0..resolution).all(|y| pixels[((y * resolution + x) * 4) as usize] == 0))
    }

    /// "Horizontal Only" means "turn this into vertical stripes": the row
    /// axis doesn't exist at all, every pixel shows the connected Shape's
    /// own full, uncropped canvas height — not squeezed into a small
    /// centered band (the earlier, now-reverted behavior reported as "a
    /// deliberately tall Shape still renders as small centered rectangles,
    /// not full-height bands"). `render_distribution`'s disc uses the Shape
    /// node's own default `Size` (0.6, see `shape::ShapeParams::default`)
    /// — half-extent 0.3 around its own center, so it spans ~0.2..0.8 of
    /// its canvas. Shown at full (1:1) scale, that's the window this test
    /// checks against, with a little margin for anti-aliasing softness.
    #[test]
    fn horizontal_only_shows_content_across_the_full_height() {
        let pixels = render_distribution(PatternDistribution::HorizontalOnly);
        let white_rows: Vec<u32> = (0..RESOLUTION)
            .filter(|&y| (0..RESOLUTION).any(|x| pixels[((y * RESOLUTION + x) * 4) as usize] > 10))
            .collect();
        let first = *white_rows.first().expect("some row should show content");
        let last = *white_rows.last().unwrap();
        assert!(
            first < RESOLUTION * 3 / 10,
            "content should reach ~0.2 up the canvas (the disc's own extent), \
             first visible row was {first}"
        );
        assert!(
            last > RESOLUTION - RESOLUTION * 3 / 10,
            "content should reach ~0.8 down the canvas (the disc's own extent), \
             last visible row was {last}"
        );
    }

    /// The mirror of the test above, for the repeating (column) axis:
    /// Horizontal Only still repeats across columns, so there should be
    /// real gaps (blank columns) between the stripes, not one solid band.
    #[test]
    fn horizontal_only_still_repeats_with_gaps_across_columns() {
        let pixels = render_distribution(PatternDistribution::HorizontalOnly);
        let any_column_blank = (0..RESOLUTION)
            .any(|x| (0..RESOLUTION).all(|y| pixels[((y * RESOLUTION + x) * 4) as usize] == 0));
        assert!(
            any_column_blank,
            "expected at least one fully blank column between repeated stripes"
        );
    }

    /// Mirror of the Horizontal Only test above, same reasoning (the
    /// disc's own ~0.2..0.8 extent, shown at full 1:1 scale).
    #[test]
    fn vertical_only_shows_content_across_the_full_width() {
        let pixels = render_distribution(PatternDistribution::VerticalOnly);
        let white_cols: Vec<u32> = (0..RESOLUTION)
            .filter(|&x| (0..RESOLUTION).any(|y| pixels[((y * RESOLUTION + x) * 4) as usize] > 10))
            .collect();
        let first = *white_cols.first().expect("some column should show content");
        let last = *white_cols.last().unwrap();
        assert!(
            first < RESOLUTION * 3 / 10,
            "content should reach ~0.2 across the canvas (the disc's own extent), \
             first visible column was {first}"
        );
        assert!(
            last > RESOLUTION - RESOLUTION * 3 / 10,
            "content should reach ~0.8 across the canvas (the disc's own extent), \
             last visible column was {last}"
        );
    }

    /// The mirror of the horizontal repeats-with-gaps test above, for
    /// Vertical Only's repeating (row) axis.
    #[test]
    fn vertical_only_still_repeats_with_gaps_across_rows() {
        let pixels = render_distribution(PatternDistribution::VerticalOnly);
        let any_row_blank = (0..RESOLUTION)
            .any(|y| (0..RESOLUTION).all(|x| pixels[((y * RESOLUTION + x) * 4) as usize] == 0));
        assert!(
            any_row_blank,
            "expected at least one fully blank row between repeated stripes"
        );
    }

    /// Regression test: `rows`/`columns` = 6 (the real node's own default
    /// Tile Count) is even, which is exactly the case a grid-aligned
    /// "middle index" (e.g. `rows / 2`) lands half a cell off true center —
    /// reported as "neither the horizontal nor the vertical version are
    /// centered." Centering is done directly in UV space now (`uv == 0.5`),
    /// independent of whether the tile count is even or odd, so the visible
    /// band's margin above should match its margin below (and left should
    /// match right) within one pixel of rounding.
    #[test]
    fn horizontal_and_vertical_only_are_centered_with_an_even_tile_count() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc",
            RESOLUTION,
            &crate::shape::ShapeParams::default(),
        );
        let base = TileGeneratorParams {
            rows: 6,
            columns: 6,
            offset: 0.0,
            ..TileGeneratorParams::default()
        };

        dispatch(
            &mut gpu,
            &mut pool,
            Some("disc"),
            None,
            "horizontal",
            RESOLUTION,
            &TileGeneratorParams {
                distribution: PatternDistribution::HorizontalOnly,
                ..base
            },
        );
        let horizontal = gpu.read_rgba8(pool.texture("horizontal").unwrap(), RESOLUTION);
        let first_row = (0..RESOLUTION)
            .find(|&y| !row_band_is_blank(&horizontal, RESOLUTION, RESOLUTION, y))
            .expect("Horizontal Only rendered nothing at all");
        let last_row = (0..RESOLUTION)
            .rev()
            .find(|&y| !row_band_is_blank(&horizontal, RESOLUTION, RESOLUTION, y))
            .unwrap();
        let top_margin = first_row;
        let bottom_margin = RESOLUTION - 1 - last_row;
        assert!(
            top_margin.abs_diff(bottom_margin) <= 1,
            "Horizontal Only isn't centered: top margin {top_margin}, bottom margin {bottom_margin}"
        );

        dispatch(
            &mut gpu,
            &mut pool,
            Some("disc"),
            None,
            "vertical",
            RESOLUTION,
            &TileGeneratorParams {
                distribution: PatternDistribution::VerticalOnly,
                ..base
            },
        );
        let vertical = gpu.read_rgba8(pool.texture("vertical").unwrap(), RESOLUTION);
        let first_col = (0..RESOLUTION)
            .find(|&x| !column_band_is_blank(&vertical, RESOLUTION, RESOLUTION, x))
            .expect("Vertical Only rendered nothing at all");
        let last_col = (0..RESOLUTION)
            .rev()
            .find(|&x| !column_band_is_blank(&vertical, RESOLUTION, RESOLUTION, x))
            .unwrap();
        let left_margin = first_col;
        let right_margin = RESOLUTION - 1 - last_col;
        assert!(
            left_margin.abs_diff(right_margin) <= 1,
            "Vertical Only isn't centered: left margin {left_margin}, right margin {right_margin}"
        );
    }

    /// Regression test: staggering alternate rows sideways (`offset`) used
    /// to knock every other tile of a single centered column out of frame —
    /// reported as "the offset ... causes problems with the vertical only
    /// setting." `offset` should now be ignored entirely outside Both.
    #[test]
    fn offset_has_no_effect_outside_both() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc",
            RESOLUTION,
            &crate::shape::ShapeParams::default(),
        );
        for distribution in [
            PatternDistribution::HorizontalOnly,
            PatternDistribution::VerticalOnly,
        ] {
            dispatch(
                &mut gpu,
                &mut pool,
                Some("disc"),
                None,
                "no_offset",
                RESOLUTION,
                &TileGeneratorParams {
                    rows: 6,
                    columns: 6,
                    offset: 0.0,
                    distribution,
                    ..TileGeneratorParams::default()
                },
            );
            let no_offset = gpu.read_rgba8(pool.texture("no_offset").unwrap(), RESOLUTION);
            dispatch(
                &mut gpu,
                &mut pool,
                Some("disc"),
                None,
                "with_offset",
                RESOLUTION,
                &TileGeneratorParams {
                    rows: 6,
                    columns: 6,
                    offset: 0.5,
                    distribution,
                    ..TileGeneratorParams::default()
                },
            );
            let with_offset = gpu.read_rgba8(pool.texture("with_offset").unwrap(), RESOLUTION);
            assert_eq!(
                no_offset, with_offset,
                "{distribution:?}: Offset should have no effect outside Both"
            );
        }
    }

    #[test]
    fn both_fills_the_whole_grid_unlike_either_single_axis_mode() {
        let both = render_distribution(PatternDistribution::Both);
        assert!(
            !row_band_is_blank(&both, RESOLUTION, 4, 0),
            "Both should fill every grid row, including the top one"
        );
        assert!(
            !column_band_is_blank(&both, RESOLUTION, 4, 0),
            "Both should fill every grid column, including the left one"
        );
        assert_ne!(
            both,
            render_distribution(PatternDistribution::HorizontalOnly),
            "Both should differ from Horizontal Only"
        );
        assert_ne!(
            both,
            render_distribution(PatternDistribution::VerticalOnly),
            "Both should differ from Vertical Only"
        );
    }

    /// Regression test for a real bug: Horizontal/Vertical Only's visible
    /// strip thickness used to be computed from the per-axis cell count
    /// (`rows`/`columns`, linked to the same "Tile Count" slider a
    /// connected Shape's sampling itself already compensates for) — so the
    /// strip kept physically shrinking as Tile Count grew even though
    /// Shape Size never changed, reported as "the shape gets smaller and
    /// smaller with more Tile Count, even though I'm touching no scale
    /// parameter." The strip's own thickness must now track `shape_size`
    /// (the same absolute, Tile-Count-independent units the sampling
    /// already uses), not the grid density.
    #[test]
    fn horizontal_only_strip_thickness_is_independent_of_tile_count() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc",
            512,
            &crate::shape::ShapeParams::default(),
        );

        let mut strip_height_px = |tile_count: u32| -> usize {
            dispatch(
                &mut gpu,
                &mut pool,
                Some("disc"),
                None,
                "test",
                512,
                &TileGeneratorParams {
                    rows: tile_count,
                    columns: tile_count,
                    offset: 0.0,
                    shape_size: 0.2,
                    distribution: PatternDistribution::HorizontalOnly,
                    ..TileGeneratorParams::default()
                },
            );
            let pixels = gpu.read_rgba8(pool.texture("test").unwrap(), 512);
            let white_rows: Vec<usize> = (0..512)
                .filter(|&y| (0..512).any(|x| pixels[(y * 512 + x) * 4] > 10))
                .collect();
            white_rows.last().unwrap() - white_rows.first().unwrap()
        };

        let height_at_4 = strip_height_px(4);
        let height_at_16 = strip_height_px(16);
        let diff = height_at_4.abs_diff(height_at_16);
        assert!(
            diff <= 4,
            "strip height should stay ~constant across Tile Count at a fixed Shape Size: \
             at 4 tiles = {height_at_4}px, at 16 tiles = {height_at_16}px (diff {diff}px)"
        );
    }

    /// A connected Shape input should actually change the output (stamped
    /// in instead of the built-in rounded rect), not just be plumbing
    /// that's never read.
    #[test]
    fn a_connected_shape_changes_the_output() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let params = TileGeneratorParams::default();

        dispatch(
            &mut gpu, &mut pool, None, None, "built_in", RESOLUTION, &params,
        );
        let built_in = gpu.read_rgba8(pool.texture("built_in").unwrap(), RESOLUTION);

        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc",
            RESOLUTION,
            &crate::shape::ShapeParams::default(),
        );
        dispatch(
            &mut gpu,
            &mut pool,
            Some("disc"),
            None,
            "with_shape",
            RESOLUTION,
            &params,
        );
        let with_shape = gpu.read_rgba8(pool.texture("with_shape").unwrap(), RESOLUTION);

        assert_ne!(
            built_in, with_shape,
            "wiring in a custom shape should change Pattern Generator's output"
        );
    }

    #[test]
    fn shape_size_changes_how_much_of_the_cell_a_connected_shape_fills() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();

        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc",
            RESOLUTION,
            &crate::shape::ShapeParams::default(),
        );
        dispatch(
            &mut gpu,
            &mut pool,
            Some("disc"),
            None,
            "small",
            RESOLUTION,
            &TileGeneratorParams {
                shape_size: 0.03,
                ..TileGeneratorParams::default()
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            Some("disc"),
            None,
            "large",
            RESOLUTION,
            &TileGeneratorParams {
                shape_size: 0.3,
                ..TileGeneratorParams::default()
            },
        );
        let small = gpu.read_rgba8(pool.texture("small").unwrap(), RESOLUTION);
        let large = gpu.read_rgba8(pool.texture("large").unwrap(), RESOLUTION);

        assert_ne!(
            small, large,
            "different shape_size values should produce different output"
        );
    }

    /// A twisted Thread is a valid Shape input too (any Gray-output node
    /// is) — checks the owner's own reported scenario of wiring Thread
    /// straight into Pattern Generator, not just into Fabric Generator.
    #[test]
    fn a_twisted_thread_used_as_a_shape_still_tiles_seamlessly() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();

        crate::thread_shape::dispatch(
            &mut gpu,
            &mut pool,
            crate::thread_shape::ThreadOutputKeys {
                height: "thread_height",
                ply_id: "thread_ply_id",
                fiber: "thread_fiber",
                mask: "thread_mask",
            },
            RESOLUTION,
            &crate::thread_shape::ThreadShapeParams {
                threads: 1,
                plies: 2,
                thickness: 0.73,
                twist_angle: 35.0,
                ply_direction: crate::thread_shape::TwistDirection::Z,
                ..crate::thread_shape::ThreadShapeParams::default()
            },
        );
        dispatch(
            &mut gpu,
            &mut pool,
            Some("thread_height"),
            None,
            "with_thread",
            RESOLUTION,
            &TileGeneratorParams::default(),
        );
        let pixels = gpu.read_rgba8(pool.texture("with_thread").unwrap(), RESOLUTION);
        assert_tiles_seamlessly(&pixels, RESOLUTION);
    }

    #[test]
    fn position_jitter_changes_output() {
        let a = render(&TileGeneratorParams {
            position_jitter: 0.0,
            ..TileGeneratorParams::default()
        });
        let b = render(&TileGeneratorParams {
            position_jitter: 0.8,
            ..TileGeneratorParams::default()
        });
        assert_ne!(a, b, "Position Jitter had no effect on the output");
    }

    /// Regression test: the first version of Position Jitter only shifted
    /// the built-in rect's own `grid_rect_value` path, never the
    /// connected-Shape branch's own (separately computed) rotation/sampling
    /// — a real reported bug (caught by the owner's own hands-on testing
    /// with a Shape connected), the same class of gap Distribution Map had
    /// for the same reason.
    #[test]
    fn position_jitter_changes_output_with_a_shape_connected() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc",
            RESOLUTION,
            &crate::shape::ShapeParams::default(),
        );
        dispatch(
            &mut gpu,
            &mut pool,
            Some("disc"),
            None,
            "no_jitter",
            RESOLUTION,
            &TileGeneratorParams {
                position_jitter: 0.0,
                ..TileGeneratorParams::default()
            },
        );
        let without_jitter = gpu.read_rgba8(pool.texture("no_jitter").unwrap(), RESOLUTION);
        dispatch(
            &mut gpu,
            &mut pool,
            Some("disc"),
            None,
            "with_jitter",
            RESOLUTION,
            &TileGeneratorParams {
                position_jitter: 0.8,
                ..TileGeneratorParams::default()
            },
        );
        let with_jitter = gpu.read_rgba8(pool.texture("with_jitter").unwrap(), RESOLUTION);
        assert_ne!(
            without_jitter, with_jitter,
            "Position Jitter had no effect with a Shape connected"
        );
    }

    /// Regression test: Position Jitter can push a connected Shape's own
    /// sampling window outside that Shape's `[0, 1]` canvas (that's the
    /// whole point — the tile has moved within its cell). The first
    /// version clamped the out-of-range position *before* sampling instead
    /// of treating "outside" as empty, which pins every such query onto
    /// the canvas's own edge texel — repeating whatever's actually drawn
    /// there, not necessarily background, across pixels that should show
    /// nothing. A real reported bug (the owner's own connected flower
    /// shape visibly "cut off" instead of cleanly sliding), not caught by
    /// the sibling test above because `disc`'s own canvas already has a
    /// black margin at its edges, so clamping there happened to repeat
    /// black either way. A shape that's solid white edge-to-edge has no
    /// such margin: the old clamp-based code would read as pure white
    /// regardless of jitter (every edge texel is white too); the fix must
    /// show real black once the sampling window is pushed out of bounds.
    #[test]
    fn position_jitter_shows_background_instead_of_repeating_a_shapes_own_edge() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        gpu.fill_rgba8(&mut pool, "solid_shape", RESOLUTION, [255, 255, 255, 255]);
        dispatch(
            &mut gpu,
            &mut pool,
            Some("solid_shape"),
            None,
            "test",
            RESOLUTION,
            &TileGeneratorParams {
                position_jitter: 1.0,
                shape_size: 0.05,
                ..TileGeneratorParams::default()
            },
        );
        let pixels = gpu.read_rgba8(pool.texture("test").unwrap(), RESOLUTION);
        assert!(
            pixels.as_chunks::<4>().0.iter().any(|p| p[0] == 0),
            "Position Jitter should reveal empty background instead of repeating a \
             solid shape's own edge pixel once it pushes the sample window out of bounds"
        );
    }

    /// Same idea as `luminance_seed_changes_output_independently_of_seed`:
    /// Position Seed re-rolls just the per-tile offset, independent of the
    /// shared Seed.
    #[test]
    fn position_seed_changes_output_independently_of_seed() {
        let a = render(&TileGeneratorParams {
            seed: 1,
            position_seed: 1,
            position_jitter: 0.8,
            ..TileGeneratorParams::default()
        });
        let b = render(&TileGeneratorParams {
            seed: 1,
            position_seed: 2,
            position_jitter: 0.8,
            ..TileGeneratorParams::default()
        });
        assert_ne!(
            a, b,
            "different Position Seeds (same Seed) produced identical pixels"
        );
    }

    /// Position Jitter only has one consistent meaning once both axes
    /// repeat at the same scale — same scope restriction, and same test
    /// shape, as `offset_has_no_effect_outside_both` above.
    #[test]
    fn position_jitter_has_no_effect_outside_both() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc",
            RESOLUTION,
            &crate::shape::ShapeParams::default(),
        );
        for distribution in [
            PatternDistribution::HorizontalOnly,
            PatternDistribution::VerticalOnly,
        ] {
            dispatch(
                &mut gpu,
                &mut pool,
                Some("disc"),
                None,
                "no_jitter",
                RESOLUTION,
                &TileGeneratorParams {
                    rows: 6,
                    columns: 6,
                    position_jitter: 0.0,
                    distribution,
                    ..TileGeneratorParams::default()
                },
            );
            let without_jitter = gpu.read_rgba8(pool.texture("no_jitter").unwrap(), RESOLUTION);
            dispatch(
                &mut gpu,
                &mut pool,
                Some("disc"),
                None,
                "with_jitter",
                RESOLUTION,
                &TileGeneratorParams {
                    rows: 6,
                    columns: 6,
                    position_jitter: 0.8,
                    distribution,
                    ..TileGeneratorParams::default()
                },
            );
            let with_jitter = gpu.read_rgba8(pool.texture("with_jitter").unwrap(), RESOLUTION);
            assert_eq!(
                without_jitter, with_jitter,
                "Position Jitter changed {distribution:?}'s output outside Both"
            );
        }
    }

    #[test]
    fn tiles_seamlessly_with_position_jitter() {
        assert_tiles_seamlessly(
            &render(&TileGeneratorParams {
                position_jitter: 1.0,
                rows: 6,
                columns: 4,
                ..TileGeneratorParams::default()
            }),
            RESOLUTION,
        );
    }

    /// Regression test: with a single cell spanning the whole image (rows
    /// = columns = 1), Position Jitter pushing a connected Shape's own
    /// tile toward one edge is the *same thing* as it wrapping around to
    /// the opposite edge — the grid's only cell is its own only neighbor,
    /// via `wrap_coord`. An earlier version had no neighbor search for a
    /// connected Shape at all, so content pushed past a cell's own edge
    /// just vanished instead of wrapping — a real reported bug (a tile's
    /// own content "cut off" in a flat line, not only at the image's outer
    /// border but at every *internal* cell boundary too; a single giant
    /// cell collapses that internal case onto this same edge, so the
    /// existing, already-trusted `assert_tiles_seamlessly` helper can
    /// catch it directly).
    #[test]
    fn tiles_seamlessly_with_position_jitter_and_a_connected_shape() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc",
            RESOLUTION,
            &crate::shape::ShapeParams::default(),
        );
        dispatch(
            &mut gpu,
            &mut pool,
            Some("disc"),
            None,
            "shape_jitter_seamless",
            RESOLUTION,
            &TileGeneratorParams {
                rows: 1,
                columns: 1,
                shape_size: 1.0,
                position_jitter: 1.0,
                ..TileGeneratorParams::default()
            },
        );
        let pixels = gpu.read_rgba8(pool.texture("shape_jitter_seamless").unwrap(), RESOLUTION);
        assert_tiles_seamlessly(&pixels, RESOLUTION);
    }

    /// Size Jitter used to only ever shrink a tile (`size_scale` topped out
    /// at exactly 1.0, its own unjittered size) — the owner's own request
    /// to let it grow tiles too, not just shrink them. With Gap wide open
    /// (0.3, plenty of empty cell left to grow into) across the default
    /// 6x4 grid, growth being possible should raise the *average* covered
    /// area: a shrink-only `size_scale` averages 1/3 of a cell's own area
    /// across many random draws, while shrink-*or*-grow averages 0.75 (see
    /// `SIZE_JITTER_GROW_MAX`'s own doc comment in the shader for the
    /// exact math) — a large enough gap, over 24 independent cells, that
    /// this isn't a coin-flip result.
    #[test]
    fn size_jitter_can_grow_tiles_not_just_shrink_them() {
        // A single cell spanning the whole image (same trick the seamless-
        // tiling tests above use) with a generous Gap: under the OLD
        // shrink-only formula, `size_scale` never exceeds 1.0, so the
        // tile's own covered area could never exceed `(1.0 - 2*gap)^2` of
        // the image — a fixed, computable ceiling regardless of hash luck.
        // Seeing meaningfully MORE than that ceiling covered proves
        // `size_scale` genuinely exceeded 1.0 for that draw.
        let gap: f32 = 0.2;
        let old_formula_max_coverage = (1.0 - 2.0 * gap).powi(2);
        let count_bright_fraction = |pixels: &[u8]| {
            let bright = pixels
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|p| p[0] > 0)
                .count();
            bright as f32 / (RESOLUTION * RESOLUTION) as f32
        };
        let max_coverage_seen = (0..40)
            .map(|seed| {
                count_bright_fraction(&render(&TileGeneratorParams {
                    rows: 1,
                    columns: 1,
                    gap,
                    size_jitter: 1.0,
                    seed,
                    ..TileGeneratorParams::default()
                }))
            })
            .fold(0.0_f32, f32::max);
        assert!(
            max_coverage_seen > old_formula_max_coverage + 0.1,
            "Size Jitter should be able to grow a tile past the old shrink-only ceiling of \
             {old_formula_max_coverage:.2} coverage (size_scale capped at 1.0) \u{2014} best \
             coverage seen across 40 seeds was only {max_coverage_seen:.2}"
        );
    }

    /// Mirror Jitter is new: a chance each tile is independently flipped
    /// per axis, only visible with a connected Shape (the built-in rect is
    /// symmetric under reflection, a mathematical no-op). A plain
    /// left-to-right ramp (Gradient Linear) is a simple, already-trusted
    /// shape that's obviously NOT symmetric under a horizontal flip — if
    /// Mirror Jitter actually flips it, the ramp should read right-to-left
    /// instead for at least one of several random seeds, and left-to-right
    /// (unflipped) for at least one other.
    #[test]
    fn mirror_jitter_can_flip_a_connected_shape_horizontally() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::gradient_linear::dispatch(
            &mut gpu,
            &mut pool,
            "ramp",
            RESOLUTION,
            &crate::gradient_linear::GradientLinearParams::default(),
        );
        let half = (RESOLUTION / 2) as usize;
        let left_minus_right_bias =
            |gpu: &mut mc_gpu::Context, pool: &mut mc_gpu::TexturePool, mirror_seed: u32| -> f32 {
                dispatch(
                    gpu,
                    pool,
                    Some("ramp"),
                    None,
                    "mirror_test",
                    RESOLUTION,
                    &TileGeneratorParams {
                        rows: 1,
                        columns: 1,
                        shape_size: 1.0,
                        mirror_jitter: 1.0,
                        mirror_seed,
                        ..TileGeneratorParams::default()
                    },
                );
                let pixels = gpu.read_rgba8(pool.texture("mirror_test").unwrap(), RESOLUTION);
                let middle_row: Vec<u8> = pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .skip(half * RESOLUTION as usize)
                    .take(RESOLUTION as usize)
                    .map(|p| p[0])
                    .collect();
                let left_avg: f32 =
                    middle_row[..half].iter().map(|&v| v as f32).sum::<f32>() / half as f32;
                let right_avg: f32 =
                    middle_row[half..].iter().map(|&v| v as f32).sum::<f32>() / half as f32;
                left_avg - right_avg
            };
        let biases: Vec<f32> = (0..10)
            .map(|seed| left_minus_right_bias(&mut gpu, &mut pool, seed))
            .collect();
        assert!(
            biases.iter().any(|&b| b < -10.0),
            "expected at least one seed to show the un-mirrored left-to-right ramp \
             (left darker than right), found: {biases:?}"
        );
        assert!(
            biases.iter().any(|&b| b > 10.0),
            "Mirror Jitter should be able to flip a connected Shape horizontally \u{2014} \
             expected at least one of several seeds to show the ramp reversed (left \
             brighter than right), found: {biases:?}"
        );
    }

    #[test]
    fn tiles_seamlessly_with_mirror_jitter_and_a_connected_shape() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc",
            RESOLUTION,
            &crate::shape::ShapeParams::default(),
        );
        dispatch(
            &mut gpu,
            &mut pool,
            Some("disc"),
            None,
            "mirror_seamless",
            RESOLUTION,
            &TileGeneratorParams {
                mirror_jitter: 1.0,
                ..TileGeneratorParams::default()
            },
        );
        let pixels = gpu.read_rgba8(pool.texture("mirror_seamless").unwrap(), RESOLUTION);
        assert_tiles_seamlessly(&pixels, RESOLUTION);
    }

    fn render_with_distribution_map(params: &TileGeneratorParams, map_color: [u8; 4]) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        gpu.fill_rgba8(&mut pool, "dist_map", RESOLUTION, map_color);
        dispatch(
            &mut gpu,
            &mut pool,
            None,
            Some("dist_map"),
            "test",
            RESOLUTION,
            params,
        );
        let texture = pool.texture("test").expect("tile_generator texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    /// A solid-black Distribution Map, at full Amount, should drop every
    /// tile — every pixel of the output is the empty-gap value (0).
    #[test]
    fn solid_black_distribution_map_at_full_amount_drops_every_tile() {
        let pixels = render_with_distribution_map(
            &TileGeneratorParams {
                distribution_amount: 1.0,
                ..TileGeneratorParams::default()
            },
            [0, 0, 0, 255],
        );
        assert!(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[0] == 0 && p[1] == 0 && p[2] == 0),
            "a solid-black Distribution Map at Amount 1.0 should drop every tile"
        );
    }

    /// Regression test: the first version of this gate only checked
    /// `has_distribution_map` inside the built-in rect's own code paths,
    /// never inside the connected-Shape branch at all — a real reported
    /// bug (caught by the owner's own hands-on testing with a Shape AND a
    /// Distribution Map both wired in at once), not hypothetical. A
    /// solid-black map at full Amount must drop every tile here too.
    #[test]
    fn solid_black_distribution_map_at_full_amount_drops_every_tile_with_a_shape_connected() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc",
            RESOLUTION,
            &crate::shape::ShapeParams::default(),
        );
        gpu.fill_rgba8(&mut pool, "dist_map", RESOLUTION, [0, 0, 0, 255]);
        dispatch(
            &mut gpu,
            &mut pool,
            Some("disc"),
            Some("dist_map"),
            "test",
            RESOLUTION,
            &TileGeneratorParams {
                distribution_amount: 1.0,
                ..TileGeneratorParams::default()
            },
        );
        let pixels = gpu.read_rgba8(pool.texture("test").unwrap(), RESOLUTION);
        assert!(
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[0] == 0 && p[1] == 0 && p[2] == 0),
            "a solid-black Distribution Map at Amount 1.0 should drop every tile, \
             even with a Shape also connected"
        );
    }

    /// A solid-white Distribution Map, at full Amount, keeps every tile — a
    /// probability of exactly 1.0 for every cell means the gate never
    /// drops anything, so the output matches the no-map baseline exactly.
    #[test]
    fn solid_white_distribution_map_at_full_amount_matches_the_unconnected_baseline() {
        let params = TileGeneratorParams::default();
        let baseline = render(&params);
        let with_white_map = render_with_distribution_map(&params, [255, 255, 255, 255]);
        assert_eq!(
            baseline, with_white_map,
            "a solid-white Distribution Map at Amount 1.0 should match the unconnected baseline"
        );
    }

    /// Distribution Amount 0 ignores the map entirely, regardless of what's
    /// connected — same "mix toward the baseline at 0" convention Overlap
    /// Opacity already uses.
    #[test]
    fn distribution_amount_zero_ignores_a_solid_black_map() {
        let params = TileGeneratorParams {
            distribution_amount: 0.0,
            ..TileGeneratorParams::default()
        };
        let baseline = render(&params);
        let with_black_map = render_with_distribution_map(&params, [0, 0, 0, 255]);
        assert_eq!(
            baseline, with_black_map,
            "Distribution Amount 0 should ignore even a solid-black map"
        );
    }

    /// Same idea as Position/Luminance Seed: Distribution Seed re-rolls
    /// just which tiles the gate keeps, independent of the shared Seed —
    /// needs a mid-gray map (probability ~0.5) so the per-cell coin flip
    /// actually has something to disagree about.
    #[test]
    fn distribution_seed_changes_output_independently_of_seed() {
        let a = render_with_distribution_map(
            &TileGeneratorParams {
                seed: 1,
                distribution_seed: 1,
                distribution_amount: 1.0,
                ..TileGeneratorParams::default()
            },
            [128, 128, 128, 255],
        );
        let b = render_with_distribution_map(
            &TileGeneratorParams {
                seed: 1,
                distribution_seed: 2,
                distribution_amount: 1.0,
                ..TileGeneratorParams::default()
            },
            [128, 128, 128, 255],
        );
        assert_ne!(
            a, b,
            "different Distribution Seeds (same Seed) produced identical pixels"
        );
    }

    #[test]
    fn tiles_seamlessly_with_a_distribution_map_connected() {
        assert_tiles_seamlessly(
            &render_with_distribution_map(
                &TileGeneratorParams {
                    distribution_amount: 1.0,
                    rows: 6,
                    columns: 4,
                    ..TileGeneratorParams::default()
                },
                [128, 128, 128, 255],
            ),
            RESOLUTION,
        );
    }
}
