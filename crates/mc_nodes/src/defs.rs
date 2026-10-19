// SPDX-License-Identifier: MIT OR Apache-2.0
//! Node metadata: display name, category, inputs/outputs, params with
//! defaults, and a tooltip for every node kind (CLAUDE.md §6.6). This is the
//! single source of truth the UI (M3) uses to populate the add-node search
//! menu, render a generic parameter panel, and block incompatible
//! connections — none of it is hardcoded per node in `mc_app`.
//!
//! The GPU dispatch code in `perlin.rs`/`levels.rs`/`blend.rs` has its own
//! small `Default` impls for its params struct; those are a safety fallback
//! for a graph that doesn't specify a param; the defaults declared here are
//! what a freshly-added node starts with. The two are meant to agree, but
//! are deliberately independent — dispatch code doesn't depend on this
//! metadata layer.

use mc_core::{GradientStop, ParamValue, PortType};

/// One parameter a node declares, independent of its current value in any
/// particular graph.
pub struct ParamSpec {
    pub name: &'static str,
    pub display_name: &'static str,
    pub tooltip: &'static str,
    pub kind: ParamSpecKind,
}

/// A param's type and allowed range/options. New variants (Bool) get added
/// here once a node that needs them exists.
pub enum ParamSpecKind {
    Float {
        min: f32,
        max: f32,
        default: f32,
    },
    Int {
        min: i32,
        max: i32,
        default: i32,
    },
    Seed {
        default: u32,
    },
    Enum {
        options: &'static [&'static str],
        default: &'static str,
    },
    Color {
        default: [f32; 4],
    },
    Bool {
        default: bool,
    },
    /// Gradient Map's color ramp — at most 8 stops (`gradient_map.wgsl`'s
    /// fixed-size uniform array), interpolated in position order.
    Gradient {
        default: &'static [GradientStop],
    },
    /// A 2-component value in the same units on both axes (e.g. a UV-space
    /// offset) — `min`/`max` apply to both components identically, since
    /// every current use (Channel Split's per-channel offsets) is
    /// symmetric. First used by the Glitch family.
    Vec2 {
        min: f32,
        max: f32,
        default: [f32; 2],
    },
    /// Image Import's "Load Image…" button + embed-or-reference toggle —
    /// unlike every other kind, there's no meaningful static default beyond
    /// "nothing loaded yet" (`ImageRef::default()`).
    ImagePath,
    /// Freeform single-line text. First used by Knit's custom stitch chart
    /// — unlike [`ParamSpecKind::Enum`], there's no fixed option list to
    /// validate against; whatever the node's own `from_graph` does with a
    /// malformed value needs to degrade gracefully rather than panic
    /// (CLAUDE.md §12: "No `unwrap()`/`expect()` in app or library code
    /// paths a user can trigger").
    Text {
        default: &'static str,
    },
    /// The Path node's own vector curve data (PATH_NODE_SPEC.md §1-2) — no
    /// static default beyond `mc_core::Path::default()` (empty, Wrap on):
    /// a fresh Path node starts blank for the user to draw into. `Wrap`
    /// itself lives *inside* the stored `Path` value (its own field), not
    /// as a separate param here, so there's exactly one source of truth
    /// for what a Path node holds — the editor's own Wrap checkbox reads
    /// and writes that same field.
    Path,
    /// A font family, picked from whatever's actually installed on this
    /// machine (`mc_nodes::type_text::list_font_families`) — unlike
    /// [`ParamSpecKind::Enum`], the option list isn't known at compile
    /// time, so there's no static `default` here either: an empty string
    /// means "no specific family requested," which `type_text::dispatch`
    /// resolves to a generic sans-serif fallback rather than failing.
    /// Stored as a plain [`crate::ParamValue::Text`], same as
    /// [`ParamSpecKind::Text`] — this only differs in which widget
    /// `params_panel.rs` draws for it (a dropdown of installed fonts
    /// instead of a free-typed box).
    FontFamily,
}

impl ParamSpec {
    fn default_value(&self) -> ParamValue {
        match self.kind {
            ParamSpecKind::Float { default, .. } => ParamValue::Float(default),
            ParamSpecKind::Int { default, .. } => ParamValue::Int(default),
            ParamSpecKind::Seed { default } => ParamValue::Seed(default),
            ParamSpecKind::Enum { default, .. } => ParamValue::Enum(default.to_string()),
            ParamSpecKind::Color { default } => ParamValue::Color(default),
            ParamSpecKind::Bool { default } => ParamValue::Bool(default),
            ParamSpecKind::Gradient { default } => ParamValue::Gradient(default.to_vec()),
            ParamSpecKind::Vec2 { default, .. } => ParamValue::Vec2(default),
            ParamSpecKind::ImagePath => ParamValue::Image(mc_core::ImageRef::default()),
            ParamSpecKind::Text { default } => ParamValue::Text(default.to_string()),
            ParamSpecKind::Path => ParamValue::Path(mc_core::Path::default()),
            ParamSpecKind::FontFamily => ParamValue::Text(String::new()),
        }
    }
}

/// The shared "Tiling Mode" param (see `crate::tiling::TilingMode`) for any
/// node whose neighbor-sampling can show a visible edge-duplication
/// artifact — one helper rather than repeating the same `ParamSpec` at
/// every call site, so the name/wording stays identical everywhere it
/// appears.
const fn tiling_mode_param() -> ParamSpec {
    ParamSpec {
        name: "tiling_mode",
        display_name: "Tiling",
        tooltip: "This node can read past a tile's edge (e.g. a blur or an offset), which \
                  normally wraps around to the opposite edge to stay seamless. Turn an axis \
                  off here if you'd rather that content just clip at the edge instead of \
                  reappearing on the other side.",
        kind: ParamSpecKind::Enum {
            options: &crate::tiling::TilingMode::NAMES,
            default: "Both (Seamless)",
        },
    }
}

/// One input or output port a node declares.
pub struct PortSpec {
    pub name: &'static str,
    pub display_name: &'static str,
    pub port_type: PortType,
}

/// A node kind's full declaration.
pub struct NodeDef {
    pub kind: &'static str,
    pub display_name: &'static str,
    pub category: &'static str,
    pub tooltip: &'static str,
    pub inputs: &'static [PortSpec],
    pub outputs: &'static [PortSpec],
    pub params: &'static [ParamSpec],
}

impl NodeDef {
    /// A fresh `mc_core::Node` of this kind, with every param set to its
    /// declared default — what dragging this node onto the canvas creates.
    pub fn instantiate(&self) -> mc_core::Node {
        let mut node = mc_core::Node::new(self.kind);
        for param in self.params {
            node.params
                .insert(param.name.to_string(), param.default_value());
        }
        node
    }

    pub fn output(&self, name: &str) -> Option<&PortSpec> {
        self.outputs.iter().find(|p| p.name == name)
    }

    pub fn input(&self, name: &str) -> Option<&PortSpec> {
        self.inputs.iter().find(|p| p.name == name)
    }
}

/// All known node kinds. Looked up by [`node_def`]; iterated by the add-node
/// search menu via [`all_node_defs`].
pub fn all_node_defs() -> &'static [NodeDef] {
    NODE_DEFS
}

pub fn node_def(kind: &str) -> Option<&'static NodeDef> {
    NODE_DEFS.iter().find(|def| def.kind == kind)
}

/// Whether the UI should allow connecting a `from`-typed output to a
/// `to`-typed input (CLAUDE.md §5): Gray broadcasts implicitly into Color,
/// but Color needs an explicit Grayscale Conversion node to become Gray —
/// which doesn't exist yet, so that direction is simply blocked for now.
/// `Path` and `Sdf` are each their own thing entirely (vector data, and
/// geometric distance, neither a plain-brightness texture — see
/// `mc_core::PortType::Path`/`Sdf`'s own doc comments): only Path-to-Path
/// or Sdf-to-Sdf is ever valid, no broadcast either direction the way
/// Gray/Color have.
pub fn ports_compatible(from: PortType, to: PortType) -> bool {
    match (from, to) {
        (PortType::Path, PortType::Path) => true,
        (PortType::Path, _) | (_, PortType::Path) => false,
        (PortType::Sdf, PortType::Sdf) => true,
        (PortType::Sdf, _) | (_, PortType::Sdf) => false,
        (PortType::Color, PortType::Gray) => false,
        _ => true,
    }
}

/// Node kinds whose single "input"/"output" pair behaves identically
/// per-channel, regardless of whether that channel is a lone Gray value or
/// one of Color's three — so their declared-Color output port is really
/// just a default, not a fact about what they produce. Left hardcoded to
/// Color, a Gray source (e.g. a height map) couldn't pass through Levels
/// and still reach a Gray-only input (e.g. Height to Normal) without an
/// unnecessary Grayscale Conversion node in between.
const TYPE_PRESERVING_KINDS: &[&str] = &[
    "levels",
    "invert",
    "clamp",
    "threshold",
    "curve",
    "block_shift",
    "scanline_jitter",
    "ordered_dither",
    "data_corrupt",
    // Added in the same audit that caught Blend (see `effective_output_type`'s
    // "blend" special case): every one of these reads its *whole* input
    // texel (full vec4 — base/blur/offset/resample math, no luminance
    // reduction) and writes it straight back out, so a Gray-broadcast input
    // (R=G=B) comes out exactly as Gray-broadcast, same as the originally
    // fixed Levels/Invert/Clamp/Threshold/Curve group — verified by reading
    // each one's own .wgsl file, not assumed from its category.
    "histogram_scan",
    "blur",
    "warp",
    "transform2d",
    "directional_blur",
    "directional_warp",
    "slope_blur",
    "bevel",
];

/// The name of the port `effective_output_type` should follow upstream for a
/// given type-preserving node kind — most call it "input", but the Glitch
/// family's image input is named "image" to read naturally alongside their
/// optional "mask" input.
fn type_preserving_input_port(kind: &str) -> &'static str {
    match kind {
        "block_shift" | "scanline_jitter" | "ordered_dither" | "data_corrupt" => "image",
        _ => "input",
    }
}

/// The port type a node's output *actually* carries right now, given what
/// (if anything) is wired into its input(s) — for the nodes in
/// `TYPE_PRESERVING_KINDS` this follows the upstream chain back through any
/// number of other type-preserving nodes (Blend included) to the first real
/// source; for every other node it's just the statically declared type.
/// Used by the UI for pin coloring and connection-compatibility checks,
/// never by evaluation (the GPU side treats Gray and Color identically as a
/// broadcast RGBA texture either way).
pub fn effective_output_type(
    graph: &mc_core::Graph,
    node_id: mc_core::NodeId,
    node_kind: &str,
    declared: PortType,
) -> PortType {
    // A graph can't contain cycles (rejected at connect time), so this bound
    // only guards against a future bug reintroducing one — 64 hops is far
    // more than any real graph would chain through type-preserving nodes.
    effective_output_type_bounded(graph, node_id, node_kind, declared, 64)
}

fn effective_output_type_bounded(
    graph: &mc_core::Graph,
    node_id: mc_core::NodeId,
    node_kind: &str,
    declared: PortType,
    depth: u32,
) -> PortType {
    let Some(depth) = depth.checked_sub(1) else {
        return declared;
    };
    if node_kind == "image_import" {
        // Depends on its own "Convert to Grayscale" param, not an upstream
        // connection (it has no inputs at all) — unlike every other case
        // here, which all follow what's wired in.
        let grayscale = graph
            .node(node_id)
            .and_then(|n| n.params.get("grayscale"))
            .and_then(mc_core::ParamValue::as_bool)
            .unwrap_or(false);
        return if grayscale { PortType::Gray } else { declared };
    }
    if node_kind == "blend" {
        // Color wins over Gray: blending in *any* real color content can
        // produce non-gray output, so Color is the only type-safe result
        // once either input is Color — the reverse of how a lone
        // type-preserving input works, where there's only one source to
        // defer to. Gray only if *both* inputs resolve to Gray.
        let base = resolve_upstream_type(graph, node_id, "base", depth);
        let blend = resolve_upstream_type(graph, node_id, "blend", depth);
        return match (base, blend) {
            (None, None) => declared,
            (Some(only), None) | (None, Some(only)) => only,
            (Some(a), Some(b)) => {
                if a == PortType::Color || b == PortType::Color {
                    PortType::Color
                } else {
                    PortType::Gray
                }
            }
        };
    }
    if !TYPE_PRESERVING_KINDS.contains(&node_kind) {
        return declared;
    }
    resolve_upstream_type(graph, node_id, type_preserving_input_port(node_kind), depth)
        .unwrap_or(declared)
}

/// What's wired into `node_id`'s `port_name` input *actually* outputs, given
/// its own effective type might in turn depend on what's wired into it —
/// recurses one node upstream via `effective_output_type_bounded` rather
/// than reading the upstream node's merely-declared type, so a chain through
/// any mix of type-preserving nodes (including Blend) resolves correctly
/// however deep it goes. `None` if `port_name` isn't connected to anything.
fn resolve_upstream_type(
    graph: &mc_core::Graph,
    node_id: mc_core::NodeId,
    port_name: &str,
    depth: u32,
) -> Option<PortType> {
    let upstream_id = graph.upstream(node_id, port_name)?;
    let upstream_node = graph.node(upstream_id)?;
    let upstream_def = node_def(&upstream_node.kind)?;
    let upstream_output = upstream_def.outputs.first()?;
    Some(effective_output_type_bounded(
        graph,
        upstream_id,
        &upstream_node.kind,
        upstream_output.port_type,
        depth,
    ))
}

/// Whether `port_name` on a node of `kind` is an optional input with a
/// meaningful built-in fallback when nothing's connected (Fabric
/// Generator's Fiber, Pattern Generator's Shape) — as opposed to an
/// ordinary required filter input, where "not connected" just means the
/// node can't evaluate yet and isn't worth a status line. Lets the
/// parameter panel show a "connected / using the built-in default" line
/// for exactly the inputs where that's genuinely ambiguous enough to
/// confuse an artist (reported: a connected Fiber using default params
/// that happen to match the fallback's own defaults looked indistinguishable
/// from not being connected at all).
pub fn input_is_optional(kind: &str, port_name: &str) -> bool {
    matches!(
        (kind, port_name),
        ("fabric_generator", "fiber")
            | ("fabric_generator", "fiber_ply_id")
            | ("tile_generator", "shape")
            | ("material_preset", "base_color")
            | ("material_preset", "roughness")
            | ("material_preset", "metallic")
            | ("distance_spread", "source")
    )
}

/// Whether `param_name` currently means anything for a node of `node_kind`
/// given its other params right now — e.g. Shape's "Sides"/"Point Depth"/
/// "Curvature" only apply to some of its "Kind" values (a Disc has no sides,
/// Point Depth only unfolds a Star, Curvature only shapes a Leaf's tips).
/// Lives here rather than in `mc_app`'s generic parameter panel so a node's
/// own quirks stay in its own metadata, not scattered into per-node-kind UI
/// code. Defaults to always-relevant for every node/param this doesn't know
/// about.
pub fn param_is_relevant(
    node_kind: &str,
    param_name: &str,
    current_params: &std::collections::BTreeMap<String, ParamValue>,
) -> bool {
    if node_kind == "dither_pattern" {
        let pattern = current_params.get("pattern").and_then(|v| match v {
            ParamValue::Enum(name) => Some(name.as_str()),
            _ => None,
        });
        return match param_name {
            "seed" => pattern == Some("Blue Noise"),
            "rotation" => pattern != Some("Blue Noise"),
            "dot_size" => pattern == Some("Dot"),
            _ => true,
        };
    }
    if node_kind == "ordered_dither" || node_kind == "palette_dither" {
        let pattern = current_params.get("pattern").and_then(|v| match v {
            ParamValue::Enum(name) => Some(name.as_str()),
            _ => None,
        });
        return match param_name {
            "seed" => pattern == Some("Blue Noise"),
            _ => true,
        };
    }
    if node_kind == "tile_generator" {
        let layout = current_params.get("layout").and_then(|v| match v {
            ParamValue::Enum(name) => Some(name.as_str()),
            _ => None,
        });
        let gap_is_negative = current_params
            .get("gap")
            .and_then(ParamValue::as_float)
            .unwrap_or(0.0)
            < 0.0;
        return match param_name {
            "plank_length" | "plank_width" => layout == Some("Herringbone"),
            // Herringbone derives its own density from Columns alone (see
            // `tile_generator::pack_params`) — Rows would otherwise sit
            // there looking like an independent control while silently
            // doing nothing.
            "rows" => layout != Some("Herringbone"),
            // Grid layout only (Herringbone's lattice doesn't overlap this
            // way — see `OverlapBlendMode`'s own doc comment), and only
            // visible once Gap is actually negative, since a non-negative
            // Gap never produces any overlap for these to affect.
            "overlap_blend_mode" | "overlap_opacity" => {
                layout != Some("Herringbone") && gap_is_negative
            }
            _ => true,
        };
    }
    if node_kind == "fabric_generator" {
        let weave_type = current_params.get("weave_type").and_then(|v| match v {
            ParamValue::Enum(name) => Some(name.as_str()),
            _ => None,
        });
        return match param_name {
            "thread_group_size" => weave_type == Some("Plain Weave"),
            "twill_count" => matches!(weave_type, Some("Twill") | Some("Herringbone")),
            "rib_width" => weave_type == Some("Rib Knit"),
            _ => true,
        };
    }
    if node_kind == "thread_shape" {
        let cable_groups = current_params
            .get("cable_groups")
            .and_then(ParamValue::as_int)
            .unwrap_or(0);
        let fiber_mode = current_params.get("fiber_mode").and_then(|v| match v {
            ParamValue::Enum(name) => Some(name.as_str()),
            _ => None,
        });
        return match param_name {
            "cable_angle" => cable_groups >= 2,
            "fibers_per_ply" | "fiber_depth" => fiber_mode != Some("None"),
            _ => true,
        };
    }
    if node_kind == "knit" {
        if param_name == "courses" {
            let auto_courses = current_params
                .get("auto_courses")
                .and_then(ParamValue::as_bool)
                .unwrap_or(true);
            return !auto_courses;
        }
        if matches!(param_name, "rib_depth" | "rib_compaction") {
            // Whether the rib rule actually fires depends on the chart's
            // own content (any chart, including a hand-typed custom one),
            // not on which named preset is picked — resolve() already does
            // exactly that detection, so reuse it rather than re-deriving
            // "is this a rib preset" from the preset name here too.
            let params = crate::knit::KnitParams::from_graph(current_params);
            let g = crate::knit::resolve(&params);
            return g.is_rib || g.is_links;
        }
        return true;
    }
    if node_kind == "stitch" {
        let stitch_kind = current_params
            .get("stitch")
            .and_then(|v| match v {
                ParamValue::Enum(name) => Some(name.as_str()),
                _ => None,
            })
            .unwrap_or("running");
        // French Knot's own params (`KnotParams`) and every other kind's
        // own (`StitchParams`) are close to disjoint — Look/Thread
        // Diameter/Plies/Ply Direction don't apply to a knot (it has its
        // own Knot Look/Knot Strand Diameter/Knot Strands instead), and
        // the knot-only params obviously don't apply to the other 5.
        // Height Range and Twist Angle are both split the same way (a
        // knot's own wraps genuinely stack, unlike a single stitch, and
        // its own strands are short and tightly bundled — see
        // "knot_height_range"/"knot_twist_angle"'s own tooltips);
        // Regularity/Seed are the real overlap (see
        // `KnotParams::from_graph`'s own doc comment for why those two
        // are shared, not duplicated). Hole Gap and Wrap Direction aren't
        // exposed at all anymore — neither one earned its own slider
        // (Hole Gap only ever moves the short rise stub, which mostly
        // dives back under the fabric regardless; Wrap Direction is a
        // mirror-image flip nobody reached for), so both are pinned to a
        // fixed value in `KnotParams::from_graph` instead, same "delete
        // the dead control, keep the field" move already made for
        // Stitch's own Tile Size/Gap/Satin Mitre.
        if stitch_kind == "french_knot" {
            return matches!(
                param_name,
                "stitch"
                    | "knot_wraps"
                    | "knot_strands"
                    | "knot_strand_mm"
                    | "knot_looseness"
                    | "knot_look"
                    | "knot_twist_angle"
                    | "regularity"
                    | "knot_height_range"
                    | "seed"
            );
        }
        return match param_name {
            "knot_wraps" | "knot_strands" | "knot_strand_mm" | "knot_looseness" | "knot_look"
            | "knot_twist_angle" | "knot_height_range" => false,
            "width_mm" | "pull_comp_mm" => matches!(stitch_kind, "zigzag" | "satin"),
            "stem_width_mm" | "stem_side" => stitch_kind == "stem",
            "density_mm" => stitch_kind == "satin",
            "bean_repeats" => stitch_kind == "bean",
            "zigzag_angle" => stitch_kind == "zigzag",
            _ => true,
        };
    }
    if node_kind != "shape" {
        return true;
    }
    let shape_kind = current_params
        .get("kind")
        .and_then(|v| match v {
            ParamValue::Enum(name) => crate::shape::ShapeKind::from_name(name),
            _ => None,
        })
        .unwrap_or(crate::shape::ShapeKind::Disc);
    match param_name {
        "sides" => matches!(
            shape_kind,
            crate::shape::ShapeKind::Polygon | crate::shape::ShapeKind::Star
        ),
        "point_depth" => matches!(shape_kind, crate::shape::ShapeKind::Star),
        "curvature_top" | "curvature_bottom" | "curvature_middle" | "leaf_width"
        | "middle_position" => {
            matches!(shape_kind, crate::shape::ShapeKind::Leaf)
        }
        _ => true,
    }
}

const fn output_node_def(kind: &'static str, display_name: &'static str) -> NodeDef {
    NodeDef {
        kind,
        display_name,
        category: "Output",
        tooltip: "Marks this input as the material's output for this channel.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[],
        params: &[],
    }
}

static NODE_DEFS: &[NodeDef] = &[
    NodeDef {
        kind: "perlin_noise",
        display_name: "Perlin Noise",
        category: "Generators/Noise",
        tooltip: "Classic gradient noise — smooth, organic variation.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "scale",
                display_name: "Scale",
                tooltip: "How many noise cells tile across the texture.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "octaves",
                display_name: "Octaves",
                tooltip: "How many layers of noise are summed.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 8,
                    default: 4,
                },
            },
            ParamSpec {
                name: "persistence",
                display_name: "Persistence",
                tooltip: "How much each octave's contribution shrinks.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern without changing its character.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "anisotropic_noise",
        display_name: "Anisotropic Noise",
        category: "Generators/Noise",
        tooltip: "Directional gradient noise — different frequency per axis for streaky, brushed-metal or wood-grain-like variation.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "scale_x",
                display_name: "Scale X",
                tooltip: "How many noise cells tile across the texture along X.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "scale_y",
                display_name: "Scale Y",
                tooltip: "How many noise cells tile across the texture along Y. A different value from Scale X is what stretches the noise into streaks.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 16,
                },
            },
            ParamSpec {
                name: "octaves",
                display_name: "Octaves",
                tooltip: "How many layers of noise are summed.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 8,
                    default: 4,
                },
            },
            ParamSpec {
                name: "persistence",
                display_name: "Persistence",
                tooltip: "How much each octave's contribution shrinks.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern without changing its character.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            ParamSpec {
                name: "sharpness",
                display_name: "Sharpness",
                tooltip: "Boosts contrast around mid-gray to make the noise read as crisper and less blurry. 1 = unchanged.",
                kind: ParamSpecKind::Float {
                    min: 1.0,
                    max: 6.0,
                    default: 1.0,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "gabor_noise",
        display_name: "Gabor Noise",
        category: "Generators/Noise",
        tooltip: "Directional, band-limited noise with exact control of direction, frequency \
                  and spread — brushed metal, wood-grain bands, sand ripples, wave patterns, \
                  anisotropic roughness.",
        inputs: &[],
        outputs: &[
            PortSpec {
                name: "value",
                display_name: "Value",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "phasor",
                display_name: "Phasor",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "phase",
                display_name: "Phase",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "intensity",
                display_name: "Intensity",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "raw",
                display_name: "Raw",
                port_type: PortType::Gray,
            },
        ],
        params: &[
            ParamSpec {
                name: "cells",
                display_name: "Cells",
                tooltip: "Grid cells per tile; kernel radius is 1/cells (bigger = smaller blobs).",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 16,
                },
            },
            ParamSpec {
                name: "frequency",
                display_name: "Frequency",
                tooltip: "Stripes (cycles) per tile along the angle.",
                kind: ParamSpecKind::Float {
                    min: 1.0,
                    max: 200.0,
                    default: 48.0,
                },
            },
            ParamSpec {
                name: "angle",
                display_name: "Angle",
                tooltip: "Direction the values change along, in degrees (0 gives vertical bands).",
                kind: ParamSpecKind::Float {
                    min: -360.0,
                    max: 360.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "spread",
                display_name: "Spread",
                tooltip: "0 = one direction, 1 = every direction (isotropic).",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "impulses",
                display_name: "Impulses",
                tooltip: "Kernels per cell (2 is already enough for most looks; 1 looks patchy).",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 8,
                    default: 4,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern without changing its character.",
                kind: ParamSpecKind::Seed { default: 1 },
            },
        ],
    },
    NodeDef {
        kind: "levels",
        display_name: "Levels",
        category: "Filters/Adjustments",
        tooltip: "Remap an input/output range, with a gamma curve in between.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "in_low",
                display_name: "In Low",
                tooltip: "Input black point.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "in_high",
                display_name: "In High",
                tooltip: "Input white point.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "gamma",
                display_name: "Gamma",
                tooltip: "Midtone curve; 1.0 is linear.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 4.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "out_low",
                display_name: "Out Low",
                tooltip: "Output black point.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "out_high",
                display_name: "Out High",
                tooltip: "Output white point.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "blend",
        display_name: "Blend",
        category: "Filters/Blending & Masking",
        tooltip: "Combine two inputs with a blend mode, opacity, and optional mask.",
        inputs: &[
            PortSpec {
                name: "base",
                display_name: "Base",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "blend",
                display_name: "Blend",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "mode",
                display_name: "Mode",
                tooltip: "How base and blend combine.",
                kind: ParamSpecKind::Enum {
                    options: &crate::blend::BlendMode::NAMES,
                    default: "Copy",
                },
            },
            ParamSpec {
                name: "opacity",
                display_name: "Opacity",
                tooltip: "Overall strength of the blend.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "voronoi",
        display_name: "Voronoi",
        category: "Generators/Noise",
        tooltip: "Cellular noise — distances to a scattered set of points.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "scale",
                display_name: "Scale",
                tooltip: "How many cells tile across the texture.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "jitter",
                display_name: "Jitter",
                tooltip: "How far each cell's point can move from center.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern without changing its character.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            ParamSpec {
                name: "mode",
                display_name: "Mode",
                tooltip: "Which cellular distance to output.",
                kind: ParamSpecKind::Enum {
                    options: &crate::voronoi::VoronoiMode::NAMES,
                    default: "F1",
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "fractal_sum",
        display_name: "Fractal Sum",
        category: "Generators/Noise",
        tooltip: "Several octaves of a base noise, summed (fBm).",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "base",
                display_name: "Base Noise",
                tooltip: "Which noise type to layer.",
                kind: ParamSpecKind::Enum {
                    options: &crate::fractal_sum::FractalBase::NAMES,
                    default: "Perlin",
                },
            },
            ParamSpec {
                name: "scale",
                display_name: "Scale",
                tooltip: "Base noise cells tiling across the texture.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "octaves",
                display_name: "Octaves",
                tooltip: "How many layers are summed.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 8,
                    default: 4,
                },
            },
            ParamSpec {
                name: "lacunarity",
                display_name: "Lacunarity",
                tooltip: "Frequency multiplier per octave.",
                kind: ParamSpecKind::Float {
                    min: 1.0,
                    max: 4.0,
                    default: 2.0,
                },
            },
            ParamSpec {
                name: "gain",
                display_name: "Gain",
                tooltip: "Amplitude multiplier per octave.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern without changing its character.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "tile_generator",
        display_name: "Pattern Generator",
        category: "Generators/Patterns",
        tooltip: "A brick or grid pattern — Offset 0.5 for a classic running-bond brick layout.",
        inputs: &[PortSpec {
            name: "shape",
            display_name: "Shape",
            port_type: PortType::Gray,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "rows",
                display_name: "Rows",
                tooltip: "Must be even for seamless vertical tiling of the brick stagger.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 6,
                },
            },
            ParamSpec {
                name: "columns",
                display_name: "Columns",
                tooltip: "Grid columns. Herringbone layout: its own density \u{2014} Rows is \
                          hidden there, since this layout's repeat unit is inherently one \
                          number, not independent row/column counts.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "offset",
                display_name: "Offset",
                tooltip: "Grid layout: stagger alternate rows by this fraction of a column \
                          (0.5 = brick). Has no effect with Horizontal Only/Vertical Only \
                          Tiling \u{2014} staggering only makes sense across a full grid. \
                          Herringbone layout: below 0.25 gives the classic 45\u{b0} diagonal \
                          chevron, 0.25 and above gives the axis-aligned 0\u{b0}/90\u{b0} look \
                          \u{2014} the same Running Bond/Stack Bond values the Stagger Preset \
                          picker already sets.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "gap",
                display_name: "Gap",
                tooltip: "Mortar gap width. Negative values grow each tile past its own cell \
                          instead, overlapping its neighbors. Only affects the built-in tile \
                          shape \u{2014} no effect once something's wired into Shape below.",
                kind: ParamSpecKind::Float {
                    min: -0.4,
                    max: 0.4,
                    // Matches Stack Bond (Grid)'s own preset value, so a
                    // freshly-added node already reads as that preset in the
                    // Stagger Preset picker instead of needing it clicked
                    // manually to see any gap at all.
                    default: 0.04,
                },
            },
            ParamSpec {
                name: "overlap_blend_mode",
                display_name: "Overlap Blend",
                tooltip: "How overlapping tiles combine once a negative Gap grows them into \
                          each other. Add Sub treats values as signed around mid-gray \u{2014} \
                          above raises, below lowers.",
                kind: ParamSpecKind::Enum {
                    options: &crate::tile_generator::OverlapBlendMode::NAMES,
                    default: "Max",
                },
            },
            ParamSpec {
                name: "overlap_opacity",
                display_name: "Overlap Opacity",
                tooltip: "How much of the overlap blend shows \u{2014} 0 ignores any overlap \
                          entirely, 1 is the full blended result.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                // Display name/tooltip only, deliberately not "Bevel" — this
                // is a flat per-tile luminance fade (see tile_generator.wgsl:
                // `mix(0.5, 1.0, bevel_amount)`), not a height-shaping
                // operation, and keeping the old name invited confusion with
                // the real Bevel node (owner feedback: built two comparable
                // graphs and found they "looked nothing alike," which they
                // never were going to — same name, unrelated effects). The
                // `name` key stays "bevel" so existing .mcgraph files with
                // this param saved under that key keep loading correctly.
                name: "bevel",
                display_name: "Edge Darkening",
                tooltip: "How much each tile's edge darkens toward the gap — a flat shading \
                          fade, not a height change. Only affects the built-in tile shape \u{2014} \
                          no effect once something's wired into Shape below.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.4,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "shape_size",
                display_name: "Shape Size",
                tooltip: "How large each stamped Shape is, as a fraction of the whole tile \u{2014} \
                          stays the same absolute size as Tile Count changes, unlike a plain \
                          per-cell zoom. 1.0 is roughly the width of the whole tile; typical \
                          values are much smaller (0.05\u{2013}0.3) once Tile Count is more than a \
                          few cells across. Only shown once something's wired into Shape below.",
                kind: ParamSpecKind::Float {
                    min: 0.01,
                    max: 2.0,
                    default: 0.1,
                },
            },
            ParamSpec {
                name: "size_jitter",
                display_name: "Size Jitter",
                tooltip: "Per-tile random shrink.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "rotation_jitter",
                display_name: "Rotation Jitter",
                tooltip: "Per-tile random tilt.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "luminance_jitter",
                display_name: "Luminance Jitter",
                tooltip: "Per-tile random brightness variation.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the per-tile randomness.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            ParamSpec {
                name: "luminance_seed",
                display_name: "Luminance Seed",
                tooltip: "Independent of Seed above \u{2014} re-rolls just the per-tile \
                          brightness variation, without reshuffling tile size/rotation too.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            // Not the shared `tiling_mode_param()` (wrap-vs-clip at a
            // sampled edge) — this is a different concept: which grid axes
            // get a tile at all. Both is the standard full grid; Horizontal/
            // Vertical Only draw a single centered row/column at normal
            // (unstretched) tile size — e.g. distributing a connected Shape
            // across one row or column instead of a full grid.
            ParamSpec {
                name: "tiling_mode",
                display_name: "Tiling",
                tooltip: "Both is the standard full grid. Horizontal/Vertical Only draw a single \
                          centered row/column of tiles at their normal size instead.",
                kind: ParamSpecKind::Enum {
                    options: &crate::tile_generator::PatternDistribution::NAMES,
                    default: "Both (Seamless)",
                },
            },
            ParamSpec {
                name: "layout",
                display_name: "Layout",
                tooltip: "Grid is the standard built-in rect tile above. Herringbone replaces it \
                          with the classic interlocking parquet pattern — real bounded planks, \
                          not a stretched/rotated single shape. Only applies with nothing wired \
                          into Shape below; hides Rows, since Columns alone sets its density.",
                kind: ParamSpecKind::Enum {
                    options: &crate::tile_generator::PatternLayout::NAMES,
                    default: "Grid",
                },
            },
            ParamSpec {
                name: "plank_length",
                display_name: "Plank Length",
                tooltip: "Herringbone only: the plank's relative length \u{2014} only its ratio \
                          to Plank Width actually matters. Changing it re-tiles the whole \
                          pattern with more or fewer, differently-proportioned pieces that \
                          still meet edge to edge, rather than leaving a gap.",
                kind: ParamSpecKind::Float {
                    min: 1.0,
                    max: 4.0,
                    default: 2.0,
                },
            },
            ParamSpec {
                name: "plank_width",
                display_name: "Plank Width",
                tooltip: "Herringbone only: same idea as Plank Length, for the plank's own \
                          width instead \u{2014} it's the ratio between the two that sets the \
                          proportion, not either value alone.",
                kind: ParamSpecKind::Float {
                    min: 0.5,
                    max: 2.0,
                    default: 1.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "fabric_generator",
        display_name: "Fabric Generator",
        category: "Generators/Patterns",
        tooltip: "Woven and knit base shapes for building fabric materials from scratch.",
        inputs: &[
            PortSpec {
                name: "fiber",
                display_name: "Thread",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "fiber_ply_id",
                display_name: "Thread Ply ID",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[
            PortSpec {
                name: "output",
                display_name: "Output",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "ply_id",
                display_name: "Ply ID",
                port_type: PortType::Gray,
            },
        ],
        params: &[
            ParamSpec {
                name: "weave_type",
                display_name: "Weave Type",
                tooltip: "Which thread/stitch structure to generate. Each type only uses some \
                          of the params below \u{2014} see their own tooltips.",
                kind: ParamSpecKind::Enum {
                    options: &crate::fabric_generator::WeaveType::NAMES,
                    default: "Plain Weave",
                },
            },
            ParamSpec {
                name: "rows",
                display_name: "Rows",
                tooltip: "Thread/stitch density, top to bottom. Keep even for seamless tiling.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 24,
                },
            },
            ParamSpec {
                name: "columns",
                display_name: "Columns",
                tooltip: "Thread/stitch density, left to right. Keep even for seamless tiling \
                          (and a multiple of Twill Count for Twill, Rib Width \u{d7} 2 for Rib \
                          Knit, or Twill Count \u{d7} 2 for Herringbone).",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 24,
                },
            },
            ParamSpec {
                name: "thread_width",
                display_name: "Thread Width",
                tooltip: "Stockinette: how much of each loop's own cell it fills. Every other \
                          weave type: the fallback fiber's width, shown only while nothing's \
                          wired into Thread below.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 1.0,
                    default: 0.85,
                },
            },
            ParamSpec {
                name: "thread_group_size",
                display_name: "Thread Group Size",
                tooltip: "Plain Weave only: threads per group acting as one woven unit. 1 is a \
                          classic plain weave; 2 or more groups threads into blocks, the \
                          Basket Weave look.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 6,
                    default: 1,
                },
            },
            ParamSpec {
                name: "twill_count",
                display_name: "Twill Count",
                tooltip: "Twill and Herringbone: the diagonal weave stripe's repeat period \
                          \u{2014} lower is a steeper, tighter diagonal. Herringbone also \
                          reuses it as the width (in columns) of each band before the diagonal \
                          reverses into the chevron zigzag.",
                kind: ParamSpecKind::Int {
                    min: 2,
                    max: 8,
                    default: 4,
                },
            },
            ParamSpec {
                name: "rib_width",
                display_name: "Rib Width",
                tooltip: "Rib Knit only: how many columns wide each rib stripe is.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 6,
                    default: 1,
                },
            },
            ParamSpec {
                name: "size_jitter",
                display_name: "Size Jitter",
                tooltip: "Per-thread/stitch random width variation, for a less mechanically \
                          perfect look.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.06,
                },
            },
            ParamSpec {
                name: "luminance_jitter",
                display_name: "Luminance Jitter",
                tooltip: "Per-thread/stitch random brightness variation.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.1,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the per-thread/stitch randomness.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "thread_shape",
        display_name: "Thread",
        category: "Generators/Patterns",
        tooltip: "Plied yarn built from real textile geometry \u{2014} wire Height into \
                  Fabric Generator's Thread input to control what its threads look like.",
        inputs: &[],
        // Height stays first so `mc_nodes::registry::resolve_texture_key`'s
        // "first declared output" default keeps showing it in this node's
        // own thumbnail/2D View/3D viewport, and so a Fabric Generator
        // wired up before this rebuild keeps working once reconnected.
        outputs: &[
            PortSpec {
                name: "height",
                display_name: "Height",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "ply_id",
                display_name: "Ply ID",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "fiber",
                display_name: "Fiber",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
        ],
        params: &[
            ParamSpec {
                name: "threads",
                display_name: "Threads",
                tooltip: "How many threads stack across the tile.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "thickness",
                display_name: "Thickness",
                tooltip: "Thread diameter as a fraction of its own row height. Gaps between \
                          rows show the background.",
                kind: ParamSpecKind::Float {
                    min: 0.2,
                    max: 1.0,
                    default: 0.8,
                },
            },
            ParamSpec {
                name: "plies",
                display_name: "Plies",
                tooltip: "How many round strands are twisted together. 1 is a single \
                          (unplied) yarn; 2-6 match 2-ply through 6-ply.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 6,
                    default: 2,
                },
            },
            ParamSpec {
                name: "ply_direction",
                display_name: "Ply Twist",
                tooltip: "Which way the plies spiral (textile S/Z-twist terms). Most spun \
                          singles are Z, so the ply twist is usually S.",
                kind: ParamSpecKind::Enum {
                    options: &crate::thread_shape::TwistDirection::NAMES,
                    default: "S",
                },
            },
            ParamSpec {
                name: "twist_angle",
                display_name: "Twist Angle",
                tooltip: "The ply helix angle. Low (\u{2248}15\u{b0}) is bulky and soft, high \
                          (\u{2248}45\u{b0}) is smooth and dense.",
                kind: ParamSpecKind::Float {
                    min: 5.0,
                    max: 60.0,
                    default: 30.0,
                },
            },
            ParamSpec {
                name: "fiber_mode",
                display_name: "Fiber Twist",
                tooltip: "Balanced (the normal case): fibers twist opposite the ply, so they \
                          read as nearly parallel to the thread. Crepe: same direction, \
                          over-twisted, kinky. None: smooth, no visible fibers.",
                kind: ParamSpecKind::Enum {
                    options: &crate::thread_shape::FiberMode::NAMES,
                    default: "Balanced",
                },
            },
            ParamSpec {
                name: "fibers_per_ply",
                display_name: "Fibers per Ply",
                tooltip: "How many visible fiber grooves wrap each ply.",
                kind: ParamSpecKind::Int {
                    min: 3,
                    max: 24,
                    default: 7,
                },
            },
            ParamSpec {
                name: "fiber_depth",
                display_name: "Fiber Depth",
                tooltip: "Groove depth, as a fraction of a ply's own radius.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.4,
                    default: 0.12,
                },
            },
            ParamSpec {
                name: "cable_groups",
                display_name: "Cable Groups",
                tooltip: "0 is off. 2+ twists that many already-plied groups together again \
                          (a helix of helices) \u{2014} a 4-ply cable, say, as 2 groups of 2-ply.",
                kind: ParamSpecKind::Int {
                    min: 0,
                    max: 4,
                    default: 0,
                },
            },
            ParamSpec {
                name: "cable_angle",
                display_name: "Cable Angle",
                tooltip: "The cable twist's own helix angle \u{2014} only visible with Cable \
                          Groups 2+.",
                kind: ParamSpecKind::Float {
                    min: 5.0,
                    max: 60.0,
                    default: 25.0,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes each thread row's own twist phase so stacked rows don't \
                          line up.",
                kind: ParamSpecKind::Seed { default: 1 },
            },
        ],
    },
    NodeDef {
        kind: "scattered_streaks",
        display_name: "Streaks",
        category: "Generators/Noise",
        tooltip: "Many thin stretched strokes — wood fibres and rays, scratches, hair and fur \
                  clumps, rain, grass blades seen from above, brushed and anisotropic detail.",
        inputs: &[PortSpec {
            name: "angle_map",
            display_name: "Angle Map",
            port_type: PortType::Gray,
        }],
        outputs: &[
            PortSpec {
                name: "height",
                display_name: "Height",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "streak_id",
                display_name: "Streak ID",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "along",
                display_name: "Along",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "direction",
                display_name: "Direction",
                port_type: PortType::Color,
            },
        ],
        params: &[
            ParamSpec {
                name: "cells",
                display_name: "Cells",
                tooltip: "Grid cells per tile.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 24,
                },
            },
            ParamSpec {
                name: "per_cell",
                display_name: "Per Cell",
                tooltip: "Streaks per cell.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 8,
                    default: 2,
                },
            },
            ParamSpec {
                name: "length",
                display_name: "Length",
                tooltip: "Streak length, in tile units.",
                kind: ParamSpecKind::Float {
                    min: 0.001,
                    max: 1.0,
                    default: 0.10,
                },
            },
            ParamSpec {
                name: "length_random",
                display_name: "Length Random",
                tooltip: "Shortens length by up to this fraction, randomly per streak.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "width",
                display_name: "Width",
                tooltip: "Streak width, in tile units.",
                kind: ParamSpecKind::Float {
                    min: 0.0001,
                    max: 0.2,
                    default: 0.004,
                },
            },
            ParamSpec {
                name: "width_random",
                display_name: "Width Random",
                tooltip: "Narrows width by up to this fraction, randomly per streak.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.3,
                },
            },
            ParamSpec {
                name: "angle",
                display_name: "Angle",
                tooltip: "Main direction, in degrees.",
                kind: ParamSpecKind::Float {
                    min: -360.0,
                    max: 360.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "angle_spread",
                display_name: "Angle Spread",
                tooltip: "0 = all streaks parallel, 1 = any direction.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.05,
                },
            },
            ParamSpec {
                name: "bend",
                display_name: "Bend",
                tooltip: "Sideways bow, as a fraction of length; random sign per streak.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.03,
                },
            },
            ParamSpec {
                name: "taper",
                display_name: "Taper",
                tooltip: "Fraction of each half-length that fades out at the ends. 0 = hard cutoff.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.35,
                },
            },
            ParamSpec {
                name: "profile",
                display_name: "Profile",
                tooltip: "Cross-section shape across a streak's width.",
                kind: ParamSpecKind::Enum {
                    options: &["round", "soft", "flat"],
                    default: "round",
                },
            },
            ParamSpec {
                name: "brightness_random",
                display_name: "Brightness Random",
                tooltip: "Dims brightness by up to this fraction, randomly per streak.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "blend",
                display_name: "Blend",
                tooltip: "How overlapping streaks combine: Max keeps the brightest; Add sums \
                          them (clamped at 1).",
                kind: ParamSpecKind::Enum {
                    options: &["max", "add"],
                    default: "max",
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern without changing its character.",
                kind: ParamSpecKind::Seed { default: 1 },
            },
        ],
    },
    NodeDef {
        kind: "path",
        display_name: "Path",
        category: "Generators/Path",
        tooltip: "An editable vector curve — draw an ornament by hand (click a segment \
                  to add a point, drag handles to curve it) or paste an SVG, then feed it \
                  into a Stitch node.",
        inputs: &[
            // Seeds this node's own geometry from an upstream Path-type
            // source (a future SVG import node, or just another Path
            // node) the moment this one has no points of its own yet —
            // after that it's a normal, independently-editable copy, not
            // a live link, so connecting it never clobbers hand-edits.
            // Functional now even without a dedicated SVG node: chaining
            // two Path nodes together already exercises it.
            PortSpec {
                name: "reference",
                display_name: "Reference",
                port_type: PortType::Path,
            },
            // Not yet consumed by anything — reserved for the planned
            // Stitch node's own output (a single stitch's thread-physics
            // geometry), which this node will repeat/orient along its own
            // curve once that node and the distribution logic exist.
            // Connecting something here has no effect yet.
            PortSpec {
                name: "stamp",
                display_name: "Stamp",
                port_type: PortType::Gray,
            },
        ],
        // Preview Mask stays first so `mc_nodes::registry::resolve_texture_key`'s
        // "first declared output" default has an actual GPU texture to show —
        // the Path port is never written into the texture pool at all (see
        // `PortType::Path`'s own doc comment), so defaulting to it left the
        // 2D View and this node's own thumbnail permanently blank
        // ("(select a node to preview it)" even while selected). Same fix,
        // same reasoning as `thread_shape`'s own Height-stays-first comment.
        outputs: &[
            PortSpec {
                name: "preview_mask",
                display_name: "Preview Mask",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "path",
                display_name: "Path",
                port_type: PortType::Path,
            },
        ],
        params: &[
            ParamSpec {
                name: "path",
                display_name: "Path",
                tooltip: "The curve's own point/handle data — edited in the 2D View, not here.",
                kind: ParamSpecKind::Path,
            },
            ParamSpec {
                name: "show_curve",
                display_name: "Show Curve",
                tooltip: "Draw the curve's own thin outline into Preview Mask. Turn off once \
                          a Stamp is wired up and you only want the stamped pattern, not the \
                          guide line underneath it.",
                kind: ParamSpecKind::Bool { default: true },
            },
            ParamSpec {
                name: "stamp_spacing",
                display_name: "Stamp Spacing",
                tooltip: "Distance along the curve between repeats of whatever's connected \
                          to the Stamp input, in tile units. Smaller = more, denser copies \
                          \u{2014} a long or highly detailed curve (e.g. an Image Trace result) \
                          may need a much smaller value than a simple hand-drawn one to read as \
                          continuous rather than sparse.",
                kind: ParamSpecKind::Float {
                    // Was 0.005 — too coarse a floor for a curve with
                    // thousands of points (an Image Trace result, say):
                    // the owner hit this exact ceiling hand-stamping one,
                    // with no way to go denser even at the slider's own
                    // minimum.
                    min: 0.0002,
                    max: 0.5,
                    default: 0.1,
                },
            },
            ParamSpec {
                name: "stamp_size",
                display_name: "Stamp Size",
                tooltip: "How large each stamped copy is, in tile units (its own footprint \
                          width and height). Kept comfortably bigger than the curve's own \
                          thin outline by default, so a fresh Stamp connection is obviously \
                          visible rather than hiding under it.",
                kind: ParamSpecKind::Float {
                    min: 0.005,
                    max: 0.5,
                    default: 0.08,
                },
            },
            ParamSpec {
                name: "stamp_rotation",
                display_name: "Stamp Rotation",
                tooltip: "Extra rotation applied to every stamped copy, on top of following \
                          the curve's own direction (if Align to Curve is on).",
                kind: ParamSpecKind::Float {
                    min: -180.0,
                    max: 180.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "stamp_align_to_curve",
                display_name: "Align to Curve",
                tooltip: "On: each stamp rotates to follow the curve's own direction at that \
                          point (useful for anything directional, like a stitch). Off: every \
                          stamp keeps the same upright orientation.",
                kind: ParamSpecKind::Bool { default: true },
            },
        ],
    },
    NodeDef {
        kind: "image_trace",
        display_name: "Image Trace",
        category: "Generators/Path",
        tooltip: "Traces a grey or colour image into a smooth vector outline (a Potrace-style \
                  bitmap tracer). Feed Path into a Path node's own Reference input to edit or \
                  stamp along the traced curve.",
        inputs: &[
            // `Color`, not `Gray` — a Gray source already broadcasts into
            // a Color input (CLAUDE.md §5's implicit Gray->Color
            // conversion), so this genuinely accepts either, matching
            // TRACE_SPEC.md's own "a grey or colour image."
            PortSpec {
                name: "image",
                display_name: "Image",
                port_type: PortType::Color,
            },
        ],
        // Preview Mask stays first, same reasoning as the Path node's own
        // comment just above: the Path output is never written into the
        // texture pool (PortType::Path bypasses it entirely), so the 2D
        // View needs a real texture port to default to.
        outputs: &[
            PortSpec {
                name: "preview_mask",
                display_name: "Preview Mask",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "path",
                display_name: "Path",
                port_type: PortType::Path,
            },
        ],
        params: &[
            // Written by this node's own dispatch (the traced result),
            // not user-editable here — same role and same `ParamSpecKind`
            // as the Path node's own `path` param.
            ParamSpec {
                name: "path",
                display_name: "Path",
                tooltip: "The traced curve's own point/handle data — computed from Image, not \
                          edited here.",
                kind: ParamSpecKind::Path,
            },
            ParamSpec {
                name: "threshold",
                display_name: "Threshold",
                tooltip: "Luminance below this becomes the traced (black) area.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "invert",
                display_name: "Invert",
                tooltip: "Trace the light areas instead of the dark ones.",
                kind: ParamSpecKind::Bool { default: false },
            },
            ParamSpec {
                name: "turn_policy",
                display_name: "Turn Policy",
                tooltip: "How to resolve a diagonal ambiguity while walking a boundary \
                          (TRACE_SPEC.md's own \"Turn policies\"). Minority is the default and \
                          usually the most natural-looking; the others matter mostly for \
                          checkerboard-like patterns.",
                kind: ParamSpecKind::Enum {
                    options: &crate::image_trace::TurnPolicy::NAMES,
                    default: "Minority",
                },
            },
            ParamSpec {
                name: "despeckle",
                display_name: "Despeckle",
                tooltip: "Drop traced regions smaller than this many pixels — removes stray \
                          flecks from a noisy source image.",
                kind: ParamSpecKind::Int {
                    min: 0,
                    max: 1000,
                    default: 2,
                },
            },
            ParamSpec {
                name: "alpha_max",
                display_name: "Alpha Max",
                tooltip: "The corner-vs-curve threshold. Lower = sharper corners kept as \
                          corners; below 0 keeps every vertex a corner (a pure polygon); above \
                          about 1.33 smooths every vertex into a curve (no corners at all).",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 2.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "optimize",
                display_name: "Optimize",
                tooltip: "Join consecutive curves where one smooth Bezier fits within \
                          Tolerance, for a cleaner curve with fewer points.",
                kind: ParamSpecKind::Bool { default: true },
            },
            ParamSpec {
                name: "opt_tolerance",
                display_name: "Tolerance",
                tooltip: "How far a joined curve may drift from the original traced outline. \
                          Only matters while Optimize is on.",
                kind: ParamSpecKind::Float {
                    min: 0.01,
                    max: 2.0,
                    default: 0.2,
                },
            },
        ],
    },
    NodeDef {
        kind: "path_simplify",
        display_name: "Path Simplify",
        category: "Generators/Path",
        tooltip: "Reduces a Path's own point count (Ramer-Douglas-Peucker vertex decimation) \u{2014} \
                  useful before stamping or exporting a very dense curve (an Image Trace result, \
                  say). Corner points always survive, regardless of Tolerance.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Path",
            port_type: PortType::Path,
        }],
        // Preview Mask stays first, same reasoning as Path's and Image
        // Trace's own comments above: the Path output is never written
        // into the texture pool, so the 2D View needs a real texture
        // port to default to.
        outputs: &[
            PortSpec {
                name: "preview_mask",
                display_name: "Preview Mask",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "path",
                display_name: "Path",
                port_type: PortType::Path,
            },
        ],
        params: &[
            // Written by this node's own dispatch (the simplified
            // result), not user-editable here — same role and same
            // `ParamSpecKind` as the Path node's own and Image Trace's
            // own `path` params.
            ParamSpec {
                name: "path",
                display_name: "Path",
                tooltip: "The simplified curve's own point/handle data \u{2014} computed from the \
                          Path input, not edited here.",
                kind: ParamSpecKind::Path,
            },
            ParamSpec {
                name: "tolerance",
                display_name: "Tolerance",
                tooltip: "How far a dropped point may have strayed from the straight line \
                          between its surviving neighbors, in tile units. Higher = fewer \
                          points, looser fit.",
                kind: ParamSpecKind::Float {
                    min: 0.0001,
                    max: 0.1,
                    default: 0.002,
                },
            },
        ],
    },
    NodeDef {
        kind: "path_to_mask",
        display_name: "Path to Mask",
        category: "Generators/Path",
        tooltip: "Rasterizes a Path into a real Gray output \u{2014} Fill (nonzero-winding \
                  polygon fill, holes handled automatically) or Outline (a constant-width \
                  stroke).",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Path",
            port_type: PortType::Path,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Mask",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "mode",
                display_name: "Mode",
                tooltip: "Fill: solid interior, holes left blank automatically (nonzero \
                          winding). Outline: a stroke of Outline Width along every subpath.",
                kind: ParamSpecKind::Enum {
                    options: &["Fill", "Outline"],
                    default: "Fill",
                },
            },
            ParamSpec {
                name: "outline_width",
                display_name: "Outline Width",
                tooltip: "Stroke width in tile units. Only used in Outline mode.",
                kind: ParamSpecKind::Float {
                    min: 0.0005,
                    max: 0.2,
                    default: 0.01,
                },
            },
        ],
    },
    NodeDef {
        kind: "path_to_sdf",
        display_name: "Path to SDF",
        category: "Generators/Path",
        tooltip: "Turns a Path into a real Sdf \u{2014} Fill (closed subpaths, even-odd rule) or \
                  Stroke (a tube of the set Width along any subpath). Engraved lines, cutouts, \
                  and custom panel shapes for the Hard-Surface nodes.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Path",
            port_type: PortType::Path,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Sdf",
            port_type: PortType::Sdf,
        }],
        params: &[
            ParamSpec {
                name: "mode",
                display_name: "Mode",
                tooltip: "Fill: closed subpaths become solids (even-odd rule); open subpaths \
                          are ignored for the fill itself. Stroke: a tube of Width along any \
                          subpath, open or closed.",
                kind: ParamSpecKind::Enum {
                    options: &crate::path_to_sdf::PathSdfMode::NAMES,
                    default: "Fill",
                },
            },
            ParamSpec {
                name: "width",
                display_name: "Width",
                tooltip: "Stroke mode only \u{2014} tube width in tile units.",
                kind: ParamSpecKind::Float {
                    min: 0.0005,
                    max: 0.2,
                    default: 0.012,
                },
            },
        ],
    },
    NodeDef {
        kind: "fasteners",
        display_name: "Fasteners",
        category: "Generators/Path",
        tooltip: "Places rivets, bolts, or screws evenly along a Path, seated on an optional \
                  base height.",
        inputs: &[
            PortSpec {
                name: "path",
                display_name: "Path",
                port_type: PortType::Path,
            },
            PortSpec {
                name: "base",
                display_name: "Base",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[
            PortSpec {
                name: "height",
                display_name: "Height",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "fastener_id",
                display_name: "Fastener ID",
                port_type: PortType::Gray,
            },
        ],
        params: &[
            ParamSpec {
                name: "kind",
                display_name: "Kind",
                tooltip: "Which fastener head to place.",
                kind: ParamSpecKind::Enum {
                    options: &crate::fasteners::FastenerKind::NAMES,
                    default: "Rivet Dome",
                },
            },
            ParamSpec {
                name: "spacing",
                display_name: "Spacing",
                tooltip: "Target distance between fasteners along the path, tile units. Every \
                          corner always gets one regardless.",
                kind: ParamSpecKind::Float {
                    min: 0.01,
                    max: 0.3,
                    default: 0.05,
                },
            },
            ParamSpec {
                name: "head_radius",
                display_name: "Head Radius",
                tooltip: "Tile units.",
                kind: ParamSpecKind::Float {
                    min: 0.0005,
                    max: 0.05,
                    default: 0.006,
                },
            },
            ParamSpec {
                name: "head_height",
                display_name: "Head Height",
                tooltip: "Relative to the bevel depth units Bevel Profile uses (1.0 = a full \
                          panel depth).",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.6,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes which fasteners are missing and how each one is rotated.",
                kind: ParamSpecKind::Seed { default: 1 },
            },
            ParamSpec {
                name: "missing",
                display_name: "Missing",
                tooltip: "Probability a given fastener is absent.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "rotate",
                display_name: "Rotate",
                tooltip: "Each fastener gets its own random rotation. Off keeps every head \
                          upright.",
                kind: ParamSpecKind::Bool { default: true },
            },
        ],
    },
    NodeDef {
        kind: "knit",
        display_name: "Knit",
        category: "Generators/Patterns",
        tooltip: "Weft-knitted fabric built from real loop geometry \u{2014} jersey, ribs, \
                  purl and interlock, with up to 4 plies twisted along each loop.",
        // No Colour input/output: KNIT_NODE_SPEC.md §7 has one (per-stitch
        // pixelated sampling, jacquard/intarsia), but the owner cut it as
        // redundant with Uniform Color + Blend + a generic Color output
        // already covering the same ground more flexibly — not worth a
        // dedicated pin on every Knit node.
        inputs: &[],
        // Height stays first, same reasoning as Thread's own outputs list
        // above (mc_nodes::registry::resolve_texture_key's "first declared
        // output" default).
        outputs: &[
            PortSpec {
                name: "height",
                display_name: "Height",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "ply_id",
                display_name: "Ply ID",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "stitch_id",
                display_name: "Stitch ID",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "stitch_type",
                display_name: "Stitch Type",
                port_type: PortType::Gray,
            },
        ],
        params: &[
            ParamSpec {
                name: "preset",
                display_name: "Preset",
                tooltip: "Which stitch chart to knit. Editing Chart below switches this to \
                          Custom.",
                kind: ParamSpecKind::Enum {
                    options: &crate::knit::KnitPreset::NAMES,
                    default: "Jersey",
                },
            },
            ParamSpec {
                name: "chart",
                display_name: "Chart",
                tooltip: "A custom stitch chart: rows of K/P, top to bottom, separated by /. \
                          For example KKPP/PPKK. Leave empty to use the Preset's own chart.",
                kind: ParamSpecKind::Text { default: "" },
            },
            ParamSpec {
                name: "side",
                display_name: "Side",
                tooltip: "Back swaps every K and P in the chart \u{2014} the reverse side of \
                          the same fabric.",
                kind: ParamSpecKind::Enum {
                    options: &crate::knit::Side::NAMES,
                    default: "Front",
                },
            },
            ParamSpec {
                name: "wales",
                display_name: "Wales",
                tooltip: "Stitches across the tile (one per needle). Snapped to a multiple of \
                          the chart's own width.",
                kind: ParamSpecKind::Int {
                    min: 2,
                    max: 64,
                    default: 8,
                },
            },
            ParamSpec {
                name: "auto_courses",
                display_name: "Auto Courses",
                tooltip: "Keeps the natural stitch shape (course spacing set from the wale \
                          width and Loop Height). Turn off to set Courses by hand.",
                kind: ParamSpecKind::Bool { default: true },
            },
            ParamSpec {
                name: "courses",
                display_name: "Courses",
                tooltip: "Rows down the tile. Snapped to a multiple of the chart's own \
                          height. Only used while Auto Courses is off.",
                kind: ParamSpecKind::Int {
                    min: 2,
                    max: 96,
                    default: 11,
                },
            },
            ParamSpec {
                name: "tension",
                display_name: "Tension",
                tooltip: "How much yarn per loop relative to its own thickness. Sets both \
                          Thickness and Loop Height unless either is overridden below.",
                kind: ParamSpecKind::Enum {
                    options: &crate::knit::Tension::NAMES,
                    default: "Normal",
                },
            },
            ParamSpec {
                name: "thickness",
                display_name: "Thickness",
                tooltip: "Advanced: yarn radius as a fraction of course spacing, overriding \
                          Tension. 0 = derive from Tension.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.26,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "loop_height",
                display_name: "Loop Height",
                tooltip: "Advanced: loop half-height as a multiple of course spacing, \
                          overriding Tension. 0 = derive from Tension.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.5,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "roundness",
                display_name: "Roundness",
                tooltip: "0 is a flat sine wave; higher rounds the loop into the \u{3a9}-shaped \
                          neck real knit stitches have.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 2.0,
                    default: 1.5,
                },
            },
            ParamSpec {
                name: "depth",
                display_name: "Depth",
                tooltip: "How far the yarn dips in and out of the fabric plane, as a multiple \
                          of its own radius.",
                kind: ParamSpecKind::Float {
                    min: 0.5,
                    max: 2.0,
                    default: 1.25,
                },
            },
            ParamSpec {
                name: "plies",
                display_name: "Plies",
                tooltip: "How many strands are twisted together into the yarn. 1 is a single \
                          (unplied) yarn.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 4,
                    default: 1,
                },
            },
            ParamSpec {
                name: "ply_direction",
                display_name: "Ply Twist",
                tooltip: "Which way the plies spiral (textile S/Z-twist terms), same meaning \
                          as the Thread node's own Ply Twist.",
                kind: ParamSpecKind::Enum {
                    options: &crate::thread_shape::TwistDirection::NAMES,
                    default: "S",
                },
            },
            ParamSpec {
                name: "twist_angle",
                display_name: "Twist Angle",
                tooltip: "The ply helix angle, same meaning as the Thread node's own Twist \
                          Angle.",
                kind: ParamSpecKind::Float {
                    min: 10.0,
                    max: 50.0,
                    default: 30.0,
                },
            },
            ParamSpec {
                name: "rib_depth",
                display_name: "Rib Depth",
                tooltip: "How far purl wales (or courses, for purl/links-links fabric) recede \
                          behind the knit ones, as a multiple of yarn radius. Only visible on \
                          ribs and purl.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 3.0,
                    default: 1.5,
                },
            },
            ParamSpec {
                name: "rib_compaction",
                display_name: "Rib Compaction",
                tooltip: "How much narrower a rib's purl wales are than its knit ones. \
                          Higher can hide the purl wale almost entirely. Ribs only.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.8,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the per-stitch Stitch ID pattern.",
                kind: ParamSpecKind::Seed { default: 1 },
            },
        ],
    },
    NodeDef {
        kind: "shape",
        display_name: "Shape",
        category: "Generators/Patterns",
        tooltip: "A single disc, square, polygon, star, or leaf, centered in the tile.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "kind",
                display_name: "Kind",
                tooltip: "Which shape to draw.",
                kind: ParamSpecKind::Enum {
                    options: &crate::shape::ShapeKind::NAMES,
                    default: "Disc",
                },
            },
            ParamSpec {
                name: "sides",
                display_name: "Sides",
                tooltip: "Vertex/point count for Polygon and Star.",
                kind: ParamSpecKind::Int {
                    min: 3,
                    max: 12,
                    default: 5,
                },
            },
            ParamSpec {
                name: "size",
                display_name: "Size",
                tooltip: "How large the shape is. Past 1.0 the shape grows larger than the \
                          tile itself, cropped by its edge.",
                kind: ParamSpecKind::Float {
                    min: 0.01,
                    max: 3.0,
                    default: 0.6,
                },
            },
            ParamSpec {
                name: "scale_x",
                display_name: "Scale X",
                tooltip: "Independent horizontal stretch on top of Size — 1.0 is no stretch. \
                          Lower makes a Disc into a tall ellipse or a Square into a tall \
                          rectangle; higher makes it wide. Applied before Rotation, so a \
                          stretched shape tilts as a whole when rotated.",
                kind: ParamSpecKind::Float {
                    min: 0.01,
                    max: 4.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "scale_y",
                display_name: "Scale Y",
                tooltip: "Independent vertical stretch on top of Size — same as Scale X, for \
                          the other axis.",
                kind: ParamSpecKind::Float {
                    min: 0.01,
                    max: 4.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "softness",
                display_name: "Softness",
                tooltip: "How soft the edge is.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "rotation",
                display_name: "Rotation",
                tooltip: "Fraction of a full turn.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "offset_x",
                display_name: "Offset X",
                tooltip: "Moves the shape away from the tile's center, tile units. Wraps past \
                          0.5, so the shape reappears on the opposite edge \u{2014} no separate \
                          Transform 2D node needed just to reposition it.",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "offset_y",
                display_name: "Offset Y",
                tooltip: "Same as Offset X, for the vertical axis.",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "point_depth",
                display_name: "Point Depth",
                tooltip: "Star only: 1.0 is a plain polygon (e.g. a triangle at 3 sides); lowering it unfolds the same outline into sharper star points.",
                kind: ParamSpecKind::Float {
                    min: 0.05,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "curvature_top",
                display_name: "Curvature Top",
                tooltip: "Leaf only: how pointed (high) or rounded (low) its top tip is.",
                kind: ParamSpecKind::Float {
                    min: 0.2,
                    max: 4.0,
                    default: 0.6,
                },
            },
            ParamSpec {
                name: "curvature_bottom",
                display_name: "Curvature Bottom",
                tooltip: "Leaf only: how pointed (high) or rounded (low) its bottom tip is.",
                kind: ParamSpecKind::Float {
                    min: 0.2,
                    max: 4.0,
                    default: 0.6,
                },
            },
            ParamSpec {
                name: "curvature_middle",
                display_name: "Curvature Middle",
                tooltip: "Leaf only: how pointed (high) or rounded (low) its sides are through the middle (the equator), independently from either tip.",
                kind: ParamSpecKind::Float {
                    min: 0.2,
                    max: 4.0,
                    default: 0.6,
                },
            },
            ParamSpec {
                name: "leaf_width",
                display_name: "Leaf Width",
                tooltip: "Leaf only: how wide it is at its fattest point (the equator), as a fraction of its own height.",
                kind: ParamSpecKind::Float {
                    min: 0.25,
                    max: 0.9,
                    default: 0.55,
                },
            },
            ParamSpec {
                name: "middle_position",
                display_name: "Middle Position",
                tooltip: "Leaf only: where the equator (widest point) sits between the top tip (low) and bottom tip (high). 0.5 is centered.",
                kind: ParamSpecKind::Float {
                    min: 0.15,
                    max: 0.85,
                    default: 0.5,
                },
            },
        ],
    },
    NodeDef {
        kind: "circular_pattern",
        display_name: "Circular Pattern",
        category: "Generators/Patterns",
        tooltip: "Distributes a Stamp input evenly around a circle \u{2014} a polar array, with \
                  optional position/rotation/scale jitter.",
        inputs: &[PortSpec {
            name: "stamp",
            display_name: "Stamp",
            port_type: PortType::Gray,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "count",
                display_name: "Count",
                tooltip: "How many copies to place around the circle.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 8,
                },
            },
            ParamSpec {
                name: "radius",
                display_name: "Radius",
                tooltip: "Distance from the tile center to each copy, in tile units.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.5,
                    default: 0.35,
                },
            },
            ParamSpec {
                name: "offset_x",
                display_name: "Offset X",
                tooltip: "Moves the circle's own center away from the tile's, tile units.",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "offset_y",
                display_name: "Offset Y",
                tooltip: "Same as Offset X, for the vertical axis.",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "stamp_size",
                display_name: "Stamp Size",
                tooltip: "How large each copy is, in tile units.",
                kind: ParamSpecKind::Float {
                    min: 0.005,
                    max: 0.5,
                    default: 0.08,
                },
            },
            ParamSpec {
                name: "start_angle",
                display_name: "Start Angle",
                tooltip: "Where the first copy sits, in degrees \u{2014} the rest follow evenly \
                          spaced from there.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 360.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "align_to_tangent",
                display_name: "Align to Tangent",
                tooltip: "On: each copy rotates to face outward from the center. Off: every \
                          copy keeps the same upright orientation.",
                kind: ParamSpecKind::Bool { default: true },
            },
            ParamSpec {
                name: "rotation_jitter",
                display_name: "Rotation Jitter",
                tooltip: "Random extra rotation per copy, up to this many degrees either way.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 180.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "scale_jitter",
                display_name: "Scale Jitter",
                tooltip: "Random size variation per copy, as a fraction of Stamp Size either \
                          way (0.3 means each copy is somewhere between 70% and 130% size).",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "position_jitter_radial",
                display_name: "Position Jitter (Radial)",
                tooltip: "Random distance each copy may move toward or away from the center, \
                          in tile units.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.2,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "position_jitter_tangential",
                display_name: "Position Jitter (Tangential)",
                tooltip: "Random distance each copy may slide sideways around the circle, in \
                          tile units.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.2,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes every jitter pattern at once, without changing Count or \
                          Radius.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            ParamSpec {
                name: "wrap",
                display_name: "Wrap",
                tooltip: "Copies near the tile edge also draw their wrapped-around neighbors, \
                          so the pattern tiles seamlessly.",
                kind: ParamSpecKind::Bool { default: true },
            },
        ],
    },
    NodeDef {
        kind: "panel_layout",
        display_name: "Panel Layout",
        category: "Generators/Patterns",
        tooltip: "A tileable guillotine-subdivided panel layout \u{2014} every pixel belongs to \
                  exactly one panel, no flood fill needed.",
        inputs: &[],
        outputs: &[
            PortSpec {
                name: "sdf",
                display_name: "Sdf",
                port_type: PortType::Sdf,
            },
            PortSpec {
                name: "panel_id",
                display_name: "Panel ID",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "panel_index",
                display_name: "Panel Index",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "uv_u",
                display_name: "Panel UV U",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "uv_v",
                display_name: "Panel UV V",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "seam",
                display_name: "Seam",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "path",
                display_name: "Path",
                port_type: PortType::Path,
            },
        ],
        params: &[
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the whole panel layout.",
                kind: ParamSpecKind::Seed { default: 3 },
            },
            ParamSpec {
                name: "grid_div",
                display_name: "Grid Divisions",
                tooltip: "Splits snap to 1 / this fraction of the tile.",
                kind: ParamSpecKind::Int {
                    min: 2,
                    max: 64,
                    default: 16,
                },
            },
            ParamSpec {
                name: "min_size",
                display_name: "Min Size",
                tooltip: "Smallest a panel's side is allowed to be, tile units.",
                kind: ParamSpecKind::Float {
                    min: 0.01,
                    max: 0.5,
                    default: 0.125,
                },
            },
            ParamSpec {
                name: "max_depth",
                display_name: "Max Depth",
                tooltip: "Hard cap on how many times a panel can be split again.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 10,
                    default: 6,
                },
            },
            ParamSpec {
                name: "min_depth",
                display_name: "Min Depth",
                tooltip: "Always splits at least this many times before chance decides whether \
                          to keep going.",
                kind: ParamSpecKind::Int {
                    min: 0,
                    max: 10,
                    default: 3,
                },
            },
            ParamSpec {
                name: "split_chance",
                display_name: "Split Chance",
                tooltip: "Past Min Depth, the chance a panel splits again.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.6,
                },
            },
            ParamSpec {
                name: "long_axis_bias",
                display_name: "Long Axis Bias",
                tooltip: "How often a split cuts across a panel's longer side rather than its \
                          shorter one.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.8,
                },
            },
            ParamSpec {
                name: "gap",
                display_name: "Gap",
                tooltip: "Seam width between panels, tile units.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.05,
                    default: 0.006,
                },
            },
            ParamSpec {
                name: "corner_radius",
                display_name: "Corner Radius",
                tooltip: "Rounds each panel's own corners.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.05,
                    default: 0.004,
                },
            },
            ParamSpec {
                name: "fastener_margin",
                display_name: "Fastener Margin",
                tooltip: "How far the Path output insets from each panel's own edge \u{2014} \
                          feeds Fasteners or any Path node.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.05,
                    default: 0.014,
                },
            },
            ParamSpec {
                name: "offset",
                display_name: "Offset",
                tooltip: "A random global offset so seams cross the tile edge, instead of the \
                          grid always starting at a corner.",
                kind: ParamSpecKind::Bool { default: true },
            },
        ],
    },
    NodeDef {
        kind: "type_text",
        display_name: "Type",
        category: "Generators/Patterns",
        tooltip: "Renders text, in any font installed on this machine, as a Gray mask.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "text",
                display_name: "Text",
                tooltip: "The text to render.",
                kind: ParamSpecKind::Text { default: "Type" },
            },
            ParamSpec {
                name: "font_family",
                display_name: "Font",
                tooltip: "Which installed font to use. System Default picks a generic \
                          sans-serif.",
                kind: ParamSpecKind::FontFamily,
            },
            ParamSpec {
                name: "size",
                display_name: "Size",
                tooltip: "Text height, in tile units.",
                kind: ParamSpecKind::Float {
                    min: 0.02,
                    max: 0.9,
                    default: 0.2,
                },
            },
            ParamSpec {
                name: "tracking",
                display_name: "Tracking",
                tooltip: "Extra spacing added between every letter, in tile units.",
                kind: ParamSpecKind::Float {
                    min: -0.05,
                    max: 0.2,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "align",
                display_name: "Align",
                tooltip: "Horizontal alignment of the text within the tile.",
                kind: ParamSpecKind::Enum {
                    options: &crate::type_text::HorizontalAlign::NAMES,
                    default: "Center",
                },
            },
        ],
    },
    NodeDef {
        kind: "stitch",
        display_name: "Stitch",
        category: "Generators/Patterns",
        tooltip: "A single embroidery stitch's own real thread-physics geometry (height, drape, \
                  plies) spanning one tile — like Shape, self-contained, with no curve of its \
                  own. Feed the output into a Path node's own Stamp input to repeat it along a \
                  real curve.",
        inputs: &[],
        outputs: &[
            PortSpec {
                name: "height",
                display_name: "Height",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "thread_id",
                display_name: "Thread ID",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "order",
                display_name: "Order",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "ply_id",
                display_name: "Ply ID",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "direction",
                display_name: "Direction",
                port_type: PortType::Color,
            },
        ],
        params: &[
            ParamSpec {
                name: "stitch",
                display_name: "Stitch",
                tooltip: "Which embroidery stitch type to simulate. Back and Split aren't \
                          their own options — with exactly one stitch spanning its own whole \
                          tile, both render pixel-identical to Running, so they're folded \
                          into it rather than offering a choice with no visible difference.",
                kind: ParamSpecKind::Enum {
                    options: &["running", "bean", "stem", "zigzag", "satin", "french_knot"],
                    default: "running",
                },
            },
            ParamSpec {
                name: "look",
                display_name: "Look",
                tooltip: "Hand: stranded cotton, nearly flat twist, small human irregularity. \
                          Machine: tight 40wt twist, perfect spacing, satin zig-zags between \
                          its own rails instead of lying flat. Picking a Look resets Thread \
                          Diameter, Plies, Twist Angle, Regularity, Stem Width and Density to \
                          its own defaults — each stays independently editable afterward.",
                kind: ParamSpecKind::Enum {
                    options: &["hand", "machine"],
                    default: "hand",
                },
            },
            ParamSpec {
                name: "stitch_length_mm",
                display_name: "Stitch Length (mm)",
                tooltip: "How far apart the needle holes are, roughly — the exact meaning \
                          depends on the stitch type (RSN satin is about 3mm by hand).",
                kind: ParamSpecKind::Float {
                    min: 0.5,
                    max: 20.0,
                    default: 3.0,
                },
            },
            ParamSpec {
                name: "width_mm",
                display_name: "Width (mm)",
                tooltip: "Zigzag/satin column width.",
                kind: ParamSpecKind::Float {
                    min: 0.5,
                    max: 30.0,
                    default: 2.5,
                },
            },
            ParamSpec {
                name: "stem_width_mm",
                display_name: "Stem Width (mm)",
                tooltip: "Stem stitch only: how far the thread slants to each side.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 5.0,
                    default: 0.45,
                },
            },
            ParamSpec {
                name: "density_mm",
                display_name: "Density (mm)",
                tooltip: "Satin stitch only: spacing between its own parallel stitches — \
                          smaller is denser coverage.",
                kind: ParamSpecKind::Float {
                    min: 0.05,
                    max: 5.0,
                    default: 0.3825,
                },
            },
            ParamSpec {
                name: "bean_repeats",
                display_name: "Bean Repeats",
                tooltip: "Bean stitch only: 1 = each stitch worked three times (triple), 2 = \
                          five times (quintuple).",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 4,
                    default: 1,
                },
            },
            ParamSpec {
                name: "stem_side",
                display_name: "Stem Side",
                tooltip: "Stem stitch only: which way the rope twists — Stem leans \"/\", \
                          Outline leans \"\\\".",
                kind: ParamSpecKind::Enum {
                    options: &["stem", "outline"],
                    default: "stem",
                },
            },
            ParamSpec {
                name: "zigzag_angle",
                display_name: "Zigzag Angle",
                tooltip: "Zigzag stitch only: rotates which direction the holes alternate \
                          toward.",
                kind: ParamSpecKind::Float {
                    min: -90.0,
                    max: 90.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "pull_comp_mm",
                display_name: "Pull Compensation (mm)",
                tooltip: "Zigzag/satin only: widens the column on each side, compensating for \
                          thread pulling the fabric inward.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 3.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "thread_mm",
                display_name: "Thread Diameter (mm)",
                tooltip: "How thick the thread itself is.",
                kind: ParamSpecKind::Float {
                    min: 0.05,
                    max: 3.0,
                    default: 0.45,
                },
            },
            ParamSpec {
                name: "plies",
                display_name: "Plies",
                tooltip: "How many strands are twisted together into one thread.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 6,
                    default: 2,
                },
            },
            ParamSpec {
                name: "twist_angle",
                display_name: "Twist Angle",
                tooltip: "How tightly the plies twist around each other — low is nearly flat, \
                          high is a tight cord.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 60.0,
                    default: 10.0,
                },
            },
            ParamSpec {
                name: "ply_direction",
                display_name: "Ply Direction",
                tooltip: "Which way the plies twist — S or Z, same convention as the Thread \
                          node's own.",
                kind: ParamSpecKind::Enum {
                    options: &["S", "Z"],
                    default: "S",
                },
            },
            ParamSpec {
                name: "regularity",
                display_name: "Regularity",
                tooltip: "1.0 = machine-perfect spacing. Lower values jitter each needle hole \
                          by hand-sewn irregularity, both along and across the stitch line.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.7,
                },
            },
            ParamSpec {
                name: "knot_wraps",
                display_name: "Knot Wraps",
                tooltip: "French Knot only: how many times the thread wraps around the \
                          needle. 1 is a pin dot, 2 is the standard bead, 3 is medium, 4 is a \
                          \"tower\" (allowed, but loose and floppy).",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 4,
                    default: 2,
                },
            },
            ParamSpec {
                name: "knot_strands",
                display_name: "Knot Strands",
                tooltip: "French Knot only: how many strands of floss make up the knot — the \
                          real size control (for a bigger knot, add strands, not wraps).",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 6,
                    default: 2,
                },
            },
            ParamSpec {
                name: "knot_strand_mm",
                display_name: "Knot Strand Diameter (mm)",
                tooltip: "French Knot only: the diameter of one strand of stranded cotton. \
                          Clamped to a narrower range than the reference's own 0.1-0.5mm — \
                          past these ends a knot stops reading as a knot (too thin to show \
                          its own coil, or so thick it dominates whatever it's stamped onto).",
                kind: ParamSpecKind::Float {
                    min: 0.15,
                    max: 0.3,
                    default: 0.2,
                },
            },
            ParamSpec {
                name: "knot_looseness",
                display_name: "Knot Looseness",
                tooltip: "French Knot only: 0 is pulled tight and neat, 1 is loose and loopy \
                          (the wraps don't sit snugly around the needle).",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.15,
                },
            },
            ParamSpec {
                name: "knot_look",
                display_name: "Knot Look",
                tooltip: "French Knot only: Hand is stranded cotton with small human \
                          irregularity. Neat is very even (sample-book, machine-like). Loose \
                          is beginner/textured work. Sets Regularity and Knot Twist Angle \
                          below, both still independently editable afterward.",
                kind: ParamSpecKind::Enum {
                    options: &["hand", "neat", "loose"],
                    default: "hand",
                },
            },
            ParamSpec {
                name: "knot_twist_angle",
                display_name: "Knot Twist Angle",
                tooltip: "French Knot only: how tightly the strands twist around each other. \
                          Clamped tighter than Stitch's own Twist Angle (10-40, not 0-60) — a \
                          knot's own strands are short and tightly bundled, so the extremes of \
                          the full range read as either indistinguishable from straight or \
                          implausibly over-twisted.",
                kind: ParamSpecKind::Float {
                    min: 10.0,
                    max: 40.0,
                    default: 10.0,
                },
            },
            ParamSpec {
                name: "height_range",
                display_name: "Height Range",
                tooltip: "How many thread diameters map to a Height output of 1.0. A thicker \
                          thread always looks taller at the same setting — this never \
                          re-normalizes per image. Raise it to deliberately dim/flatten the \
                          stitch; at the default (1.0) a single stitch's own highlight ridge \
                          reaches full brightness.",
                kind: ParamSpecKind::Float {
                    min: 1.0,
                    max: 20.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "knot_height_range",
                display_name: "Knot Height Range",
                tooltip: "French Knot only: how many bundle diameters map to a Height output \
                          of 1.0. Unlike a single stitch, a knot's own wraps genuinely pile up \
                          on top of each other — a value tuned for a flat isolated stitch \
                          clips that stacking and washes out the individual strands, so this \
                          defaults higher (4.0, matching the reference). Clamped tighter than \
                          Stitch's own Height Range (4-8, not 1-20) — below 4 the strands wash \
                          out, and much above 8 the knot reads as flat.",
                kind: ParamSpecKind::Float {
                    min: 4.0,
                    max: 8.0,
                    default: 4.0,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes which thread gets which random Thread ID/twist phase, \
                          without changing the stitch pattern itself.",
                kind: ParamSpecKind::Seed { default: 1 },
            },
        ],
    },
    NodeDef {
        kind: "gradient_linear",
        display_name: "Gradient Linear",
        category: "Generators/Patterns",
        tooltip: "A ramp from black to white, rotatable around the tile's center.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[ParamSpec {
            name: "rotation",
            display_name: "Rotation",
            tooltip: "Ramp direction. Limited to quarter turns — the only angles that keep the tile's edges lining up with its neighbor's.",
            kind: ParamSpecKind::Enum {
                options: &crate::gradient_linear::ROTATION_OPTIONS,
                default: "0°",
            },
        }],
    },
    NodeDef {
        kind: "gradient_radial",
        display_name: "Gradient Radial",
        category: "Generators/Patterns",
        tooltip: "A ramp from black at the center to white at the given radius.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[ParamSpec {
            name: "radius",
            display_name: "Radius",
            tooltip: "Distance from center where the ramp reaches white.",
            kind: ParamSpecKind::Float {
                min: 0.05,
                max: 1.0,
                default: 0.5,
            },
        }],
    },
    NodeDef {
        kind: "uniform_color",
        display_name: "Uniform Color",
        category: "Generators/Patterns",
        tooltip: "A flat, solid color fill.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[ParamSpec {
            name: "color",
            display_name: "Color",
            tooltip: "The fill color.",
            kind: ParamSpecKind::Color {
                default: [0.5, 0.5, 0.5, 1.0],
            },
        }],
    },
    NodeDef {
        kind: "image_import",
        display_name: "Image Import",
        category: "Generators/Patterns",
        tooltip: "Load your own color or grayscale image to use as a texture or pattern \
                  source.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "image",
                display_name: "Image",
                tooltip: "Stretches to fill the tile (not tiled/repeated). Treated as a \
                          standard sRGB-encoded color image \u{2014} most photos and textures \
                          are.",
                kind: ParamSpecKind::ImagePath,
            },
            ParamSpec {
                name: "embed_in_graph",
                display_name: "Embed in Graph File",
                tooltip: "Off: the saved .mcgraph just remembers this file's path \u{2014} \
                          small file, but breaks if you move/rename the image or share the \
                          graph without it. On: the image's data is copied directly into the \
                          .mcgraph, making it fully self-contained to share, at the cost of a \
                          larger file. Only affects the next time you click Load Image \u{2014} \
                          toggling it afterward doesn't retroactively re-embed or drop what's \
                          already loaded.",
                kind: ParamSpecKind::Bool { default: false },
            },
            ParamSpec {
                name: "grayscale",
                display_name: "Convert to Grayscale",
                tooltip: "Converts the loaded image to grayscale and switches this node's \
                          output to Gray, so it can be wired into a Gray-only input (e.g. \
                          Pattern Generator's Shape).",
                kind: ParamSpecKind::Bool { default: false },
            },
        ],
    },
    NodeDef {
        kind: "invert",
        display_name: "Invert",
        category: "Filters/Adjustments",
        tooltip: "Flips light and dark: output = 1 - input.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[],
    },
    NodeDef {
        kind: "histogram_scan",
        display_name: "Value Range",
        category: "Filters/Blending & Masking",
        tooltip: "Threshold/window the input around a position.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "position",
                display_name: "Position",
                tooltip: "Where the transition is centered.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "contrast",
                display_name: "Contrast",
                tooltip: "How sharp the transition is (1 = hard edge).",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "blur",
        display_name: "Blur",
        category: "Filters/Spatial",
        tooltip: "Gaussian blur — softens detail.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "radius",
                display_name: "Radius",
                tooltip: "Blur radius, in UV space.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.1,
                    default: 0.01,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "distance_spread",
        display_name: "Distance",
        category: "Filters/Spatial",
        tooltip: "Spreads Source's value outward from Mask, fading to zero past Distance — dirt \
                  pooling near seams, decay radiating from damage.",
        inputs: &[
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "source",
                display_name: "Source",
                port_type: PortType::Color,
            },
        ],
        outputs: &[
            PortSpec {
                name: "output",
                display_name: "Output",
                port_type: PortType::Color,
            },
            // Second, not first — `registry::resolve_texture_key`'s "first
            // declared output" convention is what keeps an existing saved
            // graph's own connection from this node's own default output
            // port resolving to the same texture after this one was
            // added (same reasoning as Thread's own Height staying
            // first).
            // A stable, distinct value per disconnected island in Mask —
            // the same per-blob identity Luminance Jitter already uses
            // internally, exposed directly. Independent of brightness:
            // two stones can look identical and still get different IDs,
            // or vary internally and still share one. Feed into Edge
            // Detect for a reliable mortar/seam line between stones,
            // since it won't miss a boundary just because two neighbors
            // happen to look alike.
            PortSpec {
                name: "cell_id",
                display_name: "Cell ID",
                port_type: PortType::Gray,
            },
        ],
        params: &[
            ParamSpec {
                name: "distance",
                display_name: "Distance",
                tooltip: "How far, in UV space, the value spreads from Mask before fading to \
                          zero.",
                kind: ParamSpecKind::Float {
                    min: 0.001,
                    max: 1.0,
                    default: 0.05,
                },
            },
            ParamSpec {
                name: "scale",
                display_name: "Scale",
                tooltip: "A quick multiplier on Distance's own reach, same idea as Substance \
                          Designer's own Scale on its Distance node — above 1 spreads far \
                          enough to flood through the gaps between separate shapes into soft \
                          blobs, below 1 keeps those gaps crisp and visible.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 4.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "threshold",
                display_name: "Threshold",
                tooltip: "Which Mask value counts as a seed to spread from.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "combine_mode",
                display_name: "Combine Mode",
                tooltip: "Combine fades the spread value smoothly to zero at Distance. Only \
                          Source shows it at full, flat strength everywhere within Distance \
                          instead, with a hard cutoff at the edge.",
                kind: ParamSpecKind::Enum {
                    options: &crate::distance_spread::DistanceCombineMode::NAMES,
                    default: "Combine",
                },
            },
            ParamSpec {
                name: "luminance_jitter",
                display_name: "Luminance Jitter",
                tooltip: "Random brightness variation, per separate island in Mask \u{2014} each \
                          disconnected blob gets its own consistent brightness, not a flat one \
                          everywhere.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "luminance_seed",
                display_name: "Luminance Seed",
                tooltip: "Changes the per-blob randomness.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "bevel",
        display_name: "Bevel",
        category: "Filters/Spatial",
        tooltip: "Rounds a mask or height input's edges into a beveled height map — chain \
                  into Height to Normal for the usual beveled-edge look.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "distance",
                display_name: "Distance",
                tooltip: "How far the bevel reaches from an edge, in UV space.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.2,
                    default: 0.05,
                },
            },
            ParamSpec {
                name: "convexity",
                display_name: "Convexity",
                tooltip: "0 is a plain linear chamfer; positive rounds the edge convex (a \
                          dome), negative makes it concave (a scoop).",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 1.0,
                    default: 0.3,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "warp",
        display_name: "Warp",
        category: "Filters/Spatial",
        tooltip: "Distort an input by the gradient of an intensity map.",
        inputs: &[
            PortSpec {
                name: "input",
                display_name: "Input",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "intensity",
                display_name: "Intensity",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "amount",
                display_name: "Amount",
                tooltip: "Warp strength.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.2,
                    default: 0.05,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "transform2d",
        display_name: "Transform 2D",
        category: "Filters/Spatial",
        tooltip: "Offset, rotate, and scale, with wrap-around tiling.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "offset_x",
                display_name: "Offset X",
                tooltip: "Horizontal offset.",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "offset_y",
                display_name: "Offset Y",
                tooltip: "Vertical offset.",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "rotation",
                display_name: "Rotation",
                tooltip: "Fraction of a full turn. Only multiples of 0.25 stay exactly tileable.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "scale_x",
                display_name: "Scale X",
                tooltip: "Horizontal scale.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 4.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "scale_y",
                display_name: "Scale Y",
                tooltip: "Vertical scale.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 4.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "flip_x",
                display_name: "Flip Horizontal",
                tooltip: "Mirrors the input left-to-right.",
                kind: ParamSpecKind::Bool { default: false },
            },
            ParamSpec {
                name: "flip_y",
                display_name: "Flip Vertical",
                tooltip: "Mirrors the input top-to-bottom.",
                kind: ParamSpecKind::Bool { default: false },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "height_to_normal",
        display_name: "Height to Normal",
        category: "Converters",
        tooltip: "Converts a height map to a tangent-space normal map.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Gray,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "intensity",
                display_name: "Intensity",
                tooltip: "How strongly height differences tilt the normal.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 10.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "format",
                display_name: "Format",
                tooltip: "OpenGL (Y+) for Blender, DirectX (Y-) for Unreal.",
                kind: ParamSpecKind::Enum {
                    options: &crate::height_to_normal::NormalFormat::NAMES,
                    default: "OpenGL",
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "clamp",
        display_name: "Clamp",
        category: "Filters/Adjustments",
        tooltip: "Restricts values to a [low, high] range.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "low",
                display_name: "Low",
                tooltip: "Minimum allowed output value.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "high",
                display_name: "High",
                tooltip: "Maximum allowed output value.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "threshold",
        display_name: "Threshold",
        category: "Filters/Blending & Masking",
        tooltip: "Hard black/white cutoff at a given value.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[ParamSpec {
            name: "threshold",
            display_name: "Threshold",
            tooltip: "Values at or above this become white, everything else black.",
            kind: ParamSpecKind::Float {
                min: 0.0,
                max: 1.0,
                default: 0.5,
            },
        }],
    },
    NodeDef {
        kind: "grayscale_conversion",
        display_name: "Grayscale Conversion",
        category: "Filters/Adjustments",
        tooltip: "Converts Color to Gray by luminance.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[],
    },
    NodeDef {
        kind: "checker",
        display_name: "Checker",
        category: "Generators/Patterns",
        tooltip: "An alternating black/white grid.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[ParamSpec {
            name: "scale",
            display_name: "Scale",
            tooltip: "Cells spanning the texture. Rounded up to even for seamless tiling.",
            kind: ParamSpecKind::Int {
                min: 2,
                max: 32,
                default: 8,
            },
        }],
    },
    NodeDef {
        kind: "dither_pattern",
        display_name: "Dither Pattern",
        category: "Generators/Patterns",
        tooltip: "The raw ordered-dither threshold mask (Bayer, blue noise, or a halftone dot \
                  grid), for use in Blend, Threshold, and similar nodes.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "pattern",
                display_name: "Pattern",
                tooltip: "Which ordered-dither matrix to use.",
                kind: ParamSpecKind::Enum {
                    options: &["Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Blue Noise", "Dot"],
                    default: "Bayer 4x4",
                },
            },
            ParamSpec {
                name: "cells_x",
                display_name: "Cells X",
                tooltip: "How many dither cells span the texture horizontally. Rounded up to \
                          the nearest multiple of the pattern's own tile size (2/4/8/64) for \
                          seamless tiling.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 512,
                    default: 64,
                },
            },
            ParamSpec {
                name: "cells_y",
                display_name: "Cells Y",
                tooltip: "How many dither cells span the texture vertically. Rounded the same \
                          way as Cells X.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 512,
                    default: 64,
                },
            },
            ParamSpec {
                name: "rotation",
                display_name: "Rotation",
                tooltip: "Tilts the pattern to a classic halftone screen angle (print screens \
                          are rotated per channel to hide the grid and avoid moire). Limited \
                          to angles that still tile exactly, the same way real halftone \
                          screens are restricted to a handful of rational angles — the cell \
                          size shifts slightly with the angle as a result. No effect on Blue \
                          Noise.",
                kind: ParamSpecKind::Enum {
                    options: &["0°", "18.4°", "26.6°", "45°"],
                    default: "0°",
                },
            },
            ParamSpec {
                name: "dot_size",
                display_name: "Dot Size",
                tooltip: "How much of each cell the circle fills. Only used by the Dot \
                          pattern.",
                kind: ParamSpecKind::Float {
                    min: 0.05,
                    max: 1.0,
                    default: 0.6,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Offsets the Blue Noise mask. Has no effect on other patterns.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "hsl_adjust",
        display_name: "HSL Adjust",
        category: "Filters/Adjustments",
        tooltip: "Shift hue, scale saturation, and shift lightness.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "hue_shift",
                display_name: "Hue Shift",
                tooltip: "Fraction of a full turn to rotate hue by.",
                kind: ParamSpecKind::Float {
                    min: -0.5,
                    max: 0.5,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "saturation",
                display_name: "Saturation",
                tooltip: "Multiplier on saturation (0 = grayscale).",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 2.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "lightness",
                display_name: "Lightness",
                tooltip: "Additive shift on lightness.",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "color_match",
        display_name: "Color Match",
        category: "Filters/Adjustments",
        tooltip: "Maps this input's own brightness through a color gradient \
                  built from a reference image, so its full range of hues \
                  can carry over \u{2014} not just an overall tint.",
        inputs: &[
            PortSpec {
                name: "input",
                display_name: "Input",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "reference",
                display_name: "Reference",
                port_type: PortType::Color,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "amount",
                display_name: "Amount",
                tooltip: "How strongly the match is applied \u{2014} 0 keeps Input \
                          unchanged, 1 fully adopts Reference's tone.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "saturation",
                display_name: "Saturation",
                tooltip: "Boosts or reduces the matched result's color \
                          intensity afterward \u{2014} 1 = unchanged. \
                          Reference's own color already counts its most \
                          vivid pixels more than its plainer ones, but a \
                          push here can still help when Input itself \
                          starts out very desaturated.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 3.0,
                    default: 1.0,
                },
            },
            // Edited via the Parameters panel's own crop-rectangle picker,
            // dragged directly on Reference's thumbnail — not these raw
            // Vec2 sliders (see `mc_app::params_panel`'s own node-specific
            // exception for this node, the same kind Knit's `chart` param
            // already is). Still declared as normal params so a fresh
            // Color Match node gets sane defaults and they save/load/undo
            // through the regular param machinery.
            ParamSpec {
                name: "crop_min",
                display_name: "Crop Min",
                tooltip: "Top-left corner (0..1) of the region of Reference \
                          the gradient is built from.",
                kind: ParamSpecKind::Vec2 {
                    min: 0.0,
                    max: 1.0,
                    default: [0.0, 0.0],
                },
            },
            ParamSpec {
                name: "crop_max",
                display_name: "Crop Max",
                tooltip: "Bottom-right corner (0..1) of the region of \
                          Reference the gradient is built from.",
                kind: ParamSpecKind::Vec2 {
                    min: 0.0,
                    max: 1.0,
                    default: [1.0, 1.0],
                },
            },
        ],
    },
    NodeDef {
        kind: "material_preset",
        display_name: "Material Presets",
        category: "Filters/Adjustments",
        tooltip: "Pick a real, measured material and drive Base Color/\
                  Roughness/Metallic toward it \u{2014} like a base-material \
                  preset, backed by real spectrometer data.",
        inputs: &[
            PortSpec {
                name: "base_color",
                display_name: "Base Color",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "roughness",
                display_name: "Roughness",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "metallic",
                display_name: "Metallic",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[
            PortSpec {
                name: "base_color",
                display_name: "Base Color",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "roughness",
                display_name: "Roughness",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "metallic",
                display_name: "Metallic",
                port_type: PortType::Gray,
            },
        ],
        params: &[
            ParamSpec {
                name: "preset",
                display_name: "Preset",
                tooltip: "Which real-world material's own measured values \
                          to drive toward.",
                kind: ParamSpecKind::Enum {
                    options: &crate::material_preset::PRESET_NAMES,
                    default: "Aluminum",
                },
            },
            ParamSpec {
                name: "amount",
                display_name: "Amount",
                tooltip: "How strongly the preset is applied \u{2014} 0 keeps \
                          each connected input unchanged, 1 fully adopts \
                          the preset's own values. An unconnected input \
                          always outputs the preset's flat value directly, \
                          regardless of Amount.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                },
            },
        ],
    },
    NodeDef {
        kind: "curve",
        display_name: "Curve",
        category: "Filters/Adjustments",
        tooltip: "A 5-point curve — drag each point's output value.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "v0",
                display_name: "At 0.00",
                tooltip: "Output value when the input is 0.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "v1",
                display_name: "At 0.25",
                tooltip: "Output value when the input is 0.25.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.25,
                },
            },
            ParamSpec {
                name: "v2",
                display_name: "At 0.50",
                tooltip: "Output value when the input is 0.5.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "v3",
                display_name: "At 0.75",
                tooltip: "Output value when the input is 0.75.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.75,
                },
            },
            ParamSpec {
                name: "v4",
                display_name: "At 1.00",
                tooltip: "Output value when the input is 1.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "gradient_map",
        display_name: "Gradient Map",
        category: "Filters/Adjustments",
        tooltip: "Maps a Gray input through a color gradient.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Gray,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[ParamSpec {
            name: "gradient",
            display_name: "Gradient",
            tooltip: "Color stops the input value is mapped through, in position order.",
            kind: ParamSpecKind::Gradient {
                default: &[
                    GradientStop {
                        position: 0.0,
                        color: [0.0, 0.0, 0.0, 1.0],
                    },
                    GradientStop {
                        position: 1.0,
                        color: [1.0, 1.0, 1.0, 1.0],
                    },
                ],
            },
        }],
    },
    NodeDef {
        kind: "simplex_noise",
        display_name: "Simplex Noise",
        category: "Generators/Noise",
        tooltip:
            "Gradient noise on a triangular lattice — fewer directional artifacts than Perlin.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "scale",
                display_name: "Scale",
                tooltip: "How many noise cells tile across the texture.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "octaves",
                display_name: "Octaves",
                tooltip: "How many layers of noise are summed.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 8,
                    default: 4,
                },
            },
            ParamSpec {
                name: "persistence",
                display_name: "Persistence",
                tooltip: "How much each octave's contribution shrinks.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern without changing its character.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "white_noise",
        display_name: "White Noise",
        category: "Generators/Noise",
        tooltip: "Fully uncorrelated random cells — no smooth interpolation.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "scale_x",
                display_name: "Scale X",
                tooltip: "Cells spanning the texture along X.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 16,
                },
            },
            ParamSpec {
                name: "scale_y",
                display_name: "Scale Y",
                tooltip: "Cells spanning the texture along Y. A different value from Scale X \
                          stretches the cells into rectangles.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 16,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "clouds",
        display_name: "Clouds",
        category: "Generators/Noise",
        tooltip: "An fBm preset tuned for a cloud-like look.",
        inputs: &[],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "scale",
                display_name: "Scale",
                tooltip: "How many noise cells tile across the texture.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 32,
                    default: 4,
                },
            },
            ParamSpec {
                name: "softness",
                display_name: "Softness",
                tooltip: "How soft/fluffy the clouds look.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the pattern without changing its character.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "directional_blur",
        display_name: "Directional Blur",
        category: "Filters/Spatial",
        tooltip: "A motion-blur-like smear along one fixed direction.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "radius",
                display_name: "Radius",
                tooltip: "Blur radius, in UV space.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.1,
                    default: 0.02,
                },
            },
            ParamSpec {
                name: "angle",
                display_name: "Angle",
                tooltip: "Blur direction, as a fraction of a full turn.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "directional_warp",
        display_name: "Push Warp",
        category: "Filters/Spatial",
        tooltip: "Shift pixels along one fixed direction by an intensity map's value.",
        inputs: &[
            PortSpec {
                name: "input",
                display_name: "Input",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "intensity",
                display_name: "Intensity",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "angle",
                display_name: "Angle",
                tooltip: "Warp direction, as a fraction of a full turn.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "amount",
                display_name: "Amount",
                tooltip: "Warp strength.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.2,
                    default: 0.1,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "edge_detect",
        display_name: "Edge Detect",
        category: "Filters/Spatial",
        tooltip: "Highlights where the input changes sharply.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        // Not type-preserving like its neighbors above — its own .wgsl file
        // reduces the input to luminance and broadcasts the Sobel gradient
        // magnitude of *that* back out to every channel, so the output is
        // always Gray-equivalent (R=G=B) regardless of whether the input
        // was genuine Color or a Gray broadcast. A real converter, same
        // category as Grayscale Conversion/Curvature from Normal, not a
        // passthrough that happens to need its real type resolved.
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "intensity",
                display_name: "Intensity",
                tooltip: "Multiplier on the detected edge strength.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 16.0,
                    default: 4.0,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "slope_blur",
        display_name: "Flow Blur",
        category: "Filters/Spatial",
        tooltip: "Blur along a control map's gradient, strength scaled by its steepness.",
        inputs: &[
            PortSpec {
                name: "input",
                display_name: "Input",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "control",
                display_name: "Control",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "radius",
                display_name: "Radius",
                tooltip: "Maximum blur radius, reached at the control map's steepest point.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.1,
                    default: 0.03,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "height_to_ao",
        display_name: "Height to AO",
        category: "Converters",
        tooltip: "A simple screen-space-style ambient occlusion map from height.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Gray,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "radius",
                display_name: "Radius",
                tooltip: "How far out the occlusion sample reaches, in UV space.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.1,
                    default: 0.02,
                },
            },
            ParamSpec {
                name: "intensity",
                display_name: "Intensity",
                tooltip: "Multiplier on the occlusion strength. Kept fairly low by default — \
                          `ao = clamp(1 - occlusion * intensity, 0, 1)` saturates to pure black \
                          surprisingly fast at high values, clipping away the soft Gaussian \
                          falloff the occlusion sampling itself produces and leaving a harsh \
                          silhouette instead of a gentle shadow.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 8.0,
                    default: 0.6,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "curvature_from_normal",
        display_name: "Curvature from Normal",
        category: "Converters",
        tooltip: "Convex edges read bright, concave crevices read dark.",
        inputs: &[PortSpec {
            name: "input",
            display_name: "Input",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Gray,
        }],
        params: &[
            ParamSpec {
                name: "intensity",
                display_name: "Intensity",
                tooltip: "Multiplier on the curvature strength.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 16.0,
                    default: 4.0,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "distance_field",
        display_name: "Distance Field",
        // `docs/hardsurface_nodes/HARDSURFACE_NODES.md` §3.1, the first of
        // seven planned hard-surface nodes built on a new `Sdf` port type
        // (negative inside / positive outside / 0 on the edge, tile units).
        // Grouped with Height to Normal/AO and Curvature from Normal for
        // now — all four turn one representation into another — rather
        // than a new category for just this one node; later hard-surface
        // nodes may warrant their own category once there are enough to
        // group meaningfully.
        category: "Converters",
        tooltip: "Turns a greyscale mask into an exact signed distance field of its 0.5 edge.",
        inputs: &[PortSpec {
            name: "mask",
            display_name: "Mask",
            port_type: PortType::Gray,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Sdf",
            port_type: PortType::Sdf,
        }],
        params: &[
            ParamSpec {
                name: "wrap",
                display_name: "Wrap",
                tooltip: "Distances stay correct up to 0.5 tile across the seam. Off treats the \
                          mask as a single, non-repeating island.",
                kind: ParamSpecKind::Bool { default: true },
            },
            ParamSpec {
                name: "threshold",
                display_name: "Threshold",
                tooltip: "Which mask value counts as the edge.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "max_distance",
                display_name: "Max Distance",
                tooltip: "Clamp the field to +-this many tile units. 0 disables the clamp.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.5,
                    default: 0.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "shape_boolean",
        display_name: "Shape Boolean",
        // HARDSURFACE_NODES.md §3.3, grouped with Distance Field for the
        // same "no dedicated category for just a few nodes yet" reasoning.
        category: "Converters",
        tooltip: "Combines two SDFs with a real boolean \u{2014} union, subtract, or intersect \
                  \u{2014} in a hard, smooth, round, chamfer, or groove blend.",
        inputs: &[
            PortSpec {
                name: "a",
                display_name: "A",
                port_type: PortType::Sdf,
            },
            PortSpec {
                name: "b",
                display_name: "B",
                port_type: PortType::Sdf,
            },
        ],
        outputs: &[
            PortSpec {
                name: "sdf",
                display_name: "Sdf",
                port_type: PortType::Sdf,
            },
            PortSpec {
                name: "a_only",
                display_name: "A Only",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "b_only",
                display_name: "B Only",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "overlap",
                display_name: "Overlap",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "blend",
                display_name: "Blend",
                port_type: PortType::Gray,
            },
        ],
        params: &[
            ParamSpec {
                name: "op",
                display_name: "Operation",
                tooltip: "Union, Subtract (A minus B), or Intersect. Ignored entirely in Groove \
                          mode \u{2014} there's only one sensible direction for a groove.",
                kind: ParamSpecKind::Enum {
                    options: &crate::shape_boolean::BoolOp::NAMES,
                    default: "Union",
                },
            },
            ParamSpec {
                name: "mode",
                display_name: "Mode",
                tooltip: "Hard is a plain min/max boolean. Smooth, Round, and Chamfer blend the \
                          seam by Radius. Groove cuts a groove of width Radius along B's \
                          outline into A, Groove Depth deep.",
                kind: ParamSpecKind::Enum {
                    options: &crate::shape_boolean::BoolMode::NAMES,
                    default: "Hard",
                },
            },
            ParamSpec {
                name: "radius",
                display_name: "Radius",
                tooltip: "Blend size (Smooth/Round/Chamfer) or groove width (Groove), tile \
                          units. No effect in Hard mode.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.2,
                    default: 0.02,
                },
            },
            ParamSpec {
                name: "groove_depth",
                display_name: "Groove Depth",
                tooltip: "Groove mode only: how far the groove cuts into A.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.2,
                    default: 0.02,
                },
            },
            ParamSpec {
                name: "exact",
                display_name: "Exact",
                tooltip: "Rebuilds an exact field from the result's own zero contour. A boolean \
                          built from min/max keeps the inputs' hidden interior edges, which a \
                          later Bevel would otherwise show as a false groove where two shapes \
                          meet. Only affects the Sdf output \u{2014} the four masks are always \
                          exact, from A/B's own sign.",
                kind: ParamSpecKind::Bool { default: true },
            },
        ],
    },
    NodeDef {
        kind: "redistance",
        display_name: "Redistance",
        category: "Converters",
        tooltip: "Rebuilds an exact signed distance field from any SDF's own zero contour \u{2014} \
                  fixes the false groove a boolean or warp can leave where hidden interior \
                  edges used to be.",
        inputs: &[PortSpec {
            name: "sdf",
            display_name: "Sdf",
            port_type: PortType::Sdf,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Sdf",
            port_type: PortType::Sdf,
        }],
        params: &[ParamSpec {
            name: "wrap",
            display_name: "Wrap",
            tooltip: "Distances stay correct up to 0.5 tile across the seam. Off treats the \
                      field as a single, non-repeating island.",
            kind: ParamSpecKind::Bool { default: true },
        }],
    },
    NodeDef {
        kind: "bevel_profile",
        display_name: "Bevel Profile",
        category: "Converters",
        tooltip: "Turns an Sdf into a real height map through a chosen edge profile \u{2014} the \
                  true distance to an edge, not a blurred approximation of one.",
        inputs: &[PortSpec {
            name: "sdf",
            display_name: "Sdf",
            port_type: PortType::Sdf,
        }],
        outputs: &[
            PortSpec {
                name: "height",
                display_name: "Height",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "edge",
                display_name: "Edge",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "top",
                display_name: "Top",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "u",
                display_name: "U",
                port_type: PortType::Gray,
            },
        ],
        params: &[
            ParamSpec {
                name: "preset",
                display_name: "Preset",
                tooltip: "The edge profile's shape, from a sharp Chamfer to a raised Lip. Custom \
                          lets you set 4 control points by hand.",
                kind: ParamSpecKind::Enum {
                    options: &crate::bevel_profile::BevelPreset::NAMES,
                    default: "Round",
                },
            },
            ParamSpec {
                name: "width",
                display_name: "Width",
                tooltip: "How far the bevel reaches from the edge, tile units.",
                kind: ParamSpecKind::Float {
                    min: 0.001,
                    max: 0.3,
                    default: 0.05,
                },
            },
            ParamSpec {
                name: "wall",
                display_name: "Wall",
                tooltip: "A vertical drop before the bevel starts, for a crisp stepped edge.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "depth",
                display_name: "Depth",
                tooltip: "Overall height scale. Negative recesses the bevel instead of raising \
                          it.",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 2.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "base",
                display_name: "Base",
                tooltip: "Height everywhere outside the shape.",
                kind: ParamSpecKind::Float {
                    min: -1.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "inward",
                display_name: "Inward",
                tooltip: "On bevels into the shape (the normal case). Off bevels outward \u{2014} \
                          a moat or raised surround instead.",
                kind: ParamSpecKind::Bool { default: true },
            },
            ParamSpec {
                name: "custom_y0",
                display_name: "Custom Y 0.0",
                tooltip: "Only used when Preset is set to Custom above \u{2014} the curve's \
                          value at the edge (u = 0).",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "custom_y1",
                display_name: "Custom Y 0.2",
                tooltip: "Only used when Preset is set to Custom above \u{2014} the curve's \
                          value at u = 0.2.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.6,
                },
            },
            ParamSpec {
                name: "custom_y2",
                display_name: "Custom Y 0.6",
                tooltip: "Only used when Preset is set to Custom above \u{2014} the curve's \
                          value at u = 0.6.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.7,
                },
            },
            ParamSpec {
                name: "custom_y3",
                display_name: "Custom Y 1.0",
                tooltip: "Only used when Preset is set to Custom above \u{2014} the curve's \
                          value at the flat top (u = 1).",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
        ],
    },
    NodeDef {
        kind: "edge_wear",
        display_name: "Edge Wear",
        category: "Converters",
        tooltip: "Paints chipping on convex edges and a cavity mask for dirt in seams and \
                  recesses \u{2014} exact from an Sdf, or estimated from any height map if no \
                  Sdf is connected.",
        inputs: &[
            PortSpec {
                name: "sdf",
                display_name: "Sdf",
                port_type: PortType::Sdf,
            },
            PortSpec {
                name: "height",
                display_name: "Height",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[
            PortSpec {
                name: "wear",
                display_name: "Wear",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "cavity",
                display_name: "Cavity",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "height",
                display_name: "Height",
                port_type: PortType::Gray,
            },
        ],
        params: &[
            ParamSpec {
                name: "band",
                display_name: "Band",
                tooltip: "Sdf input only \u{2014} how far the wear reaches from the edge, tile \
                          units.",
                kind: ParamSpecKind::Float {
                    min: 0.0005,
                    max: 0.1,
                    default: 0.012,
                },
            },
            ParamSpec {
                name: "amount",
                display_name: "Amount",
                tooltip: "Overall wear strength.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 2.0,
                    default: 0.5,
                },
            },
            ParamSpec {
                name: "falloff",
                display_name: "Falloff",
                tooltip: "Sdf input only \u{2014} how sharply wear fades away from the edge.",
                kind: ParamSpecKind::Float {
                    min: 0.1,
                    max: 5.0,
                    default: 1.5,
                },
            },
            ParamSpec {
                name: "chip_cells",
                display_name: "Chip Cells",
                tooltip: "Lattice size of the chip pattern's own noise.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 24,
                },
            },
            ParamSpec {
                name: "softness",
                display_name: "Softness",
                tooltip: "How soft the wear's own edge is.",
                kind: ParamSpecKind::Float {
                    min: 0.001,
                    max: 0.5,
                    default: 0.08,
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes the chip pattern.",
                kind: ParamSpecKind::Seed { default: 7 },
            },
            ParamSpec {
                name: "cavity_width",
                display_name: "Cavity Width",
                tooltip: "Sdf input only \u{2014} how far the cavity mask reaches into seams \
                          and recesses, tile units.",
                kind: ParamSpecKind::Float {
                    min: 0.0005,
                    max: 0.1,
                    default: 0.006,
                },
            },
            ParamSpec {
                name: "curvature_sigma",
                display_name: "Curvature Sigma",
                tooltip: "Height-map fallback only (no Sdf connected) \u{2014} the curvature \
                          estimate's own blur radius, pixels.",
                kind: ParamSpecKind::Float {
                    min: 0.5,
                    max: 5.0,
                    default: 1.5,
                },
            },
            ParamSpec {
                name: "chip_depth",
                display_name: "Chip Depth",
                tooltip: "How deep the Height output's own chipping cuts.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.2,
                    default: 0.04,
                },
            },
        ],
    },
    NodeDef {
        kind: "channel_pack",
        display_name: "Channel Packer",
        category: "Converters",
        tooltip: "Pack three grayscale inputs into one texture's R/G/B channels — e.g. \
                  Ambient Occlusion, Roughness, and Metallic into an ORM map for Unreal Engine. \
                  The R/G/B suggestions on each input are just that — wire up whatever three \
                  grayscale sources you actually want packed.",
        inputs: &[
            PortSpec {
                name: "r",
                display_name: "R (AO)",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "g",
                display_name: "G (Roughness)",
                port_type: PortType::Gray,
            },
            PortSpec {
                name: "b",
                display_name: "B (Metallic)",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[],
    },
    NodeDef {
        kind: "channel_split",
        display_name: "Channel Split",
        category: "Filters/Glitch",
        tooltip: "Samples each RGB channel at its own offset, for chromatic-aberration/glitch \
                  looks.",
        inputs: &[
            PortSpec {
                name: "image",
                display_name: "Image",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "offset_r",
                display_name: "Offset R",
                tooltip: "Red channel's sample offset, in UV units.",
                kind: ParamSpecKind::Vec2 {
                    min: -0.5,
                    max: 0.5,
                    default: [0.01, 0.0],
                },
            },
            ParamSpec {
                name: "offset_g",
                display_name: "Offset G",
                tooltip: "Green channel's sample offset, in UV units.",
                kind: ParamSpecKind::Vec2 {
                    min: -0.5,
                    max: 0.5,
                    default: [0.0, 0.0],
                },
            },
            ParamSpec {
                name: "offset_b",
                display_name: "Offset B",
                tooltip: "Blue channel's sample offset, in UV units.",
                kind: ParamSpecKind::Vec2 {
                    min: -0.5,
                    max: 0.5,
                    default: [-0.01, 0.0],
                },
            },
            ParamSpec {
                name: "strength",
                display_name: "Strength",
                tooltip: "Blends between the original image (0) and the fully split channels \
                          (1).",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "block_shift",
        display_name: "Block Shift",
        category: "Filters/Glitch",
        tooltip: "Shifts random blocks of the image sideways — a seeded hash per block \
                  decides whether and how far.",
        inputs: &[
            PortSpec {
                name: "image",
                display_name: "Image",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "columns",
                display_name: "Columns",
                tooltip: "How many blocks span the texture horizontally.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 8,
                },
            },
            ParamSpec {
                name: "rows",
                display_name: "Rows",
                tooltip: "How many blocks span the texture vertically.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 8,
                },
            },
            ParamSpec {
                name: "amount",
                display_name: "Amount",
                tooltip: "Fraction of blocks that shift at all.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.3,
                },
            },
            ParamSpec {
                name: "max_offset",
                display_name: "Max Offset",
                tooltip: "Largest possible shift for an affected block, in UV units.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.5,
                    default: 0.1,
                },
            },
            ParamSpec {
                name: "direction",
                display_name: "Direction",
                tooltip: "Which axis a shifting block moves along.",
                kind: ParamSpecKind::Enum {
                    options: &["Horizontal", "Vertical", "Both"],
                    default: "Both",
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes which blocks shift and by how much.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "scanline_jitter",
        display_name: "Scanline Jitter",
        category: "Filters/Glitch",
        tooltip: "Shifts random horizontal or vertical bands sideways — the classic 'bad \
                  video signal' glitch.",
        inputs: &[
            PortSpec {
                name: "image",
                display_name: "Image",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "lines",
                display_name: "Lines",
                tooltip: "How many bands span the texture.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 256,
                    default: 32,
                },
            },
            ParamSpec {
                name: "amount",
                display_name: "Amount",
                tooltip: "Fraction of bands that jitter at all.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.3,
                },
            },
            ParamSpec {
                name: "max_offset",
                display_name: "Max Offset",
                tooltip: "Largest possible jitter for an affected band, in UV units.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 0.5,
                    default: 0.05,
                },
            },
            ParamSpec {
                name: "orientation",
                display_name: "Orientation",
                tooltip: "Whether bands run horizontally or vertically.",
                kind: ParamSpecKind::Enum {
                    options: &["Horizontal", "Vertical"],
                    default: "Horizontal",
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes which bands jitter and by how much.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
            tiling_mode_param(),
        ],
    },
    NodeDef {
        kind: "ordered_dither",
        display_name: "Ordered Dither",
        category: "Filters/Stylize",
        tooltip: "Reduces an image to a few tones with an ordered (Bayer or blue noise) \
                  dither pattern.",
        inputs: &[PortSpec {
            name: "image",
            display_name: "Image",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "pattern",
                display_name: "Pattern",
                tooltip: "Which ordered-dither matrix to use.",
                kind: ParamSpecKind::Enum {
                    options: &["Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Blue Noise"],
                    default: "Bayer 4x4",
                },
            },
            ParamSpec {
                name: "cells",
                display_name: "Cells",
                tooltip: "How many dither cells span the texture. Rounded up to the nearest \
                          multiple of the pattern's own tile size (2/4/8/64) for seamless \
                          tiling.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 512,
                    default: 64,
                },
            },
            ParamSpec {
                name: "levels",
                display_name: "Levels",
                tooltip: "Tone steps per channel. 2 = pure 1-bit (black/white only).",
                kind: ParamSpecKind::Int {
                    min: 2,
                    max: 16,
                    default: 4,
                },
            },
            ParamSpec {
                name: "spread",
                display_name: "Spread",
                tooltip: "Dither strength: 0 is a plain posterize with no dithering, 1 is the \
                          full dither pattern.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "bias",
                display_name: "Bias",
                tooltip: "Shifts the threshold lighter or darker.",
                kind: ParamSpecKind::Float {
                    min: -0.5,
                    max: 0.5,
                    default: 0.0,
                },
            },
            ParamSpec {
                name: "mode",
                display_name: "Mode",
                tooltip: "Per Channel dithers R/G/B independently. Luminance quantizes \
                          brightness only and rescales all 3 channels together, preserving \
                          hue. No visible difference on a Gray-sourced image.",
                kind: ParamSpecKind::Enum {
                    options: &["Per Channel", "Luminance"],
                    default: "Luminance",
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Offsets the Blue Noise mask. Has no effect on Bayer patterns.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "palette_dither",
        display_name: "Palette Dither",
        category: "Filters/Stylize",
        tooltip: "Dithers a Color image down to a fixed palette of 2-8 colors.",
        inputs: &[PortSpec {
            name: "image",
            display_name: "Image",
            port_type: PortType::Color,
        }],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "palette",
                display_name: "Palette",
                tooltip: "The colors to dither down to, in order (position only decides \
                          order here, not blending).",
                kind: ParamSpecKind::Gradient {
                    default: &[
                        GradientStop {
                            position: 0.0,
                            color: [0.0, 0.0, 0.0, 1.0],
                        },
                        GradientStop {
                            position: 0.5,
                            color: [0.5, 0.5, 0.5, 1.0],
                        },
                        GradientStop {
                            position: 1.0,
                            color: [1.0, 1.0, 1.0, 1.0],
                        },
                    ],
                },
            },
            ParamSpec {
                name: "pattern",
                display_name: "Pattern",
                tooltip: "Which ordered-dither matrix to use.",
                kind: ParamSpecKind::Enum {
                    options: &["Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Blue Noise"],
                    default: "Bayer 4x4",
                },
            },
            ParamSpec {
                name: "cells",
                display_name: "Cells",
                tooltip: "How many dither cells span the texture. Rounded up to the nearest \
                          multiple of the pattern's own tile size (2/4/8/64) for seamless \
                          tiling.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 512,
                    default: 64,
                },
            },
            ParamSpec {
                name: "spread",
                display_name: "Spread",
                tooltip: "Dither strength: 0 picks the nearest palette color directly with no \
                          dithering, 1 uses the full dither pattern.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 1.0,
                },
            },
            ParamSpec {
                name: "distance",
                display_name: "Distance",
                tooltip: "How \"nearest\" is measured when picking a palette color. RGB is \
                          plain Euclidean distance; Perceptual weights channels by luminance.",
                kind: ParamSpecKind::Enum {
                    options: &["RGB", "Perceptual"],
                    default: "Perceptual",
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Offsets the Blue Noise mask. Has no effect on Bayer patterns.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    NodeDef {
        kind: "data_corrupt",
        display_name: "Data Corrupt",
        category: "Filters/Glitch",
        tooltip: "Posterizes, channel-swaps, or inverts random blocks — a glitchy 'corrupted \
                  file' look.",
        inputs: &[
            PortSpec {
                name: "image",
                display_name: "Image",
                port_type: PortType::Color,
            },
            PortSpec {
                name: "mask",
                display_name: "Mask",
                port_type: PortType::Gray,
            },
        ],
        outputs: &[PortSpec {
            name: "output",
            display_name: "Output",
            port_type: PortType::Color,
        }],
        params: &[
            ParamSpec {
                name: "columns",
                display_name: "Columns",
                tooltip: "How many blocks span the texture horizontally.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 8,
                },
            },
            ParamSpec {
                name: "rows",
                display_name: "Rows",
                tooltip: "How many blocks span the texture vertically.",
                kind: ParamSpecKind::Int {
                    min: 1,
                    max: 64,
                    default: 8,
                },
            },
            ParamSpec {
                name: "amount",
                display_name: "Amount",
                tooltip: "Fraction of blocks that get corrupted.",
                kind: ParamSpecKind::Float {
                    min: 0.0,
                    max: 1.0,
                    default: 0.2,
                },
            },
            ParamSpec {
                name: "effect",
                display_name: "Effect",
                tooltip: "Posterize crushes a block to 3 tones; Channel Swap permutes its RGB \
                          channels; Invert flips it; Mix picks one of the three per block.",
                kind: ParamSpecKind::Enum {
                    options: &["Posterize", "Channel Swap", "Invert", "Mix"],
                    default: "Mix",
                },
            },
            ParamSpec {
                name: "seed",
                display_name: "Seed",
                tooltip: "Changes which blocks are affected and how.",
                kind: ParamSpecKind::Seed { default: 0 },
            },
        ],
    },
    output_node_def("output_base_color", "Output: Base Color"),
    output_node_def("output_normal", "Output: Normal"),
    output_node_def("output_roughness", "Output: Roughness"),
    output_node_def("output_metallic", "Output: Metallic"),
    output_node_def("output_height", "Output: Height"),
    output_node_def("output_ambient_occlusion", "Output: Ambient Occlusion"),
    output_node_def("output_orm", "Output: ORM"),
    output_node_def("output_emissive", "Output: Emissive"),
    output_node_def("output_alpha", "Output: Alpha"),
];

#[cfg(test)]
mod tests {
    use super::effective_output_type;
    use mc_core::{Graph, Node, PortType};

    /// Regression test for the Levels/Invert/Clamp/Threshold/Curve fix: a
    /// type-preserving node fed a Gray source reports Gray, not its
    /// statically-declared Color.
    #[test]
    fn type_preserving_node_follows_a_gray_source() {
        let mut graph = Graph::new();
        let noise = graph.add_node(Node::new("perlin_noise")); // declared Gray
        let levels = graph.add_node(Node::new("levels"));
        graph.connect(noise, "output", levels, "input").unwrap();

        assert_eq!(
            effective_output_type(&graph, levels, "levels", PortType::Color),
            PortType::Gray
        );
    }

    /// An unconnected type-preserving node has nothing to follow, so it falls
    /// back to its declared type.
    #[test]
    fn type_preserving_node_with_no_input_falls_back_to_declared_type() {
        let mut graph = Graph::new();
        let levels = graph.add_node(Node::new("levels"));

        assert_eq!(
            effective_output_type(&graph, levels, "levels", PortType::Color),
            PortType::Color
        );
    }

    /// Blend fed two Gray-sourced inputs ("While testing I realized that the
    /// Blend node always considers inputs to be color and therefore only
    /// outputs color") should now report Gray.
    #[test]
    fn blend_with_two_gray_inputs_reports_gray_output() {
        let mut graph = Graph::new();
        let noise_a = graph.add_node(Node::new("perlin_noise"));
        let noise_b = graph.add_node(Node::new("perlin_noise"));
        let blend = graph.add_node(Node::new("blend"));
        graph.connect(noise_a, "output", blend, "base").unwrap();
        graph.connect(noise_b, "output", blend, "blend").unwrap();

        assert_eq!(
            effective_output_type(&graph, blend, "blend", PortType::Color),
            PortType::Gray
        );
    }

    /// Color wins: a Blend with one Gray and one Color input must report
    /// Color, since the Color input can carry real chroma.
    #[test]
    fn blend_with_one_color_input_reports_color_output() {
        let mut graph = Graph::new();
        let noise = graph.add_node(Node::new("perlin_noise")); // Gray
        let uniform_color = graph.add_node(Node::new("uniform_color")); // Color
        let blend = graph.add_node(Node::new("blend"));
        graph.connect(noise, "output", blend, "base").unwrap();
        graph
            .connect(uniform_color, "output", blend, "blend")
            .unwrap();

        assert_eq!(
            effective_output_type(&graph, blend, "blend", PortType::Color),
            PortType::Color
        );
    }

    /// A Blend with only one input wired defers entirely to that input's
    /// effective type, rather than falling back to the declared Color.
    #[test]
    fn blend_with_only_one_input_wired_follows_that_input() {
        let mut graph = Graph::new();
        let noise = graph.add_node(Node::new("perlin_noise")); // Gray
        let blend = graph.add_node(Node::new("blend"));
        graph.connect(noise, "output", blend, "base").unwrap();

        assert_eq!(
            effective_output_type(&graph, blend, "blend", PortType::Color),
            PortType::Gray
        );
    }

    /// A Blend with neither input wired falls back to its declared Color.
    #[test]
    fn blend_with_no_inputs_falls_back_to_declared_type() {
        let mut graph = Graph::new();
        let blend = graph.add_node(Node::new("blend"));

        assert_eq!(
            effective_output_type(&graph, blend, "blend", PortType::Color),
            PortType::Color
        );
    }

    /// The fix this test guards: resolving an upstream Blend's type must
    /// recurse into its *effective* type, not its merely-declared Color — a
    /// chain of Gray source -> Blend -> Levels must still report Gray all the
    /// way through, since Levels is type-preserving and Blend's two Gray
    /// inputs make it effectively Gray too.
    #[test]
    fn chained_gray_blend_through_a_type_preserving_node_stays_gray() {
        let mut graph = Graph::new();
        let noise_a = graph.add_node(Node::new("perlin_noise"));
        let noise_b = graph.add_node(Node::new("perlin_noise"));
        let blend = graph.add_node(Node::new("blend"));
        let levels = graph.add_node(Node::new("levels"));
        graph.connect(noise_a, "output", blend, "base").unwrap();
        graph.connect(noise_b, "output", blend, "blend").unwrap();
        graph.connect(blend, "output", levels, "input").unwrap();

        assert_eq!(
            effective_output_type(&graph, levels, "levels", PortType::Color),
            PortType::Gray
        );
    }

    /// Regression test for the wider audit this prompted ("a bunch of other
    /// nodes... pretend to be color value"): Transform 2D (and its
    /// newly-added neighbors) fed a Gray source must report Gray, the exact
    /// same bug class as the original Levels/Blend fixes, just not caught
    /// the first time around.
    #[test]
    fn newly_audited_type_preserving_nodes_follow_a_gray_source() {
        for kind in [
            "histogram_scan",
            "blur",
            "transform2d",
            "directional_blur",
            "bevel",
        ] {
            let mut graph = Graph::new();
            let noise = graph.add_node(Node::new("perlin_noise"));
            let node = graph.add_node(Node::new(kind));
            graph.connect(noise, "output", node, "input").unwrap();

            assert_eq!(
                effective_output_type(&graph, node, kind, PortType::Color),
                PortType::Gray,
                "{kind} should report Gray when fed a Gray source"
            );
        }
    }

    /// Warp, Push Warp, and Flow Blur each take a *second*, Gray-only
    /// "intensity"/"control" input that drives the distortion but isn't
    /// itself part of the output's content — the effective type must follow
    /// the main "input" port only, not get confused by that second port.
    #[test]
    fn dual_input_type_preserving_nodes_follow_their_main_input_only() {
        for (kind, second_port) in [
            ("warp", "intensity"),
            ("directional_warp", "intensity"),
            ("slope_blur", "control"),
        ] {
            let mut graph = Graph::new();
            let main_source = graph.add_node(Node::new("perlin_noise")); // Gray
            let control_source = graph.add_node(Node::new("perlin_noise")); // Gray too
            let node = graph.add_node(Node::new(kind));
            graph.connect(main_source, "output", node, "input").unwrap();
            graph
                .connect(control_source, "output", node, second_port)
                .unwrap();

            assert_eq!(
                effective_output_type(&graph, node, kind, PortType::Color),
                PortType::Gray,
                "{kind} should follow its main input's type"
            );
        }
    }

    /// Edge Detect is a real converter, not a passthrough: its own .wgsl
    /// reduces the input to luminance and broadcasts that back out, so the
    /// output is always Gray-equivalent regardless of whether the input was
    /// genuine Color or a Gray broadcast — declared statically Gray, not
    /// resolved via the type-preserving mechanism at all.
    #[test]
    fn edge_detect_output_is_always_gray_even_with_a_color_input() {
        let mut graph = Graph::new();
        let color_source = graph.add_node(Node::new("uniform_color"));
        let edge = graph.add_node(Node::new("edge_detect"));
        graph
            .connect(color_source, "output", edge, "input")
            .unwrap();

        assert_eq!(
            effective_output_type(&graph, edge, "edge_detect", PortType::Gray),
            PortType::Gray
        );
    }

    /// Plank Length/Width are meaningless for the Grid layout — only
    /// Herringbone's own geometry uses them.
    #[test]
    fn tile_generator_plank_params_are_relevant_only_for_herringbone_layout() {
        use super::param_is_relevant;
        use mc_core::ParamValue;
        use std::collections::BTreeMap;

        let params_for = |layout: &str| {
            let mut params = BTreeMap::new();
            params.insert("layout".to_string(), ParamValue::Enum(layout.to_string()));
            params
        };

        for param in ["plank_length", "plank_width"] {
            assert!(!param_is_relevant(
                "tile_generator",
                param,
                &params_for("Grid")
            ));
            assert!(param_is_relevant(
                "tile_generator",
                param,
                &params_for("Herringbone")
            ));
        }

        // Rows is the inverse: relevant for Grid (independent row density),
        // hidden for Herringbone (density comes from Columns alone — see
        // `tile_generator::pack_params`).
        assert!(param_is_relevant(
            "tile_generator",
            "rows",
            &params_for("Grid")
        ));
        assert!(!param_is_relevant(
            "tile_generator",
            "rows",
            &params_for("Herringbone")
        ));
    }

    /// Each Fabric Generator weave-specific param should only be relevant
    /// for its own Weave Type, so the panel can hide the other two instead
    /// of showing three mode-specific sliders that mostly do nothing.
    #[test]
    fn fabric_generator_weave_specific_params_are_relevant_only_for_their_own_weave_type() {
        use super::param_is_relevant;
        use mc_core::ParamValue;
        use std::collections::BTreeMap;

        let params_for = |weave_type: &str| {
            let mut params = BTreeMap::new();
            params.insert(
                "weave_type".to_string(),
                ParamValue::Enum(weave_type.to_string()),
            );
            params
        };

        for (weave_type, relevant_param) in [
            ("Plain Weave", "thread_group_size"),
            ("Twill", "twill_count"),
            ("Herringbone", "twill_count"),
            ("Rib Knit", "rib_width"),
        ] {
            let params = params_for(weave_type);
            assert!(
                param_is_relevant("fabric_generator", relevant_param, &params),
                "{relevant_param} should be relevant when Weave Type is {weave_type}"
            );
            for other_param in ["thread_group_size", "twill_count", "rib_width"] {
                if other_param == relevant_param {
                    continue;
                }
                assert!(
                    !param_is_relevant("fabric_generator", other_param, &params),
                    "{other_param} should NOT be relevant when Weave Type is {weave_type}"
                );
            }
        }

        // Stockinette uses none of the three.
        let params = params_for("Stockinette");
        for param in ["thread_group_size", "twill_count", "rib_width"] {
            assert!(!param_is_relevant("fabric_generator", param, &params));
        }
    }

    /// Image Import's effective output type follows its own "grayscale"
    /// param, not an upstream connection (it has no inputs at all) — the
    /// one case `effective_output_type` resolves from a node's own params
    /// instead of what's wired into it.
    #[test]
    fn image_import_effective_type_follows_its_own_grayscale_param() {
        use super::effective_output_type;
        use mc_core::ParamValue;

        let mut graph = Graph::new();
        let node = graph.add_node(Node::new("image_import"));
        assert_eq!(
            effective_output_type(&graph, node, "image_import", PortType::Color),
            PortType::Color,
            "grayscale off by default should stay Color"
        );

        graph
            .set_param(node, "grayscale", ParamValue::Bool(true))
            .unwrap();
        assert_eq!(
            effective_output_type(&graph, node, "image_import", PortType::Color),
            PortType::Gray,
            "toggling grayscale on should switch the effective output type to Gray"
        );
    }

    /// PATH_NODE_SPEC.md §1: Path wires only ever connect to another Path
    /// port — no broadcast to/from Gray or Color the way Gray->Color works.
    #[test]
    fn path_ports_only_connect_to_each_other() {
        use super::ports_compatible;
        assert!(ports_compatible(PortType::Path, PortType::Path));
        assert!(!ports_compatible(PortType::Path, PortType::Gray));
        assert!(!ports_compatible(PortType::Path, PortType::Color));
        assert!(!ports_compatible(PortType::Gray, PortType::Path));
        assert!(!ports_compatible(PortType::Color, PortType::Path));
        // Existing Gray/Color rules stay intact.
        assert!(ports_compatible(PortType::Gray, PortType::Color));
        assert!(!ports_compatible(PortType::Color, PortType::Gray));
    }

    /// HARDSURFACE_NODES.md §2: Sdf wires only ever connect to another Sdf
    /// port — same no-broadcast rule Path has, for the same reason (this
    /// carries geometric distance, not plain brightness).
    #[test]
    fn sdf_ports_only_connect_to_each_other() {
        use super::ports_compatible;
        assert!(ports_compatible(PortType::Sdf, PortType::Sdf));
        assert!(!ports_compatible(PortType::Sdf, PortType::Gray));
        assert!(!ports_compatible(PortType::Gray, PortType::Sdf));
        assert!(!ports_compatible(PortType::Sdf, PortType::Color));
        assert!(!ports_compatible(PortType::Color, PortType::Sdf));
        assert!(!ports_compatible(PortType::Sdf, PortType::Path));
        assert!(!ports_compatible(PortType::Path, PortType::Sdf));
    }
}
