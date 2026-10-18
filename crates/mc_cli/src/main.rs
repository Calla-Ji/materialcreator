// SPDX-License-Identifier: MIT OR Apache-2.0
//! mc_cli: headless rendering, with no window.
//!
//! Builds the `bricks` example graph (CLAUDE.md's M4 acceptance target),
//! saves it to `examples/graphs/bricks.mcgraph`, reloads it (exercising the
//! real file round trip, not just an in-memory one), evaluates it via
//! `mc_nodes::registry::evaluate`, dumps a quick-look PNG per output under
//! `out/`, and (M5) exports the full material via `mc_nodes::export` for
//! both the Blender and Unreal presets — proving the naming/packing/
//! color-space pipeline end to end, at a resolution different from the
//! graph's own live-preview resolution.
//!
//! Also builds `glitch_dither_demo` (the Glitch & Dither task brief's §6
//! example graph) the same way — built in code and saved, not hand-authored
//! JSON, so it stays the living source of truth the same way `bricks` does.
//!
//! Run with no arguments to do all of the above. Run `mc_cli render
//! <graph.mcgraph> --preset <blender|unreal>` (CLAUDE.md §8) to export any
//! single `.mcgraph` file instead — see `render_command`'s own doc comment,
//! including `--variations N` for previewing a graph's own Seed-param
//! jitter across several re-rolls in one command.

use anyhow::Context;
use clap::{Parser, Subcommand, ValueEnum};

fn build_bricks_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;
    // Grout depth is fine relative to a whole tile — the global 0.15
    // default over-displaces it.
    graph.settings.height_scale = 0.08;

    let bricks = graph.add_node(
        mc_core::Node::new("tile_generator")
            .with_param("rows", mc_core::ParamValue::Int(6))
            .with_param("columns", mc_core::ParamValue::Int(4))
            .with_param("offset", mc_core::ParamValue::Float(0.5))
            .with_param("gap", mc_core::ParamValue::Float(0.03))
            .with_param("bevel", mc_core::ParamValue::Float(0.1))
            .with_param("size_jitter", mc_core::ParamValue::Float(0.1))
            .with_param("luminance_jitter", mc_core::ParamValue::Float(0.2))
            .with_param("seed", mc_core::ParamValue::Seed(1)),
    );

    let mortar_color = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.42, 0.40, 0.38, 1.0])),
    );
    let brick_color = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.55, 0.28, 0.20, 1.0])),
    );
    let colorize = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Copy".to_string())),
    );

    // Clouds' soft, blobby fBm reads as the classic mottled color
    // variegation real fired clay brick faces have — masked to brick area
    // only (below) so mortar stays a flat, uniform color.
    let variation_noise = graph.add_node(
        mc_core::Node::new("clouds")
            .with_param("scale", mc_core::ParamValue::Int(6))
            .with_param("softness", mc_core::ParamValue::Float(0.6))
            .with_param("seed", mc_core::ParamValue::Seed(7)),
    );
    let varied_color = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Multiply".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.3)),
    );

    let normal = graph.add_node(
        mc_core::Node::new("height_to_normal")
            .with_param("intensity", mc_core::ParamValue::Float(2.0)),
    );

    let roughness = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.65, 0.65, 0.65, 1.0])),
    );
    let metallic = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.0, 0.0, 0.0, 1.0])),
    );

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));
    let out_normal = graph.add_node(mc_core::Node::new("output_normal"));
    let out_height = graph.add_node(mc_core::Node::new("output_height"));
    let out_roughness = graph.add_node(mc_core::Node::new("output_roughness"));
    let out_metallic = graph.add_node(mc_core::Node::new("output_metallic"));
    let out_ao = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));

    graph
        .connect(mortar_color, "output", colorize, "base")
        .unwrap();
    graph
        .connect(brick_color, "output", colorize, "blend")
        .unwrap();
    graph.connect(bricks, "output", colorize, "mask").unwrap();

    graph
        .connect(colorize, "output", varied_color, "base")
        .unwrap();
    graph
        .connect(variation_noise, "output", varied_color, "blend")
        .unwrap();
    // `bricks`' own output doubles as the brick/mortar mask (same texture
    // already used as `colorize`'s own mask above) — confines the Clouds
    // variegation to brick faces, leaving mortar a flat, uniform color.
    graph
        .connect(bricks, "output", varied_color, "mask")
        .unwrap();

    graph.connect(bricks, "output", normal, "input").unwrap();

    graph
        .connect(varied_color, "output", out_base_color, "input")
        .unwrap();
    graph
        .connect(normal, "output", out_normal, "input")
        .unwrap();
    graph
        .connect(bricks, "output", out_height, "input")
        .unwrap();
    graph
        .connect(roughness, "output", out_roughness, "input")
        .unwrap();
    graph
        .connect(metallic, "output", out_metallic, "input")
        .unwrap();
    // The mortar/bricks height pattern doubles as a crude ambient-occlusion
    // proxy: mortar gaps (dark) read as occluded, brick faces (bright) as
    // unoccluded — good enough for a demo graph, not a real AO bake.
    graph.connect(bricks, "output", out_ao, "input").unwrap();

    graph
}

/// The Glitch & Dither task brief's §6 example: `noise -> Gradient Map ->
/// Block Shift -> Channel Split -> Ordered Dither -> Output: Base Color`.
fn build_glitch_dither_demo_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;
    // Not a real height map semantically — just keep displacement subtle.
    graph.settings.height_scale = 0.03;

    let noise = graph.add_node(
        mc_core::Node::new("perlin_noise")
            .with_param("scale", mc_core::ParamValue::Int(6))
            .with_param("octaves", mc_core::ParamValue::Int(4))
            .with_param("seed", mc_core::ParamValue::Seed(2)),
    );

    let colorize = graph.add_node(mc_core::Node::new("gradient_map").with_param(
        "gradient",
        mc_core::ParamValue::Gradient(vec![
            mc_core::GradientStop {
                position: 0.0,
                color: [0.05, 0.05, 0.2, 1.0],
            },
            mc_core::GradientStop {
                position: 0.5,
                color: [0.9, 0.45, 0.1, 1.0],
            },
            mc_core::GradientStop {
                position: 1.0,
                color: [1.0, 1.0, 1.0, 1.0],
            },
        ]),
    ));

    let block_shift = graph.add_node(
        mc_core::Node::new("block_shift")
            .with_param("columns", mc_core::ParamValue::Int(12))
            .with_param("rows", mc_core::ParamValue::Int(12))
            .with_param("amount", mc_core::ParamValue::Float(0.25))
            .with_param("max_offset", mc_core::ParamValue::Float(0.08))
            .with_param("direction", mc_core::ParamValue::Enum("Both".to_string()))
            .with_param("seed", mc_core::ParamValue::Seed(3)),
    );

    let channel_split = graph.add_node(
        mc_core::Node::new("channel_split")
            .with_param("offset_r", mc_core::ParamValue::Vec2([0.015, 0.0]))
            .with_param("offset_g", mc_core::ParamValue::Vec2([0.0, 0.0]))
            .with_param("offset_b", mc_core::ParamValue::Vec2([-0.015, 0.0]))
            .with_param("strength", mc_core::ParamValue::Float(0.8)),
    );

    let dither = graph.add_node(
        mc_core::Node::new("ordered_dither")
            .with_param(
                "pattern",
                mc_core::ParamValue::Enum("Bayer 4x4".to_string()),
            )
            .with_param("cells", mc_core::ParamValue::Int(96))
            .with_param("levels", mc_core::ParamValue::Int(5))
            .with_param("spread", mc_core::ParamValue::Float(0.8))
            .with_param("mode", mc_core::ParamValue::Enum("Luminance".to_string())),
    );

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));

    graph.connect(noise, "output", colorize, "input").unwrap();
    graph
        .connect(colorize, "output", block_shift, "image")
        .unwrap();
    graph
        .connect(block_shift, "output", channel_split, "image")
        .unwrap();
    graph
        .connect(channel_split, "output", dither, "image")
        .unwrap();
    graph
        .connect(dither, "output", out_base_color, "input")
        .unwrap();

    graph
}

/// CLAUDE.md §10's M6 example graph: chunky Voronoi "F1" rock lumps, rounded
/// with Bevel, carved by two overlapping Voronoi "Edges" crack networks,
/// with mottled color variation and a sparse moss overlay favoring flat
/// tops over crevices.
fn build_rock_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;
    // Chunky bevel relief needs more headroom than the other examples.
    graph.settings.height_scale = 0.12;

    // F1's own distance field already domes smoothly within each cell —
    // a rock-chunk shape for free, with no extra shaping needed before
    // Bevel rounds its cell-boundary edges further below.
    let rock_shapes = graph.add_node(
        mc_core::Node::new("voronoi")
            .with_param("scale", mc_core::ParamValue::Int(5))
            .with_param("jitter", mc_core::ParamValue::Float(0.85))
            .with_param("mode", mc_core::ParamValue::Enum("F1".to_string()))
            .with_param("seed", mc_core::ParamValue::Seed(20)),
    );
    let rock_shapes_round = graph.add_node(
        mc_core::Node::new("bevel")
            .with_param("distance", mc_core::ParamValue::Float(0.08))
            .with_param("convexity", mc_core::ParamValue::Float(0.5)),
    );
    graph
        .connect(rock_shapes, "output", rock_shapes_round, "input")
        .unwrap();

    let surface_noise = graph.add_node(
        mc_core::Node::new("fractal_sum")
            .with_param("scale", mc_core::ParamValue::Int(6))
            .with_param("octaves", mc_core::ParamValue::Int(5))
            .with_param("gain", mc_core::ParamValue::Float(0.55))
            .with_param("seed", mc_core::ParamValue::Seed(12)),
    );
    let combined_shape = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Overlay".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.3)),
    );
    graph
        .connect(rock_shapes_round, "output", combined_shape, "base")
        .unwrap();
    graph
        .connect(surface_noise, "output", combined_shape, "blend")
        .unwrap();

    // "Edges" reads bright exactly at a cell boundary and ~0 everywhere
    // else (confirmed by rendering it alone) — the opposite polarity of
    // what a height map wants (cracks recessed/dark, faces raised/bright),
    // so it's used as a *mask* carving black into the shape above, not
    // blended directly into the driving signal itself. Two independent
    // scales (coarse fissures + fine secondary cracking), merged with Max
    // so either layer's own edge wins.
    let cracks = graph.add_node(
        mc_core::Node::new("voronoi")
            .with_param("scale", mc_core::ParamValue::Int(8))
            .with_param("jitter", mc_core::ParamValue::Float(0.9))
            .with_param("mode", mc_core::ParamValue::Enum("Edges".to_string()))
            .with_param("seed", mc_core::ParamValue::Seed(11)),
    );
    let cracks2 = graph.add_node(
        mc_core::Node::new("voronoi")
            .with_param("scale", mc_core::ParamValue::Int(14))
            .with_param("jitter", mc_core::ParamValue::Float(0.95))
            .with_param("mode", mc_core::ParamValue::Enum("Edges".to_string()))
            .with_param("seed", mc_core::ParamValue::Seed(17)),
    );
    let cracks_combined = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Max".to_string())),
    );
    graph
        .connect(cracks, "output", cracks_combined, "base")
        .unwrap();
    graph
        .connect(cracks2, "output", cracks_combined, "blend")
        .unwrap();

    let black = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.0, 0.0, 0.0, 1.0])),
    );
    let height = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Multiply".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.85)),
    );
    graph
        .connect(combined_shape, "output", height, "base")
        .unwrap();
    graph.connect(black, "output", height, "blend").unwrap();
    graph
        .connect(cracks_combined, "output", height, "mask")
        .unwrap();

    let colorize = graph.add_node(mc_core::Node::new("gradient_map").with_param(
        "gradient",
        mc_core::ParamValue::Gradient(vec![
            mc_core::GradientStop {
                position: 0.0,
                color: [0.12, 0.11, 0.10, 1.0],
            },
            mc_core::GradientStop {
                position: 0.5,
                color: [0.35, 0.33, 0.30, 1.0],
            },
            mc_core::GradientStop {
                position: 1.0,
                color: [0.58, 0.55, 0.49, 1.0],
            },
        ]),
    ));
    graph.connect(height, "output", colorize, "input").unwrap();

    // A large-scale, independent noise lightens/darkens patches of the
    // surface — real color variation, not just a single smooth
    // height-driven tone.
    let color_variation_noise = graph.add_node(
        mc_core::Node::new("fractal_sum")
            .with_param("scale", mc_core::ParamValue::Int(3))
            .with_param("octaves", mc_core::ParamValue::Int(3))
            .with_param("gain", mc_core::ParamValue::Float(0.5))
            .with_param("seed", mc_core::ParamValue::Seed(33)),
    );
    let color_with_variation = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Overlay".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.3)),
    );
    graph
        .connect(colorize, "output", color_with_variation, "base")
        .unwrap();
    graph
        .connect(
            color_variation_noise,
            "output",
            color_with_variation,
            "blend",
        )
        .unwrap();

    // A sparse, high-threshold noise mask, favoring flat tops over
    // crevices (multiplied by the inverted crack mask) — a very partial
    // moss overlay, not a wholesale green tint.
    let moss_noise = graph.add_node(
        mc_core::Node::new("fractal_sum")
            .with_param("scale", mc_core::ParamValue::Int(4))
            .with_param("octaves", mc_core::ParamValue::Int(2))
            .with_param("gain", mc_core::ParamValue::Float(0.6))
            .with_param("seed", mc_core::ParamValue::Seed(44)),
    );
    // FractalSum's own output rarely spans the full 0..1 range (2 octaves
    // at this gain measured ~0.3..0.7 in practice) — 0.6 lands around its
    // own 90th percentile, not 0..1's, for genuinely sparse (~10%) coverage
    // rather than an empty mask.
    let moss_mask = graph.add_node(
        mc_core::Node::new("threshold").with_param("threshold", mc_core::ParamValue::Float(0.6)),
    );
    graph
        .connect(moss_noise, "output", moss_mask, "input")
        .unwrap();
    let cracks_inverted = graph.add_node(mc_core::Node::new("invert"));
    graph
        .connect(cracks_combined, "output", cracks_inverted, "input")
        .unwrap();
    let moss_mask_final = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Multiply".to_string())),
    );
    graph
        .connect(moss_mask, "output", moss_mask_final, "base")
        .unwrap();
    graph
        .connect(cracks_inverted, "output", moss_mask_final, "blend")
        .unwrap();
    let moss_green = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.22, 0.35, 0.16, 1.0])),
    );
    let base_color = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Copy".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.8)),
    );
    graph
        .connect(color_with_variation, "output", base_color, "base")
        .unwrap();
    graph
        .connect(moss_green, "output", base_color, "blend")
        .unwrap();
    graph
        .connect(moss_mask_final, "output", base_color, "mask")
        .unwrap();

    let normal = graph.add_node(
        mc_core::Node::new("height_to_normal")
            .with_param("intensity", mc_core::ParamValue::Float(1.5)),
    );
    graph.connect(height, "output", normal, "input").unwrap();

    let roughness = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.85, 0.85, 0.85, 1.0])),
    );
    let metallic = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.0, 0.0, 0.0, 1.0])),
    );

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));
    let out_normal = graph.add_node(mc_core::Node::new("output_normal"));
    let out_height = graph.add_node(mc_core::Node::new("output_height"));
    let out_roughness = graph.add_node(mc_core::Node::new("output_roughness"));
    let out_metallic = graph.add_node(mc_core::Node::new("output_metallic"));
    let out_ao = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));

    graph
        .connect(base_color, "output", out_base_color, "input")
        .unwrap();
    graph
        .connect(normal, "output", out_normal, "input")
        .unwrap();
    graph
        .connect(height, "output", out_height, "input")
        .unwrap();
    graph
        .connect(roughness, "output", out_roughness, "input")
        .unwrap();
    graph
        .connect(metallic, "output", out_metallic, "input")
        .unwrap();
    // Crevices between rock faces (where cracks/surface noise are dark) read
    // as occluded — the same crude-but-plausible proxy `bricks` uses.
    graph.connect(height, "output", out_ao, "input").unwrap();

    graph
}

/// CLAUDE.md §10's M6 example graph: a directionally-blurred fractal noise
/// gives both the brush streaks and a matching narrow-band roughness
/// variation, over a fully metallic base.
fn build_brushed_metal_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;
    // Fine brush streaks, minimal real relief.
    graph.settings.height_scale = 0.02;

    let noise = graph.add_node(
        mc_core::Node::new("fractal_sum")
            .with_param("scale", mc_core::ParamValue::Int(24))
            .with_param("octaves", mc_core::ParamValue::Int(3))
            .with_param("gain", mc_core::ParamValue::Float(0.5))
            .with_param("seed", mc_core::ParamValue::Seed(21)),
    );
    let brushed = graph.add_node(
        mc_core::Node::new("directional_blur")
            .with_param("radius", mc_core::ParamValue::Float(0.04))
            .with_param("angle", mc_core::ParamValue::Float(0.0)),
    );
    graph.connect(noise, "output", brushed, "input").unwrap();

    let steel_tint = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.72, 0.73, 0.75, 1.0])),
    );
    let colorize = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Multiply".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.4)),
    );
    graph
        .connect(steel_tint, "output", colorize, "base")
        .unwrap();
    graph.connect(brushed, "output", colorize, "blend").unwrap();

    let normal = graph.add_node(
        mc_core::Node::new("height_to_normal")
            .with_param("intensity", mc_core::ParamValue::Float(0.6)),
    );
    graph.connect(brushed, "output", normal, "input").unwrap();

    // Remaps the same brush-streak noise into a narrow roughness band
    // instead of a flat value — streaks read as faint variation in
    // reflection sharpness, not just in the normal map.
    let roughness = graph.add_node(
        mc_core::Node::new("levels")
            .with_param("out_low", mc_core::ParamValue::Float(0.25))
            .with_param("out_high", mc_core::ParamValue::Float(0.45)),
    );
    graph
        .connect(brushed, "output", roughness, "input")
        .unwrap();

    let metallic = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([1.0, 1.0, 1.0, 1.0])),
    );
    let ao = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([1.0, 1.0, 1.0, 1.0])),
    );

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));
    let out_normal = graph.add_node(mc_core::Node::new("output_normal"));
    let out_height = graph.add_node(mc_core::Node::new("output_height"));
    let out_roughness = graph.add_node(mc_core::Node::new("output_roughness"));
    let out_metallic = graph.add_node(mc_core::Node::new("output_metallic"));
    let out_ao = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));

    graph
        .connect(colorize, "output", out_base_color, "input")
        .unwrap();
    graph
        .connect(normal, "output", out_normal, "input")
        .unwrap();
    graph
        .connect(brushed, "output", out_height, "input")
        .unwrap();
    graph
        .connect(roughness, "output", out_roughness, "input")
        .unwrap();
    graph
        .connect(metallic, "output", out_metallic, "input")
        .unwrap();
    graph.connect(ao, "output", out_ao, "input").unwrap();

    graph
}

/// Real plied-yarn geometry (Thread) woven into cloth (Fabric Generator) —
/// not Pattern Generator standing in with brick-sized tiles shrunk down to
/// look vaguely thread-like, which is what this example used to do before
/// both of those nodes existed.
fn build_fabric_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;
    // Fine thread texture — the global 0.15 default reads as a furry cube.
    graph.settings.height_scale = 0.03;

    let thread = graph.add_node(
        mc_core::Node::new("thread_shape").with_param("plies", mc_core::ParamValue::Int(3)),
    );

    let weave = graph.add_node(
        mc_core::Node::new("fabric_generator")
            .with_param(
                "weave_type",
                mc_core::ParamValue::Enum("Plain Weave".to_string()),
            )
            .with_param("rows", mc_core::ParamValue::Int(24))
            .with_param("columns", mc_core::ParamValue::Int(24)),
    );
    graph.connect(thread, "height", weave, "fiber").unwrap();
    graph
        .connect(thread, "ply_id", weave, "fiber_ply_id")
        .unwrap();

    let fiber_noise = graph.add_node(
        mc_core::Node::new("fractal_sum")
            .with_param("scale", mc_core::ParamValue::Int(32))
            .with_param("octaves", mc_core::ParamValue::Int(4))
            .with_param("gain", mc_core::ParamValue::Float(0.5))
            .with_param("seed", mc_core::ParamValue::Seed(32)),
    );
    let combined = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Overlay".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.3)),
    );
    graph.connect(weave, "output", combined, "base").unwrap();
    graph
        .connect(fiber_noise, "output", combined, "blend")
        .unwrap();

    let colorize = graph.add_node(mc_core::Node::new("gradient_map").with_param(
        "gradient",
        mc_core::ParamValue::Gradient(vec![
            mc_core::GradientStop {
                position: 0.0,
                color: [0.10, 0.16, 0.30, 1.0],
            },
            mc_core::GradientStop {
                position: 1.0,
                color: [0.22, 0.32, 0.52, 1.0],
            },
        ]),
    ));
    graph
        .connect(combined, "output", colorize, "input")
        .unwrap();

    // Thread's own per-ply color data (3 plies, matching `thread` above),
    // mapped to genuinely contrasting fleck colors (not just lighter/darker
    // shades of the same denim blue — multiplying same-hue stops onto a
    // dark base only dims it, it doesn't read as multicolor) and composited
    // with Copy so each ply's own color actually shows through, the classic
    // heathered-yarn look.
    let ply_color = graph.add_node(mc_core::Node::new("gradient_map").with_param(
        "gradient",
        mc_core::ParamValue::Gradient(vec![
            mc_core::GradientStop {
                position: 0.0,
                color: [0.12, 0.16, 0.35, 1.0],
            },
            mc_core::GradientStop {
                position: 0.5,
                color: [0.55, 0.30, 0.12, 1.0],
            },
            mc_core::GradientStop {
                position: 1.0,
                color: [0.80, 0.75, 0.60, 1.0],
            },
        ]),
    ));
    graph.connect(weave, "ply_id", ply_color, "input").unwrap();
    let base_color = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Copy".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.5)),
    );
    graph
        .connect(colorize, "output", base_color, "base")
        .unwrap();
    graph
        .connect(ply_color, "output", base_color, "blend")
        .unwrap();

    let normal = graph.add_node(
        mc_core::Node::new("height_to_normal")
            .with_param("intensity", mc_core::ParamValue::Float(0.8)),
    );
    graph.connect(combined, "output", normal, "input").unwrap();

    let roughness = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.9, 0.9, 0.9, 1.0])),
    );
    let metallic = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.0, 0.0, 0.0, 1.0])),
    );

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));
    let out_normal = graph.add_node(mc_core::Node::new("output_normal"));
    let out_height = graph.add_node(mc_core::Node::new("output_height"));
    let out_roughness = graph.add_node(mc_core::Node::new("output_roughness"));
    let out_metallic = graph.add_node(mc_core::Node::new("output_metallic"));
    let out_ao = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));

    graph
        .connect(base_color, "output", out_base_color, "input")
        .unwrap();
    graph
        .connect(normal, "output", out_normal, "input")
        .unwrap();
    graph
        .connect(combined, "output", out_height, "input")
        .unwrap();
    graph
        .connect(roughness, "output", out_roughness, "input")
        .unwrap();
    graph
        .connect(metallic, "output", out_metallic, "input")
        .unwrap();
    graph.connect(combined, "output", out_ao, "input").unwrap();

    graph
}

/// Herringbone parquet, not a single squashed noise: one "V" repeat unit
/// (two `Shape` rectangles rotated to \u{b1}45\u{b0} — safe to rotate at any
/// angle since a single centered Shape is a pure procedural SDF evaluated
/// directly per pixel, never resampled across a wrapped boundary the way
/// `Transform2D`'s own "only 0.25-turn multiples stay exactly tileable"
/// warning is about), tiled via two separate `Pattern Generator`s sharing
/// one grid so each diagonal direction's own mask survives independently
/// for per-plank-orientation color variation, plus grain noise and a couple
/// of sparse knots.
fn build_wood_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;
    // Plank relief reads as moderate, not deep canyons.
    graph.settings.height_scale = 0.05;

    // Real interlocking parquet herringbone, built directly by Pattern
    // Generator's own Herringbone layout — replaces an earlier 5-node
    // recipe (two Shape rectangles rotated \u{b1}45\u{b0} each feeding their own
    // Pattern Generator, combined with Blend: Max) from before that layout
    // existed as a built-in option. Offset below 0.25 selects the layout's
    // 45\u{b0}-diagonal chevron, matching what the old recipe's own rotated
    // Shapes were built to produce. Luminance Jitter gives each plank its
    // own brightness, which Gradient Map below turns into a spread of wood
    // tones instead of the old two-tone alternating-orientation trick
    // (which needed a per-orientation mask this single node doesn't
    // expose).
    let height = graph.add_node(
        mc_core::Node::new("tile_generator")
            .with_param(
                "layout",
                mc_core::ParamValue::Enum("Herringbone".to_string()),
            )
            // Herringbone's own per-plank jitter hash is wrapped to the
            // smallest period that still guarantees exact seamless tiling
            // (`periods`/`n`, derived from `columns` — see
            // `tile_generator.wgsl`'s own doc comment) — at `columns: 4`
            // that period was only ~16 distinct values total, far fewer
            // than the number of visibly separate boards, so many
            // genuinely different planks were forced to share one shade
            // (confirmed via a pixel-level diagnostic, not assumed). 10
            // gives ~100, enough that jitter reads as genuinely per-board
            // instead of blocky — the real tradeoff this reveals is that
            // jitter resolution is tied to density for this layout, same
            // as every other tiling node here; there's no way to keep
            // exactly this density AND get finer jitter without that
            // tradeoff.
            .with_param("columns", mc_core::ParamValue::Int(10))
            .with_param("offset", mc_core::ParamValue::Float(0.0))
            .with_param("gap", mc_core::ParamValue::Float(0.03))
            .with_param("bevel", mc_core::ParamValue::Float(0.05))
            .with_param("size_jitter", mc_core::ParamValue::Float(0.08))
            .with_param("luminance_jitter", mc_core::ParamValue::Float(0.5))
            .with_param("seed", mc_core::ParamValue::Seed(41)),
    );

    // Kept clear of near-black/near-white on purpose: Luminance Jitter above
    // can legitimately drive the raw per-plank value all the way from 0 to
    // 1 (same jitter convention every other node here uses — not a bug),
    // and the first version of this gradient mapped that extreme straight
    // to near-black/near-white, which read as blown-out/broken rather than
    // "a darker board" at high jitter. Every stop below stays inside a
    // plausible wood-tone range regardless of where in 0..1 it gets hit.
    let wood_tone = graph.add_node(mc_core::Node::new("gradient_map").with_param(
        "gradient",
        mc_core::ParamValue::Gradient(vec![
            mc_core::GradientStop {
                position: 0.0,
                color: [0.18, 0.11, 0.06, 1.0],
            },
            mc_core::GradientStop {
                position: 0.5,
                color: [0.40, 0.25, 0.13, 1.0],
            },
            mc_core::GradientStop {
                position: 1.0,
                color: [0.58, 0.40, 0.22, 1.0],
            },
        ]),
    ));
    graph.connect(height, "output", wood_tone, "input").unwrap();

    // Fine grain, warped by a slow Perlin field so the streaks wander
    // organically instead of running in perfectly straight lines — real
    // wood grain never does — then multiplied on at low opacity, the same
    // role the previous (unwarped) streaky fBm played here.
    let grain_noise = graph.add_node(
        mc_core::Node::new("fractal_sum")
            .with_param("scale", mc_core::ParamValue::Int(8))
            .with_param("octaves", mc_core::ParamValue::Int(4))
            .with_param("gain", mc_core::ParamValue::Float(0.5))
            .with_param("seed", mc_core::ParamValue::Seed(41)),
    );
    let grain_distortion = graph.add_node(
        mc_core::Node::new("perlin_noise")
            .with_param("scale", mc_core::ParamValue::Int(4))
            .with_param("octaves", mc_core::ParamValue::Int(2))
            .with_param("seed", mc_core::ParamValue::Seed(7)),
    );
    let warped_grain = graph.add_node(
        mc_core::Node::new("warp").with_param("amount", mc_core::ParamValue::Float(0.04)),
    );
    graph
        .connect(grain_noise, "output", warped_grain, "input")
        .unwrap();
    graph
        .connect(grain_distortion, "output", warped_grain, "intensity")
        .unwrap();

    let with_grain = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Multiply".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.25)),
    );
    graph
        .connect(wood_tone, "output", with_grain, "base")
        .unwrap();
    graph
        .connect(warped_grain, "output", with_grain, "blend")
        .unwrap();

    // Soft organic whorls instead of hard-edged Voronoi blobs: Anisotropic
    // Noise's own streaky, wood-grain-shaped base, warped by a separate
    // Perlin field (its contrast boosted by Levels' gamma so the distortion
    // reads clearly) — no threshold at all, so the result is a continuous
    // value that blends in as soft burl-like swirls with a soft edge, not a
    // discrete blob with a hard binary cutoff.
    let whorl_base = graph.add_node(
        mc_core::Node::new("anisotropic_noise")
            .with_param("scale_x", mc_core::ParamValue::Int(3))
            .with_param("scale_y", mc_core::ParamValue::Int(10))
            .with_param("octaves", mc_core::ParamValue::Int(3))
            .with_param("seed", mc_core::ParamValue::Seed(60)),
    );
    let whorl_distortion = graph.add_node(
        mc_core::Node::new("perlin_noise")
            .with_param("scale", mc_core::ParamValue::Int(3))
            .with_param("octaves", mc_core::ParamValue::Int(3))
            .with_param("seed", mc_core::ParamValue::Seed(61)),
    );
    let whorl_distortion_shaped = graph.add_node(
        mc_core::Node::new("levels").with_param("gamma", mc_core::ParamValue::Float(1.5)),
    );
    graph
        .connect(whorl_distortion, "output", whorl_distortion_shaped, "input")
        .unwrap();
    let knot_mask = graph.add_node(
        mc_core::Node::new("warp").with_param("amount", mc_core::ParamValue::Float(0.08)),
    );
    graph
        .connect(whorl_base, "output", knot_mask, "input")
        .unwrap();
    graph
        .connect(whorl_distortion_shaped, "output", knot_mask, "intensity")
        .unwrap();
    let knot_color = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.12, 0.07, 0.04, 1.0])),
    );

    // Confine knots to the plank surface itself (`height > 0`), not the
    // dark mortar/seam line between boards — without this, a knot that
    // happens to straddle a seam paints straight through the gap too,
    // reading as a smudge floating over the structure rather than grain
    // that belongs to one board. A knot that genuinely spans two adjacent
    // planks now gets naturally interrupted by the gap between them, the
    // way a real floor's knot pattern would be cut by the joint.
    let plank_mask = graph.add_node(
        mc_core::Node::new("threshold").with_param("threshold", mc_core::ParamValue::Float(0.05)),
    );
    graph
        .connect(height, "output", plank_mask, "input")
        .unwrap();
    let confined_knot_mask = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Multiply".to_string())),
    );
    graph
        .connect(knot_mask, "output", confined_knot_mask, "base")
        .unwrap();
    graph
        .connect(plank_mask, "output", confined_knot_mask, "blend")
        .unwrap();

    let base_color = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Copy".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.6)),
    );
    graph
        .connect(with_grain, "output", base_color, "base")
        .unwrap();
    graph
        .connect(knot_color, "output", base_color, "blend")
        .unwrap();
    graph
        .connect(confined_knot_mask, "output", base_color, "mask")
        .unwrap();

    let normal = graph.add_node(mc_core::Node::new("height_to_normal"));
    graph.connect(height, "output", normal, "input").unwrap();
    let ao = graph.add_node(mc_core::Node::new("height_to_ao"));
    graph.connect(height, "output", ao, "input").unwrap();

    let roughness = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.5, 0.5, 0.5, 1.0])),
    );
    let metallic = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.0, 0.0, 0.0, 1.0])),
    );

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));
    let out_normal = graph.add_node(mc_core::Node::new("output_normal"));
    let out_height = graph.add_node(mc_core::Node::new("output_height"));
    let out_roughness = graph.add_node(mc_core::Node::new("output_roughness"));
    let out_metallic = graph.add_node(mc_core::Node::new("output_metallic"));
    let out_ao = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));

    graph
        .connect(base_color, "output", out_base_color, "input")
        .unwrap();
    graph
        .connect(normal, "output", out_normal, "input")
        .unwrap();
    graph
        .connect(height, "output", out_height, "input")
        .unwrap();
    graph
        .connect(roughness, "output", out_roughness, "input")
        .unwrap();
    graph
        .connect(metallic, "output", out_metallic, "input")
        .unwrap();
    graph.connect(ao, "output", out_ao, "input").unwrap();

    graph
}

/// Hard-Surface Nodes' own demo chain (`docs/hardsurface_nodes/
/// HARDSURFACE_NODES.md` §4): Panel Layout -> a per-panel 0.75/1.0/1.25
/// height variety (`lvl`, a Gradient Map with razor-thin transition
/// bands — not Blend, since `blend.wgsl`'s own `combine()` result is
/// always clamped to `[0, 1]` regardless of mode, so no chain of Blend
/// nodes could ever reach `1.25` at all; `graph.connect` itself does no
/// port-type validation, confirmed by reading `mc_core::Graph::connect`
/// directly, so Gradient Map's declared-`Color` output can freely feed
/// an `Sdf`/`Gray` input the same way the editor's own Gray<->Color
/// broadcast already treats them) -> Bevel Profile -> Fasteners seated
/// on the beveled, leveled height -> Edge Wear. Verified against
/// `golden/demo_height.png` by `tests/hardsurface_demo.rs`.
fn build_hardsurface_demo_graph() -> mc_core::Graph {
    let mut graph = mc_core::Graph::new();
    graph.settings.resolution = 512;
    // Panel bevels/rivets at tile-unit scale, not canyon-deep.
    graph.settings.height_scale = 0.06;

    let panels = graph.add_node(mc_core::Node::new("panel_layout"));

    // lvl = 0.75 / 1.0 / 1.25 below/within/above the [0.25, 0.85] band,
    // matching the reference's own `where(pid<0.25, 0.75, where(pid>0.85,
    // 1.25, 1.0))`. A single Gradient Map with razor-thin transition bands
    // (width 0.0002, far narrower than any gap between this seed's 18
    // panels' own hashed IDs could ever land inside) reproduces that hard
    // step directly — deliberately *not* Blend for this: `blend.wgsl`'s
    // own `combine()` result is always clamped to `[0, 1]` regardless of
    // mode, so no chain of Blend nodes can ever produce `1.25` at all.
    // Gradient Map's own interpolated color has no such clamp.
    let lvl = graph.add_node(mc_core::Node::new("gradient_map").with_param(
        "gradient",
        mc_core::ParamValue::Gradient(vec![
            mc_core::GradientStop {
                position: 0.0,
                color: [0.75, 0.75, 0.75, 1.0],
            },
            mc_core::GradientStop {
                position: 0.2499,
                color: [0.75, 0.75, 0.75, 1.0],
            },
            mc_core::GradientStop {
                position: 0.2501,
                color: [1.0, 1.0, 1.0, 1.0],
            },
            mc_core::GradientStop {
                position: 0.8499,
                color: [1.0, 1.0, 1.0, 1.0],
            },
            mc_core::GradientStop {
                position: 0.8501,
                color: [1.25, 1.25, 1.25, 1.0],
            },
            mc_core::GradientStop {
                position: 1.0,
                color: [1.25, 1.25, 1.25, 1.0],
            },
        ]),
    ));
    graph.connect(panels, "panel_id", lvl, "input").unwrap();

    // Depth: 0.5 pre-folds the reference's own separate `* 0.5` — height
    // scales linearly with depth (base = 0), so bevel(depth=0.5) ==
    // bevel(depth=1.0) * 0.5 exactly.
    let bevel = graph.add_node(
        mc_core::Node::new("bevel_profile")
            .with_param("preset", mc_core::ParamValue::Enum("Round".to_string()))
            .with_param("width", mc_core::ParamValue::Float(0.006))
            .with_param("wall", mc_core::ParamValue::Float(0.35))
            .with_param("depth", mc_core::ParamValue::Float(0.5)),
    );
    graph.connect(panels, "sdf", bevel, "sdf").unwrap();

    let leveled_height = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Multiply".to_string())),
    );
    graph
        .connect(bevel, "height", leveled_height, "base")
        .unwrap();
    graph
        .connect(lvl, "output", leveled_height, "blend")
        .unwrap();

    let fasteners = graph.add_node(
        mc_core::Node::new("fasteners")
            .with_param("kind", mc_core::ParamValue::Enum("Rivet Dome".to_string()))
            .with_param("spacing", mc_core::ParamValue::Float(0.045))
            .with_param("head_radius", mc_core::ParamValue::Float(0.0055))
            .with_param("head_height", mc_core::ParamValue::Float(0.22))
            .with_param("seed", mc_core::ParamValue::Seed(3))
            .with_param("missing", mc_core::ParamValue::Float(0.04)),
    );
    graph.connect(panels, "path", fasteners, "path").unwrap();
    graph
        .connect(leveled_height, "output", fasteners, "base")
        .unwrap();

    let wear = graph.add_node(
        mc_core::Node::new("edge_wear").with_param("amount", mc_core::ParamValue::Float(0.45)),
    );
    graph.connect(panels, "sdf", wear, "sdf").unwrap();
    graph.connect(fasteners, "height", wear, "height").unwrap();

    let normal = graph.add_node(mc_core::Node::new("height_to_normal"));
    graph.connect(wear, "height", normal, "input").unwrap();
    let ao = graph.add_node(mc_core::Node::new("height_to_ao"));
    graph.connect(wear, "height", ao, "input").unwrap();

    // Base Color: painted panels with Wear exposing rusty bare metal
    // underneath and Cavity darkening seams/recesses with rust-stained
    // dirt — actually using the two masks Edge Wear exists to produce,
    // not just a flat fill.
    let paint_color = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.30, 0.31, 0.33, 1.0])),
    );
    let rust_color = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.45, 0.24, 0.11, 1.0])),
    );
    let dirt_color = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.16, 0.10, 0.06, 1.0])),
    );
    let worn_color = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Copy".to_string())),
    );
    graph
        .connect(paint_color, "output", worn_color, "base")
        .unwrap();
    graph
        .connect(rust_color, "output", worn_color, "blend")
        .unwrap();
    graph.connect(wear, "wear", worn_color, "mask").unwrap();
    let dirtied_color = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Copy".to_string())),
    );
    graph
        .connect(worn_color, "output", dirtied_color, "base")
        .unwrap();
    graph
        .connect(dirt_color, "output", dirtied_color, "blend")
        .unwrap();
    graph
        .connect(wear, "cavity", dirtied_color, "mask")
        .unwrap();

    // Rivets read a touch darker than the surrounding panel regardless of
    // wear/dirt state underneath — composited last via Fasteners' own
    // `mask` output (every rivet head's footprint).
    let rivet_color = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.18, 0.19, 0.20, 1.0])),
    );
    let base_color = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Copy".to_string()))
            .with_param("opacity", mc_core::ParamValue::Float(0.6)),
    );
    graph
        .connect(dirtied_color, "output", base_color, "base")
        .unwrap();
    graph
        .connect(rivet_color, "output", base_color, "blend")
        .unwrap();
    graph
        .connect(fasteners, "mask", base_color, "mask")
        .unwrap();

    // Roughness/Metallic: both driven by the same two masks — scratched
    // and dirty patches read rougher, and Wear's exposed metal reads
    // partially metallic, the paint around it fully dielectric. A single
    // `white` fill is reused as every "how much" `blend` layer below;
    // each mask (not `opacity`) is what varies the result per pixel
    // (`combine() = clamp(base + blend, 0, 1)`, then `mix(base, combine(),
    // mask * opacity)` — opacity just scales the mask uniformly, so a
    // mask value of 0.6 at `opacity: 1.0` already lands at exactly 0.6
    // of the way from `base` to the fully-added result).
    let white = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([1.0, 1.0, 1.0, 1.0])),
    );
    let roughness_base = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.5, 0.5, 0.5, 1.0])),
    );
    let roughness_worn = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Add".to_string())),
    );
    graph
        .connect(roughness_base, "output", roughness_worn, "base")
        .unwrap();
    graph
        .connect(white, "output", roughness_worn, "blend")
        .unwrap();
    graph.connect(wear, "wear", roughness_worn, "mask").unwrap();
    let roughness = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Add".to_string())),
    );
    graph
        .connect(roughness_worn, "output", roughness, "base")
        .unwrap();
    graph.connect(white, "output", roughness, "blend").unwrap();
    graph.connect(wear, "cavity", roughness, "mask").unwrap();

    // A painted metal panel reads with a subtle metallic sheen overall
    // (0.18, not 0.0 — flat 0 is what read as plastic), but rust and dirt
    // are dielectric, not shiny exposed steel, so both masks *subtract*
    // metallic back toward 0 instead of adding to it.
    let metallic_base = graph.add_node(
        mc_core::Node::new("uniform_color")
            .with_param("color", mc_core::ParamValue::Color([0.18, 0.18, 0.18, 1.0])),
    );
    let metallic_worn = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Subtract".to_string())),
    );
    graph
        .connect(metallic_base, "output", metallic_worn, "base")
        .unwrap();
    graph
        .connect(white, "output", metallic_worn, "blend")
        .unwrap();
    graph.connect(wear, "wear", metallic_worn, "mask").unwrap();
    let metallic = graph.add_node(
        mc_core::Node::new("blend")
            .with_param("mode", mc_core::ParamValue::Enum("Subtract".to_string())),
    );
    graph
        .connect(metallic_worn, "output", metallic, "base")
        .unwrap();
    graph.connect(white, "output", metallic, "blend").unwrap();
    graph.connect(wear, "cavity", metallic, "mask").unwrap();

    let out_base_color = graph.add_node(mc_core::Node::new("output_base_color"));
    let out_normal = graph.add_node(mc_core::Node::new("output_normal"));
    let out_height = graph.add_node(mc_core::Node::new("output_height"));
    let out_ao = graph.add_node(mc_core::Node::new("output_ambient_occlusion"));
    let out_roughness = graph.add_node(mc_core::Node::new("output_roughness"));
    let out_metallic = graph.add_node(mc_core::Node::new("output_metallic"));
    graph
        .connect(base_color, "output", out_base_color, "input")
        .unwrap();
    graph
        .connect(normal, "output", out_normal, "input")
        .unwrap();
    graph.connect(wear, "height", out_height, "input").unwrap();
    graph.connect(ao, "output", out_ao, "input").unwrap();
    graph
        .connect(roughness, "output", out_roughness, "input")
        .unwrap();
    graph
        .connect(metallic, "output", out_metallic, "input")
        .unwrap();

    graph
}

/// Column = topological depth from a source node, row = barycenter-ordered
/// within that column to cut down on wire crossings — the same algorithm
/// `mc_app::canvas::auto_layout` uses interactively, re-implemented here
/// rather than shared: `mc_app` is a GUI binary crate with no `[lib]`
/// target `mc_cli` could depend on, and pulling in `egui`/`eframe` as a
/// dependency of this headless tool just to reuse one function would be
/// backwards. Every node built in code above starts at `[0, 0]`
/// (`mc_core::Node`'s own default) — without this, every example graph
/// opened in the app shows every node stacked on top of the next at the
/// origin, confirmed by the owner's own screenshot of `hardsurface_demo`'s
/// Node Graph panel.
fn auto_layout_for_save(graph: &mut mc_core::Graph) {
    let Ok(order) = graph.topological_order() else {
        return;
    };
    if order.is_empty() {
        return;
    }

    let mut column: std::collections::HashMap<mc_core::NodeId, u32> =
        std::collections::HashMap::new();
    for &id in &order {
        let col = graph
            .connections()
            .iter()
            .filter(|c| c.to_node == id)
            .filter_map(|c| column.get(&c.from_node))
            .max()
            .map(|&upstream| upstream + 1)
            .unwrap_or(0);
        column.insert(id, col);
    }

    let column_count = column.values().max().map(|&c| c + 1).unwrap_or(0) as usize;
    let mut columns: Vec<Vec<mc_core::NodeId>> = vec![Vec::new(); column_count];
    for &id in &order {
        columns[column[&id] as usize].push(id);
    }

    // Every `output_*` node is always a graph leaf (nothing ever reads
    // from one), so moving them all into one shared final column — to
    // the right of everything else, regardless of each one's own natural
    // topological depth — is purely cosmetic and never disturbs a real
    // dependency. Matches the owner's own explicit request for example
    // graphs: "all output nodes are in one column".
    let mut output_ids: Vec<mc_core::NodeId> = Vec::new();
    for nodes_in_column in &mut columns {
        nodes_in_column.retain(|&id| {
            let is_output = graph
                .node(id)
                .is_some_and(|node| node.kind.starts_with("output_"));
            if is_output {
                output_ids.push(id);
            }
            !is_output
        });
    }
    columns.retain(|nodes_in_column| !nodes_in_column.is_empty());
    if !output_ids.is_empty() {
        columns.push(output_ids);
    }

    let mut row_of: std::collections::HashMap<mc_core::NodeId, f32> =
        std::collections::HashMap::new();
    for nodes_in_column in &mut columns {
        nodes_in_column.sort_by(|&a, &b| {
            let barycenter = |id: mc_core::NodeId| -> f32 {
                let upstream_rows: Vec<f32> = graph
                    .connections()
                    .iter()
                    .filter(|c| c.to_node == id)
                    .filter_map(|c| row_of.get(&c.from_node).copied())
                    .collect();
                if upstream_rows.is_empty() {
                    0.0
                } else {
                    upstream_rows.iter().sum::<f32>() / upstream_rows.len() as f32
                }
            };
            barycenter(a)
                .partial_cmp(&barycenter(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (row, &id) in nodes_in_column.iter().enumerate() {
            row_of.insert(id, row as f32);
        }
    }

    // Mirrors `mc_app::canvas`'s own node-box constants exactly, so a
    // graph built here lays out the same way the in-app "Auto Layout"
    // button would arrange it.
    const NODE_WIDTH: f32 = 170.0;
    const HEADER_HEIGHT: f32 = 24.0;
    const PORT_ROW_HEIGHT: f32 = 20.0;
    const BODY_PADDING: f32 = 6.0;
    const THUMBNAIL_SIZE: f32 = NODE_WIDTH - 2.0 * BODY_PADDING;
    const COLUMN_GAP: f32 = 80.0;
    const ROW_GAP: f32 = 30.0;

    let node_height = |kind: &str| -> f32 {
        let port_rows = mc_nodes::defs::node_def(kind)
            .map(|def| def.inputs.len().max(def.outputs.len()).max(1))
            .unwrap_or(1) as f32;
        HEADER_HEIGHT
            + BODY_PADDING
            + port_rows * PORT_ROW_HEIGHT
            + BODY_PADDING
            + THUMBNAIL_SIZE
            + BODY_PADDING
    };

    let mut x = 0.0f32;
    for nodes_in_column in &columns {
        let mut y = 0.0f32;
        for &id in nodes_in_column {
            let _ = graph.set_node_position(id, [x, y]);
            let height = graph
                .node(id)
                .map(|node| node_height(&node.kind))
                .unwrap_or(0.0);
            y += height + ROW_GAP;
        }
        x += NODE_WIDTH + COLUMN_GAP;
    }
}

/// Saves `graph` to `examples/graphs/{name}.mcgraph`, reloads it (the real
/// file round trip, not just in-memory), evaluates it, and writes its
/// Output: Base Color to `{out_dir}/{name}_base_color.png` — the common
/// "example graph" lifecycle every graph after `bricks` follows (`bricks`
/// keeps its own hand-written version above since it additionally exercises
/// the full multi-channel export pipeline, not just a quick-look).
fn save_evaluate_and_render_base_color(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    name: &str,
    mut graph: mc_core::Graph,
    out_dir: &str,
) -> anyhow::Result<()> {
    auto_layout_for_save(&mut graph);
    let graph_path = format!("examples/graphs/{name}.mcgraph");
    graph.save_to_file(&graph_path)?;
    let mut graph = mc_core::Graph::load_from_file(&graph_path)
        .with_context(|| format!("failed to load {graph_path}"))?;
    mc_nodes::registry::evaluate(gpu, pool, &mut graph)
        .with_context(|| format!("failed to evaluate {name}"))?;

    let resolution = graph.settings.resolution;
    let node_id = graph
        .nodes()
        .find(|(_, node)| node.kind == "output_base_color")
        .map(|(id, _)| id)
        .with_context(|| format!("{name} graph has no output_base_color node"))?;
    let key = mc_nodes::registry::resolve_texture_key(&graph, node_id)
        .with_context(|| format!("{name}'s Output: Base Color input isn't connected"))?;
    let texture = pool
        .texture(&key)
        .context("evaluated output texture missing from the pool")?;
    let pixels = gpu.read_rgba8(texture, resolution);
    let image = image::RgbaImage::from_raw(resolution, resolution, pixels)
        .context("pixel buffer size didn't match the image dimensions")?;
    let image_path = format!("{out_dir}/{name}_base_color.png");
    image.save(&image_path)?;
    println!("Wrote {image_path}");
    println!("Wrote {graph_path}");
    Ok(())
}

/// `mc_cli render <graph.mcgraph> --preset <blender|unreal>` (CLAUDE.md §8):
/// loads a single saved graph file and exports it, instead of the fixed
/// example-graph set `regenerate_examples` always rebuilds. `--res` overrides
/// the graph's own saved resolution (left alone if omitted); `--out`
/// defaults to `out/`, the same convention `regenerate_examples` uses.
///
/// `--variations N` renders N independent re-rolls of every node's own Seed
/// param in one command — for previewing how a graph's jitter/seed-driven
/// look varies without re-running the CLI by hand for each one. It
/// deliberately only touches Seed params (not every slider the way the
/// app's own "randomize all params" dice-button feature does, see
/// `mc_app::params_panel::randomize_all_params`'s own doc comment for that
/// one) — the point here is previewing the SAME graph's own built-in
/// variation, not generating different-looking graphs. Each node's new seed
/// is derived from its *current* seed plus the variation index via
/// SplitMix64 (`mc_app::params_panel::splitmix64_next`'s same algorithm,
/// reimplemented here rather than shared — `mc_app` isn't a dependency
/// `mc_cli` has, or should gain just for this), so re-running the exact same
/// command always reproduces the exact same N variations. Variation 0 is
/// always the graph exactly as saved (no re-roll at all), so the first
/// output in a `--variations` run matches what a plain `render` would have
/// produced. Output file names get a `_varN` suffix (via `graph_name`'s own
/// naming-template substitution — no special variation-aware code needed in
/// `mc_export` at all) whenever more than one variation is requested; with
/// no `--variations` (or `--variations 1`), file names are unchanged.
#[derive(clap::Args)]
struct RenderArgs {
    /// Path to a saved .mcgraph file.
    graph: std::path::PathBuf,
    /// Export preset — sets which output files get written and how
    /// they're named (CLAUDE.md §8).
    #[arg(long, value_enum)]
    preset: PresetArg,
    /// Overrides the graph's own saved resolution.
    #[arg(long)]
    res: Option<u32>,
    /// Output directory.
    #[arg(long, default_value = "out")]
    out: std::path::PathBuf,
    /// Renders this many independent re-rolls of every node's own Seed
    /// param, instead of just the graph as saved.
    #[arg(long)]
    variations: Option<u32>,
}

#[derive(Clone, Copy, ValueEnum)]
enum PresetArg {
    Blender,
    Unreal,
}

impl From<PresetArg> for mc_export::naming::Preset {
    fn from(preset: PresetArg) -> Self {
        match preset {
            PresetArg::Blender => mc_export::naming::Preset::Blender,
            PresetArg::Unreal => mc_export::naming::Preset::Unreal,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Render a single saved graph file (CLAUDE.md §8), instead of
    /// rebuilding the fixed example-graph set.
    Render(RenderArgs),
}

#[derive(Parser)]
#[command(name = "mc_cli", about = "MaterialCreator headless renderer")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

/// SplitMix64, one step — see `mc_app::params_panel::splitmix64_next`'s own
/// doc comment; duplicated rather than shared since `mc_cli` doesn't
/// otherwise depend on `mc_app`.
fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Every node param currently holding a Seed value, as (node id, param
/// name) pairs — what `--variations` re-rolls each round.
fn seed_params(graph: &mc_core::Graph) -> Vec<(mc_core::NodeId, String)> {
    graph
        .nodes()
        .flat_map(|(id, node)| {
            node.params
                .iter()
                .filter(|(_, value)| matches!(value, mc_core::ParamValue::Seed(_)))
                .map(move |(name, _)| (id, name.clone()))
        })
        .collect()
}

fn render_command(args: RenderArgs) -> anyhow::Result<()> {
    let mut graph = mc_core::Graph::load_from_file(&args.graph)
        .with_context(|| format!("failed to load {}", args.graph.display()))?;
    if let Some(res) = args.res {
        graph.settings.resolution = res;
    }

    let graph_name = args
        .graph
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("material");
    let variations = args.variations.unwrap_or(1).max(1);
    let seeds = seed_params(&graph);

    let mut gpu = mc_gpu::Context::new_headless()?;
    let mut pool = mc_gpu::TexturePool::default();

    for variation in 0..variations {
        if variation > 0 {
            for (id, name) in &seeds {
                let Some(&mc_core::ParamValue::Seed(current)) =
                    graph.node(*id).unwrap().params.get(name)
                else {
                    continue;
                };
                let mut state =
                    (current as u64) ^ (variation as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let new_seed = splitmix64_next(&mut state) as u32;
                graph.set_param(*id, name, mc_core::ParamValue::Seed(new_seed))?;
            }
        }
        let name = if variations > 1 {
            format!("{graph_name}_var{variation}")
        } else {
            graph_name.to_string()
        };
        let resolution = graph.settings.resolution;
        let written = mc_nodes::export::export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            args.preset.into(),
            resolution,
            &name,
            &args.out,
            &mc_nodes::export::ExportOptions::default(),
        )
        .with_context(|| format!("failed to export {name}"))?;
        for path in &written {
            println!("Wrote {}", path.display());
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if let Some(Command::Render(args)) = cli.command {
        return render_command(args);
    }

    let out_dir = "out";
    std::fs::create_dir_all(out_dir)?;

    let graph_path = "examples/graphs/bricks.mcgraph";
    let mut bricks_graph = build_bricks_graph();
    auto_layout_for_save(&mut bricks_graph);
    bricks_graph.save_to_file(graph_path)?;
    let mut graph = mc_core::Graph::load_from_file(graph_path)
        .with_context(|| format!("failed to load {graph_path}"))?;

    let mut gpu = mc_gpu::Context::new_headless()?;
    let mut pool = mc_gpu::TexturePool::default();
    mc_nodes::registry::evaluate(&mut gpu, &mut pool, &mut graph)
        .context("failed to evaluate graph")?;

    let resolution = graph.settings.resolution;
    let quick_looks = [
        ("output_base_color", "bricks_base_color.png"),
        ("output_normal", "bricks_normal.png"),
        ("output_height", "bricks_height.png"),
    ];

    for (kind, file_name) in quick_looks {
        let node_id = graph
            .nodes()
            .find(|(_, node)| node.kind == kind)
            .map(|(id, _)| id)
            .with_context(|| format!("bricks graph has no {kind} node"))?;
        let texture_key = mc_nodes::registry::resolve_texture_key(&graph, node_id)
            .with_context(|| format!("{kind}'s input isn't connected"))?;
        let texture = pool
            .texture(&texture_key)
            .context("evaluated output texture missing from the pool")?;

        let pixels = gpu.read_rgba8(texture, resolution);
        let image = image::RgbaImage::from_raw(resolution, resolution, pixels)
            .context("pixel buffer size didn't match the image dimensions")?;
        let image_path = format!("{out_dir}/{file_name}");
        image.save(&image_path)?;
        println!("Wrote {image_path}");
    }
    println!("Wrote {graph_path}");

    // Full material export, both presets, at a resolution different from
    // the graph's own live-preview resolution (512) to prove that path.
    let export_resolution = 1024;
    for preset in [
        mc_export::naming::Preset::Blender,
        mc_export::naming::Preset::Unreal,
    ] {
        // A filesystem-safe directory slug, independent of `preset.name()`'s
        // own UI-facing display string (which can contain spaces, e.g.
        // "Unreal Engine").
        let slug = match preset {
            mc_export::naming::Preset::Blender => "blender",
            mc_export::naming::Preset::Unreal => "unreal",
        };
        let export_dir = format!("{out_dir}/export_{slug}");
        // Exercise the full optional-output surface for Unreal so this
        // smoke test actually covers it, not just the required outputs.
        let options = match preset {
            mc_export::naming::Preset::Unreal => mc_nodes::export::ExportOptions {
                opacity: mc_nodes::export::OpacityMode::SeparateFile,
                ..mc_nodes::export::ExportOptions::default()
            },
            mc_export::naming::Preset::Blender => mc_nodes::export::ExportOptions::default(),
        };
        let written = mc_nodes::export::export_material(
            &mut gpu,
            &mut pool,
            &mut graph,
            preset,
            export_resolution,
            "bricks",
            std::path::Path::new(&export_dir),
            &options,
        )
        .with_context(|| format!("failed to export {preset:?} preset"))?;
        for path in &written {
            println!("Wrote {}", path.display());
        }
    }

    save_evaluate_and_render_base_color(
        &mut gpu,
        &mut pool,
        "glitch_dither_demo",
        build_glitch_dither_demo_graph(),
        out_dir,
    )?;

    for (name, graph) in [
        ("rock", build_rock_graph()),
        ("brushed_metal", build_brushed_metal_graph()),
        ("fabric", build_fabric_graph()),
        ("wood", build_wood_graph()),
        ("hardsurface_demo", build_hardsurface_demo_graph()),
    ] {
        save_evaluate_and_render_base_color(&mut gpu, &mut pool, name, graph, out_dir)?;
    }

    Ok(())
}
