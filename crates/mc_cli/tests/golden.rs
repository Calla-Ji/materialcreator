// SPDX-License-Identifier: MIT OR Apache-2.0
//! Golden-image regression tests (CLAUDE.md §10's M6 item): loads an example
//! graph from disk, evaluates it exactly the way `mc_cli` does, and compares
//! its Base Color output against a checked-in reference PNG in
//! `tests/golden/`.
//!
//! These guard against an *unintentional* visual regression in an example
//! graph — a node's math changing in a way nobody meant to change how
//! `bricks.mcgraph` looks, say. They are **not** a substitute for a node's
//! own tiling/determinism/behavior tests (CLAUDE.md §6.7): those catch a
//! specific node breaking; these catch a whole graph's rendered look
//! drifting, from any cause.
//!
//! A small per-channel tolerance (not bit-exact equality) is deliberate:
//! CLAUDE.md §6.3's determinism guarantee is same-GPU only ("same graph +
//! same seed = bit-identical output on the *same* GPU"), so a reference
//! image checked in from one machine could differ by a texel or two of
//! floating-point noise on another GPU/driver without any real regression
//! having happened.
//!
//! To add or intentionally update a reference image, run:
//! `MC_UPDATE_GOLDEN=1 cargo test -p mc_cli --test golden`

use std::path::Path;

/// Max allowed per-channel 8-bit difference at any single pixel. Loose
/// enough to absorb cross-GPU floating-point noise, tight enough to still
/// catch a real visual change (an actual different color, a shifted
/// pattern, a missing node's worth of effect).
const TOLERANCE: u8 = 6;

fn render_base_color(graph_path: &str) -> (Vec<u8>, u32) {
    let mut graph = mc_core::Graph::load_from_file(graph_path)
        .unwrap_or_else(|err| panic!("failed to load {graph_path}: {err}"));

    let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
    let mut pool = mc_gpu::TexturePool::default();
    mc_nodes::registry::evaluate(&mut gpu, &mut pool, &mut graph)
        .unwrap_or_else(|err| panic!("failed to evaluate {graph_path}: {err}"));

    let resolution = graph.settings.resolution;
    let node_id = graph
        .nodes()
        .find(|(_, node)| node.kind == "output_base_color")
        .map(|(id, _)| id)
        .unwrap_or_else(|| panic!("{graph_path} has no output_base_color node"));
    let key = mc_nodes::registry::resolve_texture_key(&graph, node_id)
        .unwrap_or_else(|| panic!("{graph_path}'s Output: Base Color input isn't connected"));
    let texture = pool
        .texture(&key)
        .expect("evaluated output texture missing from the pool");

    (gpu.read_rgba8(texture, resolution), resolution)
}

/// Compares `pixels` against `golden_path`, or (with `MC_UPDATE_GOLDEN=1` set)
/// writes `pixels` there instead of comparing. `tolerance` overrides
/// `TOLERANCE` — see `glitch_dither_demo_matches_golden` for why one graph
/// needs a much looser one.
fn assert_matches_golden_with_tolerance(
    pixels: &[u8],
    resolution: u32,
    golden_path: &str,
    tolerance: u8,
) {
    if std::env::var_os("MC_UPDATE_GOLDEN").is_some() {
        let image = image::RgbaImage::from_raw(resolution, resolution, pixels.to_vec())
            .expect("pixel buffer size matches the image dimensions");
        image
            .save(golden_path)
            .unwrap_or_else(|err| panic!("failed to write {golden_path}: {err}"));
        return;
    }

    assert!(
        Path::new(golden_path).exists(),
        "missing golden image {golden_path} — run `MC_UPDATE_GOLDEN=1 cargo test -p mc_cli \
         --test golden` to create it"
    );
    let golden = image::open(golden_path)
        .unwrap_or_else(|err| panic!("failed to read {golden_path}: {err}"))
        .to_rgba8();
    assert_eq!(
        golden.dimensions(),
        (resolution, resolution),
        "{golden_path} is a different resolution than the graph currently renders at"
    );

    let golden_pixels = golden.into_raw();
    let max_diff = pixels
        .iter()
        .zip(golden_pixels.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    if max_diff > tolerance {
        // One-time diagnostic dump (not committed, not read by anything else)
        // so a cross-platform CI failure's actual render can be pulled down
        // and looked at, instead of guessing at a tolerance bump blind.
        if let Some(dir) = std::env::var_os("MC_GOLDEN_FAILURE_DIR") {
            let dir = Path::new(&dir);
            let _ = std::fs::create_dir_all(dir);
            let name = Path::new(golden_path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "unknown.png".to_string());
            if let Some(actual) =
                image::RgbaImage::from_raw(resolution, resolution, pixels.to_vec())
            {
                let _ = actual.save(dir.join(format!("actual_{name}")));
            }
        }
    }
    assert!(
        max_diff <= tolerance,
        "{golden_path}: rendered output differs from the golden image by up to {max_diff} \
         (tolerance {tolerance}) — if this is an intentional change to the graph, re-run with \
         MC_UPDATE_GOLDEN=1 to update the reference"
    );
}

/// Compares against `TOLERANCE` — see `assert_matches_golden_with_tolerance`
/// for a graph that needs a different one.
fn assert_matches_golden(pixels: &[u8], resolution: u32, golden_path: &str) {
    assert_matches_golden_with_tolerance(pixels, resolution, golden_path, TOLERANCE);
}

#[test]
fn bricks_matches_golden() {
    let (pixels, resolution) = render_base_color("../../examples/graphs/bricks.mcgraph");
    assert_matches_golden(
        &pixels,
        resolution,
        "../../tests/golden/bricks_base_color.png",
    );
}

#[test]
fn glitch_dither_demo_matches_golden() {
    // This graph (noise -> Gradient Map -> Block Shift -> Channel Split ->
    // Ordered Dither) needs a much looser tolerance than every other
    // example: dithering's hard quantization thresholds take a tiny
    // cross-GPU floating-point difference upstream (the kind every other
    // graph here absorbs within the normal TOLERANCE) and amplify it into a
    // large per-pixel delta — a dithered pixel landing on one side of a
    // threshold instead of the other reads as a big numeric jump even
    // though the two results are visually indistinguishable. Confirmed by
    // actually downloading and looking at the Linux/Windows CI renders that
    // failed at the normal tolerance (via MC_GOLDEN_FAILURE_DIR, see
    // test-build.yml) side by side with this golden: same dither pattern,
    // same macro structure, no visible difference, 100% confirmed not a
    // real regression — not just assumed from the "cross-GPU noise" theory
    // every other graph here already accounts for. Re-confirmed the same way
    // (rendered + looked at side by side, still indistinguishable) when the
    // node pipeline moved from `Rgba8Unorm` to `Rgba16Float` — that migration
    // nudges every upstream node's exact float value by the same kind of
    // tiny amount cross-GPU noise does, so it trips this same amplifier.
    // Re-confirmed again on GitHub's `ubuntu-latest` (Lavapipe/llvmpipe
    // software Vulkan — max diff 114, briefly over the old tolerance of
    // 100): pulled the actual render via `MC_GOLDEN_FAILURE_DIR` and
    // compared side by side — same dither pattern, same macro structure,
    // no visible difference. Bumped to 130 for headroom.
    let (pixels, resolution) =
        render_base_color("../../examples/graphs/glitch_dither_demo.mcgraph");
    assert_matches_golden_with_tolerance(
        &pixels,
        resolution,
        "../../tests/golden/glitch_dither_demo_base_color.png",
        130,
    );
}

#[test]
fn rock_matches_golden() {
    let (pixels, resolution) = render_base_color("../../examples/graphs/rock.mcgraph");
    assert_matches_golden(
        &pixels,
        resolution,
        "../../tests/golden/rock_base_color.png",
    );
}

#[test]
fn brushed_metal_matches_golden() {
    let (pixels, resolution) = render_base_color("../../examples/graphs/brushed_metal.mcgraph");
    assert_matches_golden(
        &pixels,
        resolution,
        "../../tests/golden/brushed_metal_base_color.png",
    );
}

#[test]
fn fabric_matches_golden() {
    let (pixels, resolution) = render_base_color("../../examples/graphs/fabric.mcgraph");
    assert_matches_golden(
        &pixels,
        resolution,
        "../../tests/golden/fabric_base_color.png",
    );
}

#[test]
fn wooden_planks_matches_golden() {
    // Needs a somewhat looser tolerance than most examples: this graph's
    // grain/whorl layers lean heavily on noise and Warp, which (like
    // `glitch_dither_demo` below, just to a much smaller degree) amplify a
    // tiny cross-GPU floating-point difference upstream more than a
    // mostly-flat-color graph like `bricks` would. Confirmed by actually
    // failing CI (macOS max diff 7, Windows max diff 11, both just over the
    // normal tolerance of 6) and downloading the real renders via
    // `MC_GOLDEN_FAILURE_DIR` to compare side by side with this golden —
    // same plank layout, same grain, no visible difference, same "pulled
    // the actual failing render and looked at it" methodology
    // `glitch_dither_demo_matches_golden` already established, not assumed
    // from the "cross-GPU noise" theory alone.
    let (pixels, resolution) = render_base_color("../../examples/graphs/wooden_planks.mcgraph");
    assert_matches_golden_with_tolerance(
        &pixels,
        resolution,
        "../../tests/golden/wooden_planks_base_color.png",
        20,
    );
}
