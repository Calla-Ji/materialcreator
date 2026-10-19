// SPDX-License-Identifier: MIT OR Apache-2.0
//! Precision regression test for the Hard-Surface Nodes demo chain
//! (`docs/hardsurface_nodes/HARDSURFACE_NODES.md` §4/§9): compares the
//! example graph's `Output: Height` against the hand-off spec's own
//! `demo_height.png`, within the spec's own stated tolerance (§6: mean
//! abs under 0.004). This is a *precision* check against an external
//! ground truth, not `golden.rs`'s own "did this project's self-
//! referential 8-bit render visibly change" cross-GPU tolerance style —
//! kept in a separate file for that reason.

/// Loads a 16-bit grayscale PNG as `0..1` `f32`s — same minimal reader
/// `mc_nodes::test_support::read_png16_as_unit_floats` uses, duplicated
/// here since that module is `#[cfg(test)]`-only and not visible from
/// another crate's own test binary (`golden.rs` already inlines its own
/// PNG loading for the same reason).
fn read_png16_as_unit_floats(path: &str) -> (Vec<f32>, u32) {
    let img = image::open(path)
        .unwrap_or_else(|err| panic!("failed to open {path}: {err}"))
        .to_luma16();
    let resolution = img.width();
    assert_eq!(img.height(), resolution, "{path}: not square");
    let values = img
        .into_raw()
        .into_iter()
        .map(|v| f32::from(v) / 65535.0)
        .collect();
    (values, resolution)
}

#[test]
fn hardsurface_demo_height_matches_the_reference_golden() {
    let graph_path = "../../examples/graphs/hardsurface_demo.mcgraph";
    let mut graph = mc_core::Graph::load_from_file(graph_path)
        .unwrap_or_else(|err| panic!("failed to load {graph_path}: {err}"));

    let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
    let mut pool = mc_gpu::TexturePool::default();
    mc_nodes::registry::evaluate(&mut gpu, &mut pool, &mut graph, false)
        .unwrap_or_else(|err| panic!("failed to evaluate {graph_path}: {err}"));

    let resolution = graph.settings.resolution;
    let node_id = graph
        .nodes()
        .find(|(_, node)| node.kind == "output_height")
        .map(|(id, _)| id)
        .unwrap_or_else(|| panic!("{graph_path} has no output_height node"));
    let key = mc_nodes::registry::resolve_texture_key(&graph, node_id)
        .unwrap_or_else(|| panic!("{graph_path}'s Output: Height input isn't connected"));
    let texture = pool
        .texture(&key)
        .expect("evaluated output texture missing from the pool");

    // Full float precision — `Output` nodes are pure pass-throughs with
    // no texture of their own, so this reads Edge Wear's real `height`
    // output directly, no lossy 8-bit round trip.
    let got: Vec<f32> = gpu
        .read_rgba16f(texture, resolution)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| p[0])
        .collect();

    let (want, want_res) =
        read_png16_as_unit_floats("../../tests/golden/hardsurface/demo_height.png");
    assert_eq!(
        resolution, want_res,
        "demo_height.png is a different resolution than the graph currently renders at"
    );

    let mean: f64 = got
        .iter()
        .zip(want.iter())
        .map(|(&g, &w)| f64::from((g - w).abs()))
        .sum::<f64>()
        / got.len() as f64;
    assert!(
        mean < 0.004,
        "hardsurface_demo's Height output differs from demo_height.png by a mean of {mean} \
         (tolerance 0.004)"
    );
}
