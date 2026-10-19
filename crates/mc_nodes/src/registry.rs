// SPDX-License-Identifier: MIT OR Apache-2.0
//! Bridges mc_core's generic graph model to actual GPU node dispatch.
//!
//! Given a [`mc_core::Graph`], [`evaluate`] walks it in topological order,
//! skips nodes the graph says are already up to date, and dispatches each
//! remaining node's GPU work by its `kind`. Each evaluated node's texture
//! lives in the [`mc_gpu::TexturePool`] under the key from [`texture_key`].

use std::collections::HashSet;

use crate::{
    anisotropic_noise, bevel, bevel_profile, blend, block_shift, blur, channel_pack, channel_split,
    checker, circular_pattern, clamp, clouds, color_match, curvature_from_normal, curve,
    data_corrupt, directional_blur, directional_warp, distance_field, distance_spread,
    dither_pattern, edge_detect, edge_wear, fabric_generator, fasteners, fractal_sum, french_knot,
    gabor_noise, gradient_linear, gradient_map, gradient_radial, grayscale_conversion,
    height_to_ao, height_to_normal, histogram_scan, hsl_adjust, image_import, image_trace, invert,
    knit, levels, material_preset, ordered_dither, palette_dither, panel_layout, path,
    path_simplify, path_to_mask, path_to_sdf, perlin, redistance, scanline_jitter,
    scattered_streaks, shape, shape_boolean, simplex_noise, slope_blur, stitch, thread_shape,
    threshold, tile_generator, transform2d, type_text, uniform_color, voronoi, warp, white_noise,
};

/// The default output port name every single-output `NodeDef` uses.
pub const DEFAULT_OUTPUT_PORT: &str = "output";

/// The texture pool key a node's output `port` is stored under. Every
/// single-output node (the overwhelming majority) always passes
/// [`DEFAULT_OUTPUT_PORT`] here; a multi-output node (Thread, Knit) stores
/// each of its named outputs under its own key.
pub fn texture_key(id: mc_core::NodeId, port: &str) -> String {
    format!("node_{}_{}", id.0, port)
}

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error(transparent)]
    Graph(#[from] mc_core::GraphError),
    #[error("node {0:?} has kind {1:?}, which mc_nodes doesn't know how to evaluate")]
    UnknownKind(mc_core::NodeId, String),
    #[error("node {0:?}'s required input {1:?} isn't connected")]
    MissingInput(mc_core::NodeId, String),
}

/// Evaluate every node in `graph` that needs it, in dependency order.
///
/// Output nodes (`kind` starting with `"output_"`) are pass-through markers
/// with no shader of their own — see [`resolve_texture_key`] to find what
/// they actually point at.
///
/// A node with an unconnected required input (completely normal mid-edit —
/// e.g. a just-added filter node nobody's wired up yet) does *not* abort the
/// whole pass: it, and anything downstream of it, are skipped and left
/// dirty for next time, but every other, independent part of the graph
/// still evaluates and gets its texture this frame. Without that, one loose
/// node anywhere in the graph used to silently freeze every node after it
/// in topological order — including already-fully-wired, already-working
/// ones — since the first `?` on a missing input aborted the entire
/// function. The first error encountered is still returned (after the full
/// pass completes) so callers like `mc_cli`, which genuinely want a hard
/// failure for a broken graph file, still get one.
pub fn evaluate(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    graph: &mut mc_core::Graph,
) -> Result<(), EvalError> {
    let order = graph.topological_order()?;
    let resolution = graph.settings.resolution;

    let mut failed: HashSet<mc_core::NodeId> = HashSet::new();
    let mut first_error: Option<EvalError> = None;

    for id in order {
        if !graph.needs_evaluation(id) {
            continue;
        }

        let node = graph
            .node(id)
            .expect("node from topological_order always exists");
        if node.kind.starts_with("output_") {
            graph.mark_evaluated(id);
            continue;
        }

        // A required input: missing entirely, or wired to a node that
        // itself failed to evaluate this pass, are the same "can't run this
        // node right now" condition — either way there's no real texture to
        // dispatch with. Returns the texture key for the *specific upstream
        // output port* this input is wired to, not just "the" upstream
        // node's texture — needed now that a node (Thread) can have more
        // than one named output.
        let resolve_input = |port: &str| -> Result<String, EvalError> {
            let (input_id, from_port) = graph
                .upstream_from(id, port)
                .ok_or_else(|| EvalError::MissingInput(id, port.to_string()))?;
            if failed.contains(&input_id) {
                return Err(EvalError::MissingInput(id, port.to_string()));
            }
            Ok(texture_key(input_id, &from_port))
        };
        // An optional input (e.g. Blend's mask): absent is a supported
        // state on its own, so a failed upstream degrades to "absent"
        // rather than failing this node too.
        let resolve_optional_input = |port: &str| -> Option<String> {
            let (input_id, from_port) = graph.upstream_from(id, port)?;
            if failed.contains(&input_id) {
                return None;
            }
            Some(texture_key(input_id, &from_port))
        };

        let key = texture_key(id, DEFAULT_OUTPUT_PORT);
        // Set by the "image_trace"/"path_simplify" arms below, written to
        // the graph only *after* the closure returns — `node` (borrowed
        // from `graph`) is still alive for the closure's whole body
        // (every other arm reads it too), so `graph.set_computed_param`
        // can't be called from inside it without a borrow conflict. A
        // local variable the closure can freely assign into, read back
        // out here once it's done, sidesteps that cleanly.
        let mut computed_path_result: Option<mc_core::Path> = None;
        let result: Result<(), EvalError> = (|| {
            match node.kind.as_str() {
                "perlin_noise" => {
                    let params = perlin::PerlinParams::from_graph(&node.params);
                    perlin::dispatch(gpu, pool, &key, resolution, &params);
                }
                "anisotropic_noise" => {
                    let params =
                        anisotropic_noise::AnisotropicNoiseParams::from_graph(&node.params);
                    anisotropic_noise::dispatch(gpu, pool, &key, resolution, &params);
                }
                "levels" => {
                    let input_key = resolve_input("input")?;
                    let params = levels::LevelsParams::from_graph(&node.params);
                    levels::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "channel_pack" => {
                    let r_key = resolve_optional_input("r");
                    let g_key = resolve_optional_input("g");
                    let b_key = resolve_optional_input("b");
                    channel_pack::dispatch(
                        gpu,
                        pool,
                        r_key.as_deref(),
                        g_key.as_deref(),
                        b_key.as_deref(),
                        &key,
                        resolution,
                    );
                }
                "blend" => {
                    let base_key = resolve_input("base")?;
                    let blend_key = resolve_input("blend")?;
                    let mask_key = resolve_optional_input("mask");
                    let params = blend::BlendParams::from_graph(&node.params);
                    blend::dispatch(
                        gpu,
                        pool,
                        &base_key,
                        &blend_key,
                        mask_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "color_match" => {
                    let input_key = resolve_input("input")?;
                    let reference_key = resolve_input("reference")?;
                    let params = color_match::ColorMatchParams::from_graph(&node.params);
                    color_match::dispatch(
                        gpu,
                        pool,
                        &input_key,
                        &reference_key,
                        &key,
                        resolution,
                        &params,
                    );
                }
                "material_preset" => {
                    let base_color_key = resolve_optional_input("base_color");
                    let roughness_key = resolve_optional_input("roughness");
                    let metallic_key = resolve_optional_input("metallic");
                    let base_color_out_key = texture_key(id, "base_color");
                    let roughness_out_key = texture_key(id, "roughness");
                    let metallic_out_key = texture_key(id, "metallic");
                    let params = material_preset::MaterialPresetParams::from_graph(&node.params);
                    material_preset::dispatch(
                        gpu,
                        pool,
                        base_color_key.as_deref(),
                        roughness_key.as_deref(),
                        metallic_key.as_deref(),
                        &base_color_out_key,
                        &roughness_out_key,
                        &metallic_out_key,
                        resolution,
                        &params,
                    );
                }
                "voronoi" => {
                    let params = voronoi::VoronoiParams::from_graph(&node.params);
                    voronoi::dispatch(gpu, pool, &key, resolution, &params);
                }
                "fractal_sum" => {
                    let params = fractal_sum::FractalSumParams::from_graph(&node.params);
                    fractal_sum::dispatch(gpu, pool, &key, resolution, &params);
                }
                "tile_generator" => {
                    let shape_key = resolve_optional_input("shape");
                    let params = tile_generator::TileGeneratorParams::from_graph(&node.params);
                    tile_generator::dispatch(
                        gpu,
                        pool,
                        shape_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "fabric_generator" => {
                    let fiber_key = resolve_optional_input("fiber");
                    let fiber_ply_id_key = resolve_optional_input("fiber_ply_id");
                    let ply_id_key = texture_key(id, "ply_id");
                    let params = fabric_generator::FabricGeneratorParams::from_graph(&node.params);
                    fabric_generator::dispatch(
                        gpu,
                        pool,
                        fiber_key.as_deref(),
                        fiber_ply_id_key.as_deref(),
                        fabric_generator::FabricGeneratorOutputKeys {
                            height: &key,
                            ply_id: &ply_id_key,
                        },
                        resolution,
                        &params,
                    );
                }
                "thread_shape" => {
                    let height_key = texture_key(id, "height");
                    let ply_id_key = texture_key(id, "ply_id");
                    let fiber_key = texture_key(id, "fiber");
                    let mask_key = texture_key(id, "mask");
                    let params = thread_shape::ThreadShapeParams::from_graph(&node.params);
                    thread_shape::dispatch(
                        gpu,
                        pool,
                        thread_shape::ThreadOutputKeys {
                            height: &height_key,
                            ply_id: &ply_id_key,
                            fiber: &fiber_key,
                            mask: &mask_key,
                        },
                        resolution,
                        &params,
                    );
                }
                "gabor_noise" => {
                    let value_key = texture_key(id, "value");
                    let phasor_key = texture_key(id, "phasor");
                    let phase_key = texture_key(id, "phase");
                    let intensity_key = texture_key(id, "intensity");
                    let raw_key = texture_key(id, "raw");
                    let params = gabor_noise::GaborParams::from_graph(&node.params, resolution);
                    gabor_noise::dispatch(
                        gpu,
                        pool,
                        gabor_noise::GaborOutputKeys {
                            value: &value_key,
                            phasor: &phasor_key,
                            phase: &phase_key,
                            intensity: &intensity_key,
                            raw: &raw_key,
                        },
                        resolution,
                        &params,
                    );
                }
                "scattered_streaks" => {
                    let angle_map_key = resolve_optional_input("angle_map");
                    let height_key = texture_key(id, "height");
                    let mask_key = texture_key(id, "mask");
                    let streak_id_key = texture_key(id, "streak_id");
                    let along_key = texture_key(id, "along");
                    let direction_key = texture_key(id, "direction");
                    let params =
                        scattered_streaks::StreakParams::from_graph(&node.params, resolution);
                    scattered_streaks::dispatch(
                        gpu,
                        pool,
                        scattered_streaks::StreakOutputKeys {
                            height: &height_key,
                            mask: &mask_key,
                            streak_id: &streak_id_key,
                            along: &along_key,
                            direction: &direction_key,
                        },
                        angle_map_key.as_deref(),
                        resolution,
                        &params,
                    );
                }
                "path" => {
                    // The "path" output isn't a texture at all (see
                    // mc_core::PortType::Path's own doc comment) — nothing
                    // to dispatch for it here. `resolve_effective_path`
                    // folds in the Reference input's own fallback (this
                    // node's own data if it has any points, else
                    // whatever's connected there) so Preview Mask and the
                    // "path" output a downstream Stitch node reads agree.
                    let preview_mask_key = texture_key(id, "preview_mask");
                    let path_value = path::resolve_effective_path(graph, id);
                    let path_params = path::PathParams::from_graph(&node.params);
                    let stamp_key = resolve_optional_input("stamp");

                    match stamp_key {
                        None => {
                            if path_params.show_curve {
                                path::dispatch_preview_mask(
                                    gpu,
                                    pool,
                                    &preview_mask_key,
                                    resolution,
                                    &path_value,
                                );
                            } else {
                                gpu.fill_rgba8(pool, &preview_mask_key, resolution, [0, 0, 0, 255]);
                            }
                        }
                        Some(stamp_key) => {
                            // Stamps composite over a separate scratch
                            // texture, not `preview_mask_key` directly —
                            // the stamp shader reads its own "base" while
                            // writing the real output, so they can't be
                            // the same key.
                            let outline_key = format!("{preview_mask_key}__outline_scratch");
                            if path_params.show_curve {
                                path::dispatch_preview_mask(
                                    gpu,
                                    pool,
                                    &outline_key,
                                    resolution,
                                    &path_value,
                                );
                            } else {
                                gpu.fill_rgba8(pool, &outline_key, resolution, [0, 0, 0, 255]);
                            }
                            let instances = path::stamp_instances(
                                &path_value,
                                path_params.stamp_spacing as f64,
                                path_params.stamp_align_to_curve,
                                path_params.stamp_rotation_deg.to_radians(),
                            );
                            path::dispatch_stamp(
                                gpu,
                                pool,
                                &stamp_key,
                                &outline_key,
                                &preview_mask_key,
                                resolution,
                                &instances,
                                path_params.stamp_size,
                                path_value.wrap,
                            );
                        }
                    }
                }
                "knit" => {
                    let height_key = texture_key(id, "height");
                    let mask_key = texture_key(id, "mask");
                    let ply_id_key = texture_key(id, "ply_id");
                    let stitch_id_key = texture_key(id, "stitch_id");
                    let stitch_type_key = texture_key(id, "stitch_type");
                    let params = knit::KnitParams::from_graph(&node.params);
                    knit::dispatch(
                        gpu,
                        pool,
                        knit::KnitOutputKeys {
                            height: &height_key,
                            mask: &mask_key,
                            ply_id: &ply_id_key,
                            stitch_id: &stitch_id_key,
                            stitch_type: &stitch_type_key,
                        },
                        resolution,
                        &params,
                    );
                }
                "shape" => {
                    let params = shape::ShapeParams::from_graph(&node.params);
                    shape::dispatch(gpu, pool, &key, resolution, &params);
                }
                "circular_pattern" => {
                    let params = circular_pattern::CircularPatternParams::from_graph(&node.params);
                    match resolve_optional_input("stamp") {
                        Some(stamp_key) => {
                            circular_pattern::dispatch(
                                gpu, pool, &stamp_key, &key, resolution, &params,
                            );
                        }
                        // No real fallback, unlike Path's own un-stamped
                        // Preview Mask (which at least still shows the
                        // curve outline) — there's nothing of this node's
                        // own to show without a Stamp input, so just blank.
                        None => gpu.fill_rgba8(pool, &key, resolution, [0, 0, 0, 255]),
                    }
                }
                "type_text" => {
                    let params = type_text::TypeTextParams::from_graph(&node.params);
                    type_text::dispatch(gpu, pool, &key, resolution, &params);
                }
                "stitch" => {
                    // French Knot has its own `KnotParams`/`crate::french_knot`
                    // entirely (see `mc_nodes::stitch::StitchKind`'s own doc
                    // comment) — branches here, before either side ever
                    // builds the other model's own params, rather than
                    // inside `stitch::dispatch_render` itself.
                    let is_french_knot = node.params.get("stitch").and_then(|v| match v {
                        mc_core::ParamValue::Enum(s) => Some(s.as_str()),
                        _ => None,
                    }) == Some("french_knot");
                    if is_french_knot {
                        let params = crate::french_knot::KnotParams::from_graph(&node.params);
                        french_knot::dispatch_render(
                            gpu,
                            pool,
                            french_knot::RenderOutputKeys {
                                height: &texture_key(id, "height"),
                                mask: &texture_key(id, "mask"),
                                knot_id: &texture_key(id, "thread_id"),
                                along: &texture_key(id, "order"),
                                strand_id: &texture_key(id, "ply_id"),
                                direction: &texture_key(id, "direction"),
                            },
                            &params,
                            resolution,
                        );
                    } else {
                        let params = crate::stitch::StitchParams::from_graph(&node.params);
                        let local_path = crate::stitch::synthetic_local_path(&params);
                        stitch::dispatch_render(
                            gpu,
                            pool,
                            stitch::RenderOutputKeys {
                                height: &texture_key(id, "height"),
                                mask: &texture_key(id, "mask"),
                                thread_id: &texture_key(id, "thread_id"),
                                order: &texture_key(id, "order"),
                                ply_id: &texture_key(id, "ply_id"),
                                direction: &texture_key(id, "direction"),
                            },
                            &local_path,
                            &params,
                            resolution,
                        );
                    }
                }
                "gradient_linear" => {
                    let params = gradient_linear::GradientLinearParams::from_graph(&node.params);
                    gradient_linear::dispatch(gpu, pool, &key, resolution, &params);
                }
                "gradient_radial" => {
                    let params = gradient_radial::GradientRadialParams::from_graph(&node.params);
                    gradient_radial::dispatch(gpu, pool, &key, resolution, &params);
                }
                "uniform_color" => {
                    let params = uniform_color::UniformColorParams::from_graph(&node.params);
                    uniform_color::dispatch(gpu, pool, &key, resolution, &params);
                }
                "image_import" => {
                    let params = image_import::ImageImportParams::from_graph(&node.params);
                    image_import::dispatch(gpu, pool, &key, resolution, &params);
                }
                "invert" => {
                    let input_key = resolve_input("input")?;
                    invert::dispatch(gpu, pool, &input_key, &key, resolution);
                }
                "histogram_scan" => {
                    let input_key = resolve_input("input")?;
                    let params = histogram_scan::HistogramScanParams::from_graph(&node.params);
                    histogram_scan::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "blur" => {
                    let input_key = resolve_input("input")?;
                    let params = blur::BlurParams::from_graph(&node.params);
                    blur::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "distance_spread" => {
                    let mask_key = resolve_input("mask")?;
                    let source_key = resolve_optional_input("source");
                    let params = distance_spread::DistanceSpreadParams::from_graph(&node.params);
                    distance_spread::dispatch(
                        gpu,
                        pool,
                        &mask_key,
                        source_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "bevel" => {
                    let input_key = resolve_input("input")?;
                    let params = bevel::BevelParams::from_graph(&node.params);
                    bevel::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "warp" => {
                    let input_key = resolve_input("input")?;
                    let intensity_key = resolve_input("intensity")?;
                    let params = warp::WarpParams::from_graph(&node.params);
                    warp::dispatch(
                        gpu,
                        pool,
                        &input_key,
                        &intensity_key,
                        &key,
                        resolution,
                        &params,
                    );
                }
                "transform2d" => {
                    let input_key = resolve_input("input")?;
                    let params = transform2d::Transform2DParams::from_graph(&node.params);
                    transform2d::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "height_to_normal" => {
                    let input_key = resolve_input("input")?;
                    let params = height_to_normal::HeightToNormalParams::from_graph(&node.params);
                    height_to_normal::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "clamp" => {
                    let input_key = resolve_input("input")?;
                    let params = clamp::ClampParams::from_graph(&node.params);
                    clamp::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "threshold" => {
                    let input_key = resolve_input("input")?;
                    let params = threshold::ThresholdParams::from_graph(&node.params);
                    threshold::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "grayscale_conversion" => {
                    let input_key = resolve_input("input")?;
                    grayscale_conversion::dispatch(gpu, pool, &input_key, &key, resolution);
                }
                "checker" => {
                    let params = checker::CheckerParams::from_graph(&node.params);
                    checker::dispatch(gpu, pool, &key, resolution, &params);
                }
                "dither_pattern" => {
                    let params = dither_pattern::DitherPatternParams::from_graph(&node.params);
                    dither_pattern::dispatch(gpu, pool, &key, resolution, &params);
                }
                "ordered_dither" => {
                    let image_key = resolve_input("image")?;
                    let params = ordered_dither::OrderedDitherParams::from_graph(&node.params);
                    ordered_dither::dispatch(gpu, pool, &image_key, &key, resolution, &params);
                }
                "palette_dither" => {
                    let image_key = resolve_input("image")?;
                    let params = palette_dither::PaletteDitherParams::from_graph(&node.params);
                    palette_dither::dispatch(gpu, pool, &image_key, &key, resolution, &params);
                }
                "channel_split" => {
                    let image_key = resolve_input("image")?;
                    let mask_key = resolve_optional_input("mask");
                    let params = channel_split::ChannelSplitParams::from_graph(&node.params);
                    channel_split::dispatch(
                        gpu,
                        pool,
                        &image_key,
                        mask_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "block_shift" => {
                    let image_key = resolve_input("image")?;
                    let mask_key = resolve_optional_input("mask");
                    let params = block_shift::BlockShiftParams::from_graph(&node.params);
                    block_shift::dispatch(
                        gpu,
                        pool,
                        &image_key,
                        mask_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "data_corrupt" => {
                    let image_key = resolve_input("image")?;
                    let mask_key = resolve_optional_input("mask");
                    let params = data_corrupt::DataCorruptParams::from_graph(&node.params);
                    data_corrupt::dispatch(
                        gpu,
                        pool,
                        &image_key,
                        mask_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "scanline_jitter" => {
                    let image_key = resolve_input("image")?;
                    let mask_key = resolve_optional_input("mask");
                    let params = scanline_jitter::ScanlineJitterParams::from_graph(&node.params);
                    scanline_jitter::dispatch(
                        gpu,
                        pool,
                        &image_key,
                        mask_key.as_deref(),
                        &key,
                        resolution,
                        &params,
                    );
                }
                "hsl_adjust" => {
                    let input_key = resolve_input("input")?;
                    let params = hsl_adjust::HslAdjustParams::from_graph(&node.params);
                    hsl_adjust::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "curve" => {
                    let input_key = resolve_input("input")?;
                    let params = curve::CurveParams::from_graph(&node.params);
                    curve::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "gradient_map" => {
                    let input_key = resolve_input("input")?;
                    let params = gradient_map::GradientMapParams::from_graph(&node.params);
                    gradient_map::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "simplex_noise" => {
                    let params = simplex_noise::SimplexParams::from_graph(&node.params);
                    simplex_noise::dispatch(gpu, pool, &key, resolution, &params);
                }
                "white_noise" => {
                    let params = white_noise::WhiteNoiseParams::from_graph(&node.params);
                    white_noise::dispatch(gpu, pool, &key, resolution, &params);
                }
                "clouds" => {
                    let params = clouds::CloudsParams::from_graph(&node.params);
                    clouds::dispatch(gpu, pool, &key, resolution, &params);
                }
                "directional_blur" => {
                    let input_key = resolve_input("input")?;
                    let params = directional_blur::DirectionalBlurParams::from_graph(&node.params);
                    directional_blur::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "directional_warp" => {
                    let input_key = resolve_input("input")?;
                    let intensity_key = resolve_input("intensity")?;
                    let params = directional_warp::DirectionalWarpParams::from_graph(&node.params);
                    directional_warp::dispatch(
                        gpu,
                        pool,
                        &input_key,
                        &intensity_key,
                        &key,
                        resolution,
                        &params,
                    );
                }
                "edge_detect" => {
                    let input_key = resolve_input("input")?;
                    let params = edge_detect::EdgeDetectParams::from_graph(&node.params);
                    edge_detect::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "slope_blur" => {
                    let input_key = resolve_input("input")?;
                    let control_key = resolve_input("control")?;
                    let params = slope_blur::SlopeBlurParams::from_graph(&node.params);
                    slope_blur::dispatch(
                        gpu,
                        pool,
                        &input_key,
                        &control_key,
                        &key,
                        resolution,
                        &params,
                    );
                }
                "height_to_ao" => {
                    let input_key = resolve_input("input")?;
                    let params = height_to_ao::HeightToAoParams::from_graph(&node.params);
                    height_to_ao::dispatch(gpu, pool, &input_key, &key, resolution, &params);
                }
                "curvature_from_normal" => {
                    let input_key = resolve_input("input")?;
                    let params =
                        curvature_from_normal::CurvatureFromNormalParams::from_graph(&node.params);
                    curvature_from_normal::dispatch(
                        gpu, pool, &input_key, &key, resolution, &params,
                    );
                }
                "distance_field" => {
                    let mask_key = resolve_input("mask")?;
                    let params = distance_field::DistanceFieldParams::from_graph(&node.params);
                    distance_field::dispatch(gpu, pool, &mask_key, &key, resolution, &params);
                }
                "shape_boolean" => {
                    let a_key = resolve_input("a")?;
                    let b_key = resolve_input("b")?;
                    let sdf_key = texture_key(id, "sdf");
                    let a_only_key = texture_key(id, "a_only");
                    let b_only_key = texture_key(id, "b_only");
                    let overlap_key = texture_key(id, "overlap");
                    let blend_key = texture_key(id, "blend");
                    let params = shape_boolean::ShapeBooleanParams::from_graph(&node.params);
                    shape_boolean::dispatch(
                        gpu,
                        pool,
                        &a_key,
                        &b_key,
                        shape_boolean::ShapeBooleanOutputKeys {
                            sdf: &sdf_key,
                            a_only: &a_only_key,
                            b_only: &b_only_key,
                            overlap: &overlap_key,
                            blend: &blend_key,
                        },
                        resolution,
                        &params,
                    );
                }
                "redistance" => {
                    let sdf_key = resolve_input("sdf")?;
                    let params = redistance::RedistanceParams::from_graph(&node.params);
                    redistance::dispatch(gpu, pool, &sdf_key, &key, resolution, &params);
                }
                "bevel_profile" => {
                    let sdf_key = resolve_input("sdf")?;
                    let height_key = texture_key(id, "height");
                    let edge_key = texture_key(id, "edge");
                    let top_key = texture_key(id, "top");
                    let u_key = texture_key(id, "u");
                    let params = bevel_profile::BevelProfileParams::from_graph(&node.params);
                    bevel_profile::dispatch(
                        gpu,
                        pool,
                        &sdf_key,
                        bevel_profile::BevelProfileOutputKeys {
                            height: &height_key,
                            edge: &edge_key,
                            top: &top_key,
                            u: &u_key,
                        },
                        resolution,
                        &params,
                    );
                }
                "edge_wear" => {
                    let sdf_key = resolve_optional_input("sdf");
                    let height_key = resolve_optional_input("height");
                    let wear_key = texture_key(id, "wear");
                    let cavity_key = texture_key(id, "cavity");
                    let height_out_key = texture_key(id, "height");
                    let params = edge_wear::EdgeWearParams::from_graph(&node.params);
                    edge_wear::dispatch(
                        gpu,
                        pool,
                        sdf_key.as_deref(),
                        height_key.as_deref(),
                        edge_wear::EdgeWearOutputKeys {
                            wear: &wear_key,
                            cavity: &cavity_key,
                            height: &height_out_key,
                        },
                        resolution,
                        &params,
                    );
                }
                "panel_layout" => {
                    let sdf_key = texture_key(id, "sdf");
                    let panel_id_key = texture_key(id, "panel_id");
                    let panel_index_key = texture_key(id, "panel_index");
                    let uv_u_key = texture_key(id, "uv_u");
                    let uv_v_key = texture_key(id, "uv_v");
                    let seam_key = texture_key(id, "seam");
                    let params = panel_layout::PanelLayoutParams::from_graph(&node.params);
                    computed_path_result = Some(panel_layout::dispatch(
                        gpu,
                        pool,
                        panel_layout::PanelLayoutOutputKeys {
                            sdf: &sdf_key,
                            panel_id: &panel_id_key,
                            panel_index: &panel_index_key,
                            uv_u: &uv_u_key,
                            uv_v: &uv_v_key,
                            seam: &seam_key,
                        },
                        resolution,
                        &params,
                    ));
                }
                "image_trace" => {
                    let input_key = resolve_input("image")?;
                    let preview_mask_key = texture_key(id, "preview_mask");
                    let params = image_trace::TraceParams::from_graph(&node.params);
                    computed_path_result = Some(image_trace::dispatch(
                        gpu,
                        pool,
                        &input_key,
                        &preview_mask_key,
                        resolution,
                        &params,
                    ));
                }
                "path_simplify" => {
                    // A `Path` input, not a texture one — bypasses
                    // `resolve_input`/the texture pool entirely, same as
                    // the "path" node's own `Reference` (read directly
                    // from the upstream node's own `path` param).
                    // Unconnected degrades to an empty path rather than
                    // a `MissingInput` error, matching `Reference`'s own
                    // "optional" precedent.
                    let input_path = graph
                        .upstream_from(id, "input")
                        .and_then(|(upstream_id, _)| graph.node(upstream_id))
                        .and_then(|n| n.params.get("path"))
                        .and_then(mc_core::ParamValue::as_path)
                        .cloned()
                        .unwrap_or_default();
                    let params = path_simplify::SimplifyParams::from_graph(&node.params);
                    let simplified = path_simplify::simplify(&input_path, params.tolerance);
                    let preview_mask_key = texture_key(id, "preview_mask");
                    path::dispatch_preview_mask(
                        gpu,
                        pool,
                        &preview_mask_key,
                        resolution,
                        &simplified,
                    );
                    computed_path_result = Some(simplified);
                }
                "path_to_mask" => {
                    // Same "read the Path input directly, bypassing the
                    // texture pool" pattern as `path_simplify` above —
                    // unconnected degrades to an empty (blank) path
                    // rather than a `MissingInput` error.
                    let input_path = graph
                        .upstream_from(id, "input")
                        .and_then(|(upstream_id, _)| graph.node(upstream_id))
                        .and_then(|n| n.params.get("path"))
                        .and_then(mc_core::ParamValue::as_path)
                        .cloned()
                        .unwrap_or_default();
                    let params = path_to_mask::MaskParams::from_graph(&node.params);
                    path_to_mask::dispatch(gpu, pool, &key, resolution, &input_path, &params);
                }
                "path_to_sdf" => {
                    // Same "read the Path input directly, bypassing the
                    // texture pool" pattern as `path_to_mask` above.
                    let input_path = graph
                        .upstream_from(id, "input")
                        .and_then(|(upstream_id, _)| graph.node(upstream_id))
                        .and_then(|n| n.params.get("path"))
                        .and_then(mc_core::ParamValue::as_path)
                        .cloned()
                        .unwrap_or_default();
                    let params = path_to_sdf::PathToSdfParams::from_graph(&node.params);
                    path_to_sdf::dispatch(gpu, pool, &key, resolution, &input_path, &params);
                }
                "fasteners" => {
                    // Same "read the Path input directly, bypassing the
                    // texture pool" pattern as `path_to_sdf` above.
                    let input_path = graph
                        .upstream_from(id, "path")
                        .and_then(|(upstream_id, _)| graph.node(upstream_id))
                        .and_then(|n| n.params.get("path"))
                        .and_then(mc_core::ParamValue::as_path)
                        .cloned()
                        .unwrap_or_default();
                    let base_key = resolve_optional_input("base");
                    let height_key = texture_key(id, "height");
                    let mask_key = texture_key(id, "mask");
                    let fastener_id_key = texture_key(id, "fastener_id");
                    let params = fasteners::FastenersParams::from_graph(&node.params);
                    fasteners::dispatch(
                        gpu,
                        pool,
                        &input_path,
                        base_key.as_deref(),
                        fasteners::FastenersOutputKeys {
                            height: &height_key,
                            mask: &mask_key,
                            fastener_id: &fastener_id_key,
                        },
                        resolution,
                        &params,
                    );
                }
                other => return Err(EvalError::UnknownKind(id, other.to_string())),
            }
            Ok(())
        })();

        match result {
            Ok(()) => {
                // See `computed_path_result`'s own comment above: written
                // here, once `node`'s borrow of `graph` from this
                // iteration has ended, not from inside the closure.
                if let Some(path) = computed_path_result {
                    let _ = graph.set_computed_param(id, "path", mc_core::ParamValue::Path(path));
                }
                graph.mark_evaluated(id);
            }
            Err(err) => {
                failed.insert(id);
                if first_error.is_none() {
                    first_error = Some(err);
                }
            }
        }
    }

    match first_error {
        Some(err) => Err(err),
        None => Ok(()),
    }
}

/// The texture pool key actually holding `node`'s visible output — the one
/// shown in its thumbnail, the 2D View, and the 3D viewport. For most nodes
/// that's just its own (single) output; an Output node is a pass-through
/// marker, so its "output" is whatever feeds its `input` port; a
/// multi-output node (currently just Thread) shows its *first* declared
/// output (Height, for Thread — deliberately kept first in its `NodeDef` so
/// this falls out for free rather than needing a port picker).
pub fn resolve_texture_key(graph: &mc_core::Graph, node: mc_core::NodeId) -> Option<String> {
    let kind = &graph.node(node)?.kind;
    if kind.starts_with("output_") {
        // Which *port* fed this Output node's input matters now that a
        // node can have more than one (Thread) — `upstream` alone would
        // discard that and fall through to whatever the upstream node's
        // own default/first output is, showing Height no matter which of
        // Thread's four outputs was actually wired in (confirmed bug: the
        // owner wired Ply ID, Fiber, and Mask into three separate Output
        // nodes and all three showed Height). An Output node is never
        // itself a multi-output node, so if the *upstream* one is, follow
        // its specific port directly instead of recursing into the
        // single-output branch below, which would re-derive the wrong
        // (first/default) port all over again.
        let (upstream, upstream_port) = graph.upstream_from(node, "input")?;
        let upstream_kind = &graph.node(upstream)?.kind;
        if upstream_kind.starts_with("output_") {
            resolve_texture_key(graph, upstream)
        } else {
            Some(texture_key(upstream, &upstream_port))
        }
    } else {
        let port = crate::defs::node_def(kind)
            .and_then(|def| def.outputs.first())
            .map(|p| p.name)
            .unwrap_or(DEFAULT_OUTPUT_PORT);
        Some(texture_key(node, port))
    }
}

/// `node`'s own effectively-previewed output port type — mirrors
/// `resolve_texture_key`'s exact traversal above (including the
/// Output-node upstream redirect: an Output node's own declared port type
/// is irrelevant, what matters is whichever upstream port actually feeds
/// it) but returns the resolved port's `defs::effective_output_type`
/// instead of a texture key. Kept as a separate function rather than
/// changing `resolve_texture_key`'s own return type, since that function
/// already has many call sites that only want the key. Used only to decide
/// whether a preview should render Sdf iso-bands instead of a normal sRGB
/// encode (`mc_app`'s `display_encode`).
pub fn resolve_output_port_type(
    graph: &mc_core::Graph,
    node: mc_core::NodeId,
) -> Option<mc_core::PortType> {
    let kind = &graph.node(node)?.kind;
    if kind.starts_with("output_") {
        let (upstream, upstream_port) = graph.upstream_from(node, "input")?;
        let upstream_kind = &graph.node(upstream)?.kind;
        if upstream_kind.starts_with("output_") {
            resolve_output_port_type(graph, upstream)
        } else {
            let declared = crate::defs::node_def(upstream_kind)
                .and_then(|def| def.output(&upstream_port))
                .map(|p| p.port_type)?;
            Some(crate::defs::effective_output_type(
                graph,
                upstream,
                upstream_kind,
                declared,
            ))
        }
    } else {
        let def = crate::defs::node_def(kind)?;
        let port = def.outputs.first()?;
        Some(crate::defs::effective_output_type(
            graph,
            node,
            kind,
            port.port_type,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{evaluate, path, resolve_texture_key, texture_key};

    /// Regression test for a real bug the owner hit: wiring Thread's Ply
    /// ID, Fiber, and Mask outputs each into their own Output node, and
    /// all three showing Height instead. Root cause: `resolve_texture_key`
    /// recursed into a multi-output upstream node without carrying along
    /// *which* of its outputs the connection actually used, silently
    /// falling back to that node's first/default output every time.
    #[test]
    fn an_output_node_shows_the_specific_upstream_port_it_is_wired_to() {
        let mut graph = mc_core::Graph::new();
        let thread = graph.add_node(mc_core::Node::new("thread_shape"));
        let height_out = graph.add_node(mc_core::Node::new("output_height"));
        let ply_id_out = graph.add_node(mc_core::Node::new("output_height"));
        let mask_out = graph.add_node(mc_core::Node::new("output_height"));
        graph
            .connect(thread, "height", height_out, "input")
            .unwrap();
        graph
            .connect(thread, "ply_id", ply_id_out, "input")
            .unwrap();
        graph.connect(thread, "mask", mask_out, "input").unwrap();
        graph.settings.resolution = 32;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let height_key = resolve_texture_key(&graph, height_out).unwrap();
        let ply_id_key = resolve_texture_key(&graph, ply_id_out).unwrap();
        let mask_key = resolve_texture_key(&graph, mask_out).unwrap();

        assert_eq!(height_key, texture_key(thread, "height"));
        assert_eq!(ply_id_key, texture_key(thread, "ply_id"));
        assert_eq!(mask_key, texture_key(thread, "mask"));

        let height_pixels = gpu.read_rgba8(pool.texture(&height_key).unwrap(), 32);
        let ply_id_pixels = gpu.read_rgba8(pool.texture(&ply_id_key).unwrap(), 32);
        let mask_pixels = gpu.read_rgba8(pool.texture(&mask_key).unwrap(), 32);
        assert_ne!(
            height_pixels, ply_id_pixels,
            "an Output node fed by Ply ID should not show Height's pixels"
        );
        assert_ne!(
            height_pixels, mask_pixels,
            "an Output node fed by Mask should not show Height's pixels"
        );
    }

    /// Knit's own version of the test above: a graph-level (not just
    /// `knit::dispatch`-level) check that wiring two of its outputs into
    /// separate Output nodes shows each one's own pixels, not a repeat of
    /// whichever output happens to be declared first.
    #[test]
    fn knit_output_node_shows_the_specific_upstream_port_it_is_wired_to() {
        let mut graph = mc_core::Graph::new();
        let knit_node = graph.add_node(crate::defs::node_def("knit").unwrap().instantiate());
        let height_out = graph.add_node(mc_core::Node::new("output_height"));
        let stitch_id_out = graph.add_node(mc_core::Node::new("output_height"));
        graph
            .connect(knit_node, "height", height_out, "input")
            .unwrap();
        graph
            .connect(knit_node, "stitch_id", stitch_id_out, "input")
            .unwrap();
        graph.settings.resolution = 32;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let height_key = resolve_texture_key(&graph, height_out).unwrap();
        let stitch_id_key = resolve_texture_key(&graph, stitch_id_out).unwrap();
        assert_eq!(height_key, texture_key(knit_node, "height"));
        assert_eq!(stitch_id_key, texture_key(knit_node, "stitch_id"));

        let height_pixels = gpu.read_rgba8(pool.texture(&height_key).unwrap(), 32);
        let stitch_id_pixels = gpu.read_rgba8(pool.texture(&stitch_id_key).unwrap(), 32);
        assert_ne!(
            height_pixels, stitch_id_pixels,
            "an Output node fed by Stitch ID should not show Height's pixels"
        );
    }

    /// Every param `knit::defs::NodeDef::instantiate` sets as a fresh
    /// node's default should round-trip through `KnitParams::from_graph`
    /// and evaluate without panicking — the same contract every other node
    /// gets implicitly by being drag-and-droppable onto the canvas.
    #[test]
    fn knit_default_instantiation_evaluates() {
        let mut graph = mc_core::Graph::new();
        let knit_node = graph.add_node(crate::defs::node_def("knit").unwrap().instantiate());
        let height_out = graph.add_node(mc_core::Node::new("output_height"));
        graph
            .connect(knit_node, "height", height_out, "input")
            .unwrap();
        graph.settings.resolution = 32;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");
    }

    /// A fresh Path node (no points — the user hasn't drawn anything yet)
    /// should still evaluate without panicking, same contract as every
    /// other node's default instantiation.
    #[test]
    fn path_default_instantiation_evaluates() {
        let mut graph = mc_core::Graph::new();
        let path_node = graph.add_node(crate::defs::node_def("path").unwrap().instantiate());
        let mask_out = graph.add_node(mc_core::Node::new("output_height"));
        graph
            .connect(path_node, "preview_mask", mask_out, "input")
            .unwrap();
        graph.settings.resolution = 32;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");
    }

    #[test]
    fn stitch_default_instantiation_evaluates() {
        let mut graph = mc_core::Graph::new();
        let stitch_node = graph.add_node(crate::defs::node_def("stitch").unwrap().instantiate());
        let height_out = graph.add_node(mc_core::Node::new("output_height"));
        graph
            .connect(stitch_node, "height", height_out, "input")
            .unwrap();
        // Thread radius at the default 0.45mm/50mm diameter is genuinely
        // sub-pixel below roughly 128px — not a bug, just too coarse a
        // test resolution for a physically-thin thread to reliably cover
        // any whole pixel.
        graph.settings.resolution = 256;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let height_key = resolve_texture_key(&graph, height_out).unwrap();
        let pixels = gpu.read_rgba8(pool.texture(&height_key).unwrap(), 256);
        let lit = pixels.iter().step_by(4).filter(|&&v| v > 0).count();
        assert!(
            lit > 0,
            "a fresh Stitch node's default back-stitch should be visible"
        );
    }

    /// French Knot has its own `KnotParams`/`crate::french_knot` entirely
    /// (see `mc_nodes::stitch::StitchKind`'s own doc comment) — this is
    /// the one test exercising this module's own branch to it, end to end
    /// through a real graph, rather than `crate::french_knot`'s own tests
    /// calling `dispatch_render` directly.
    #[test]
    fn stitch_node_evaluates_as_french_knot_when_selected() {
        let mut graph = mc_core::Graph::new();
        let stitch_node = graph.add_node(crate::defs::node_def("stitch").unwrap().instantiate());
        graph
            .set_param(
                stitch_node,
                "stitch",
                mc_core::ParamValue::Enum("french_knot".to_string()),
            )
            .unwrap();
        let height_out = graph.add_node(mc_core::Node::new("output_height"));
        graph
            .connect(stitch_node, "height", height_out, "input")
            .unwrap();
        graph.settings.resolution = 128;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let height_key = resolve_texture_key(&graph, height_out).unwrap();
        let pixels = gpu.read_rgba8(pool.texture(&height_key).unwrap(), 128);
        let lit = pixels.iter().step_by(4).filter(|&&v| v > 0).count();
        assert!(
            lit > 0,
            "a Stitch node set to French Knot should be visible"
        );
    }

    /// Every one of Stitch's six outputs should show its own data, not a
    /// repeat of whichever output happens to be declared first — the same
    /// bug class `knit_output_node_shows_the_specific_upstream_port_it_is_wired_to`
    /// already regression-tests for Knit.
    #[test]
    fn stitch_output_nodes_show_the_specific_port_they_are_wired_to() {
        let mut graph = mc_core::Graph::new();
        let stitch_node = graph.add_node(crate::defs::node_def("stitch").unwrap().instantiate());
        let height_out = graph.add_node(mc_core::Node::new("output_height"));
        let mask_out = graph.add_node(mc_core::Node::new("output_height"));
        graph
            .connect(stitch_node, "height", height_out, "input")
            .unwrap();
        graph
            .connect(stitch_node, "mask", mask_out, "input")
            .unwrap();
        graph.settings.resolution = 256;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let height_key = resolve_texture_key(&graph, height_out).unwrap();
        let mask_key = resolve_texture_key(&graph, mask_out).unwrap();
        assert_eq!(height_key, texture_key(stitch_node, "height"));
        assert_eq!(mask_key, texture_key(stitch_node, "mask"));

        let height_pixels = gpu.read_rgba8(pool.texture(&height_key).unwrap(), 256);
        let mask_pixels = gpu.read_rgba8(pool.texture(&mask_key).unwrap(), 256);
        assert_ne!(
            height_pixels, mask_pixels,
            "an Output node fed by Mask should not show Height's own pixels"
        );
    }

    /// The owner's own exact workflow: a Shape node wired into Path's own
    /// Stamp input should change Path's output — the graph-level version
    /// of `path::tests::dispatch_stamp_composites_the_stamp_texture_over_the_base`,
    /// confirming the NodeDef/registry wiring actually reaches it, not
    /// just the raw GPU call.
    #[test]
    fn shape_wired_into_paths_stamp_input_changes_its_output() {
        fn build_graph() -> (mc_core::Graph, mc_core::NodeId, mc_core::NodeId) {
            let mut graph = mc_core::Graph::new();
            let shape_node = graph.add_node(crate::defs::node_def("shape").unwrap().instantiate());
            let path_node = graph.add_node(crate::defs::node_def("path").unwrap().instantiate());
            let line_path = mc_core::Path {
                subpaths: vec![mc_core::SubPath::new(
                    vec![
                        mc_core::Point::new([0.1, 0.5]),
                        mc_core::Point::new([0.9, 0.5]),
                    ],
                    false,
                )],
                wrap: true,
            };
            graph
                .set_param(path_node, "path", mc_core::ParamValue::Path(line_path))
                .unwrap();
            graph.settings.resolution = 64;
            (graph, shape_node, path_node)
        }

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();

        let (mut graph, _shape, path_node) = build_graph();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates without a stamp wired");
        let without_stamp = gpu.read_rgba8(
            pool.texture(&texture_key(path_node, "preview_mask"))
                .unwrap(),
            64,
        );

        let (mut graph, shape_node, path_node) = build_graph();
        graph
            .connect(shape_node, "output", path_node, "stamp")
            .unwrap();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates with a stamp wired");
        let with_stamp = gpu.read_rgba8(
            pool.texture(&texture_key(path_node, "preview_mask"))
                .unwrap(),
            64,
        );

        assert_ne!(
            without_stamp, with_stamp,
            "wiring a Shape into Path's Stamp input should change its output"
        );
    }

    /// The actual intended workflow (not just Shape standing in for it):
    /// Stitch's own Height output wired into Path's Stamp input should
    /// splatter real embroidery-stitch geometry along the curve.
    #[test]
    fn stitch_wired_into_paths_stamp_input_changes_its_output() {
        let mut graph = mc_core::Graph::new();
        let stitch_node = graph.add_node(crate::defs::node_def("stitch").unwrap().instantiate());
        let path_node = graph.add_node(crate::defs::node_def("path").unwrap().instantiate());
        let line_path = mc_core::Path {
            subpaths: vec![mc_core::SubPath::new(
                vec![
                    mc_core::Point::new([0.1, 0.5]),
                    mc_core::Point::new([0.9, 0.5]),
                ],
                false,
            )],
            wrap: true,
        };
        graph
            .set_param(path_node, "path", mc_core::ParamValue::Path(line_path))
            .unwrap();
        graph
            .connect(stitch_node, "height", path_node, "stamp")
            .unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let preview_mask = gpu.read_rgba8(
            pool.texture(&texture_key(path_node, "preview_mask"))
                .unwrap(),
            64,
        );
        let lit = preview_mask.iter().step_by(4).filter(|&&v| v > 0).count();
        assert!(
            lit > 0,
            "Stitch stamped along Path's curve should be visible"
        );
    }

    /// CLAUDE.md's M2 acceptance check: a 3-node graph (Noise → Levels →
    /// Output) evaluates headlessly.
    #[test]
    fn evaluates_noise_levels_output_graph() {
        let mut graph = mc_core::Graph::new();
        let noise = graph.add_node(mc_core::Node::new("perlin_noise"));
        let levels = graph.add_node(mc_core::Node::new("levels"));
        let output = graph.add_node(mc_core::Node::new("output_base_color"));
        graph.connect(noise, "output", levels, "input").unwrap();
        graph.connect(levels, "output", output, "input").unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let key = resolve_texture_key(&graph, output).expect("output resolves to a texture");
        let pixels = gpu.read_rgba8(pool.texture(&key).expect("texture exists"), 64);
        assert_eq!(pixels.len(), 64 * 64 * 4);

        for id in [noise, levels, output] {
            assert!(
                !graph.needs_evaluation(id),
                "node {id:?} should be marked up to date after evaluate()"
            );
        }
    }

    /// Regression test for the Pattern Generator/Value Range/Flow Blur/Push
    /// Warp rename (`display_name` only, `kind` strings unchanged) — proves
    /// the registry's `"slope_blur"` dispatch arm (now showing as "Flow
    /// Blur" in the UI) still resolves and runs correctly end to end, not
    /// just the node's own isolated dispatch test. Both its inputs ("input"
    /// and "control") are required — the Output node is deliberately *not*
    /// fed through a single-input shortcut here, since the real-world
    /// report this guards against ("Flow Blur isn't visualizing") was most
    /// likely a missing second ("control") connection, not a dispatch bug.
    #[test]
    fn flow_blur_evaluates_through_the_registry_after_its_rename() {
        let mut graph = mc_core::Graph::new();
        let input_noise = graph.add_node(
            mc_core::Node::new("perlin_noise").with_param("seed", mc_core::ParamValue::Seed(1)),
        );
        let control_noise = graph.add_node(
            mc_core::Node::new("perlin_noise").with_param("seed", mc_core::ParamValue::Seed(2)),
        );
        let flow_blur = graph.add_node(mc_core::Node::new("slope_blur"));
        let output = graph.add_node(mc_core::Node::new("output_base_color"));

        graph
            .connect(input_noise, "output", flow_blur, "input")
            .unwrap();
        graph
            .connect(control_noise, "output", flow_blur, "control")
            .unwrap();
        graph.connect(flow_blur, "output", output, "input").unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates with both inputs wired");

        let key = resolve_texture_key(&graph, output).expect("output resolves to a texture");
        let pixels = gpu.read_rgba8(pool.texture(&key).expect("texture exists"), 64);
        assert!(
            pixels.iter().any(|&b| b != pixels[0]),
            "Flow Blur's output is perfectly flat — looks like it isn't actually running"
        );
    }

    /// CLAUDE.md's M3 acceptance demo: a 4-node graph (Noise → Levels →
    /// Blend → Output) evaluates headlessly. Blend's second input comes from
    /// a second Perlin node so Blend has two genuinely different layers to
    /// combine.
    #[test]
    fn evaluates_noise_levels_blend_output_graph() {
        let mut graph = mc_core::Graph::new();
        let noise_a = graph.add_node(
            mc_core::Node::new("perlin_noise").with_param("seed", mc_core::ParamValue::Seed(1)),
        );
        let noise_b = graph.add_node(
            mc_core::Node::new("perlin_noise").with_param("seed", mc_core::ParamValue::Seed(2)),
        );
        let levels = graph.add_node(mc_core::Node::new("levels"));
        let blend = graph.add_node(mc_core::Node::new("blend"));
        let output = graph.add_node(mc_core::Node::new("output_base_color"));

        graph.connect(noise_a, "output", levels, "input").unwrap();
        graph.connect(levels, "output", blend, "base").unwrap();
        graph.connect(noise_b, "output", blend, "blend").unwrap();
        graph.connect(blend, "output", output, "input").unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let key = resolve_texture_key(&graph, output).expect("output resolves to a texture");
        let pixels = gpu.read_rgba8(pool.texture(&key).expect("texture exists"), 64);
        assert_eq!(pixels.len(), 64 * 64 * 4);

        for id in [noise_a, noise_b, levels, blend, output] {
            assert!(
                !graph.needs_evaluation(id),
                "node {id:?} should be marked up to date after evaluate()"
            );
        }
    }

    #[test]
    fn re_evaluating_an_unchanged_graph_is_a_cache_hit() {
        let mut graph = mc_core::Graph::new();
        let noise = graph.add_node(mc_core::Node::new("perlin_noise"));
        graph.settings.resolution = 32;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).unwrap();
        assert!(!graph.needs_evaluation(noise));

        // Nothing changed, so a second evaluate() pass should be a no-op —
        // verified indirectly: it must not error and the node stays clean.
        evaluate(&mut gpu, &mut pool, &mut graph).unwrap();
        assert!(!graph.needs_evaluation(noise));
    }

    /// Regression test: a single unconnected filter node anywhere in the
    /// graph (completely normal mid-edit state — e.g. a just-added node
    /// nobody's wired up yet) used to abort the *entire* evaluation pass at
    /// the first `?`, silently leaving every other node after it in
    /// topological order un-evaluated too, including fully-wired,
    /// independent ones. A standalone generator with no inputs at all
    /// (Voronoi here) must still get its texture even when an unrelated,
    /// unconnected Blur node sorts before it.
    #[test]
    fn one_unconnected_node_does_not_block_the_rest_of_the_graph() {
        let mut graph = mc_core::Graph::new();
        let orphan_blur = graph.add_node(mc_core::Node::new("blur"));
        let voronoi = graph.add_node(mc_core::Node::new("voronoi"));
        graph.settings.resolution = 32;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        let result = evaluate(&mut gpu, &mut pool, &mut graph);

        assert!(
            result.is_err(),
            "the orphan Blur's missing input should still be reported"
        );
        let key = resolve_texture_key(&graph, voronoi).expect("voronoi resolves to a texture");
        assert!(
            pool.texture(&key).is_some(),
            "an independent, fully-standalone node must still get evaluated even though \
             an unrelated node elsewhere in the graph is missing a required input"
        );
        assert!(
            graph.needs_evaluation(orphan_blur),
            "the actually-broken node should stay dirty, not get marked evaluated"
        );
        assert!(
            !graph.needs_evaluation(voronoi),
            "the independent node should be marked up to date"
        );
    }

    /// End-to-end through the real graph/registry path (defs.rs defaults
    /// via `from_graph`, multi-output texture keys, the optional
    /// `angle_map` input left unconnected) rather than calling `dispatch`
    /// directly, for both of GABOR_STREAKS_SPEC.md's nodes together —
    /// Gabor Noise's `value` feeding Base Color, Scattered Streaks'
    /// `height` feeding a second Output.
    #[test]
    fn gabor_noise_and_scattered_streaks_evaluate_through_the_graph() {
        let mut graph = mc_core::Graph::new();
        let gabor = graph.add_node(mc_core::Node::new("gabor_noise"));
        let streaks = graph.add_node(mc_core::Node::new("scattered_streaks"));
        let gabor_output = graph.add_node(mc_core::Node::new("output_base_color"));
        let streaks_output = graph.add_node(mc_core::Node::new("output_height"));
        graph
            .connect(gabor, "value", gabor_output, "input")
            .unwrap();
        graph
            .connect(streaks, "height", streaks_output, "input")
            .unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let gabor_key =
            resolve_texture_key(&graph, gabor_output).expect("gabor output resolves to a texture");
        let gabor_pixels = gpu.read_rgba8(pool.texture(&gabor_key).expect("texture exists"), 64);
        assert_eq!(gabor_pixels.len(), 64 * 64 * 4);

        let streaks_key = resolve_texture_key(&graph, streaks_output)
            .expect("streaks output resolves to a texture");
        let streaks_pixels =
            gpu.read_rgba8(pool.texture(&streaks_key).expect("texture exists"), 64);
        assert_eq!(streaks_pixels.len(), 64 * 64 * 4);

        for id in [gabor, streaks, gabor_output, streaks_output] {
            assert!(
                !graph.needs_evaluation(id),
                "node {id:?} should be marked up to date after evaluate()"
            );
        }
    }

    /// End-to-end through the real graph/registry path: a Shape node's
    /// output feeds Image Trace's `Image` input, traced into both a real
    /// Preview Mask texture and this node's own `path` param. The second
    /// `evaluate()` call (nothing changed in between) is the point of
    /// this test — it directly exercises `set_computed_param`'s own
    /// reason for existing (its doc comment on `mc_core::Graph`): writing
    /// the traced path into this node's own params during its own
    /// evaluation must *not* re-dirty anything, or every node downstream
    /// of an Image Trace would needlessly re-evaluate on every single
    /// pass forever.
    #[test]
    fn image_trace_evaluates_through_the_graph_without_re_dirtying_itself() {
        let mut graph = mc_core::Graph::new();
        let shape = graph.add_node(mc_core::Node::new("shape"));
        let tracer = graph.add_node(mc_core::Node::new("image_trace"));
        graph.connect(shape, "output", tracer, "image").unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let preview_key = texture_key(tracer, "preview_mask");
        let pixels = gpu.read_rgba8(
            pool.texture(&preview_key)
                .expect("preview mask texture exists"),
            64,
        );
        assert_eq!(pixels.len(), 64 * 64 * 4);

        let path = graph
            .node(tracer)
            .and_then(|n| n.params.get("path"))
            .and_then(mc_core::ParamValue::as_path)
            .expect("image_trace wrote its own computed path param");
        assert!(
            !path.subpaths.is_empty(),
            "tracing a Shape node's own disc should find at least one boundary"
        );

        for id in [shape, tracer] {
            assert!(
                !graph.needs_evaluation(id),
                "node {id:?} should be marked up to date after evaluate()"
            );
        }

        evaluate(&mut gpu, &mut pool, &mut graph).expect("second no-op evaluate");
        for id in [shape, tracer] {
            assert!(
                !graph.needs_evaluation(id),
                "node {id:?} must stay up to date after a no-op second evaluate() -- \
                 re-dirtying here would mean set_computed_param regressed into the \
                 dirty-cascading behavior it exists specifically to avoid"
            );
        }
    }

    /// Toggling `Invert` through the real UI-equivalent path
    /// (`graph.set_param`, not calling `image_trace::dispatch` directly)
    /// must actually reach a re-trace with the new value — added after
    /// the owner reported the checkbox "doesn't seem to do anything."
    /// It does: this traces a Shape node's own disc with and without
    /// Invert and confirms the two resulting paths genuinely differ (not
    /// just that the node got marked dirty). The owner's own report
    /// turned out to be about *Preview Mask only ever stroking a
    /// subpath's outline, never filling it* — inverting a mostly-solid
    /// shape swaps which side is foreground, but an outline-only render
    /// of "the same boundary curves, opposite fill" can look deceptively
    /// similar at a glance, especially for line-art-style source images.
    #[test]
    fn invert_actually_changes_the_traced_path_through_the_real_graph() {
        let mut graph = mc_core::Graph::new();
        let shape = graph.add_node(mc_core::Node::new("shape"));
        let tracer = graph.add_node(mc_core::Node::new("image_trace"));
        graph.connect(shape, "output", tracer, "image").unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");
        let path_normal = graph
            .node(tracer)
            .and_then(|n| n.params.get("path"))
            .and_then(mc_core::ParamValue::as_path)
            .cloned()
            .expect("image_trace wrote its own computed path param");

        graph
            .set_param(tracer, "invert", mc_core::ParamValue::Bool(true))
            .expect("invert is a real param on image_trace");
        assert!(
            graph.needs_evaluation(tracer),
            "toggling Invert via the normal set_param path should dirty this node"
        );
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates again");
        let path_inverted = graph
            .node(tracer)
            .and_then(|n| n.params.get("path"))
            .and_then(mc_core::ParamValue::as_path)
            .cloned()
            .expect("image_trace wrote its own computed path param");

        let same_structure = path_normal.subpaths.len() == path_inverted.subpaths.len()
            && path_normal
                .subpaths
                .iter()
                .zip(&path_inverted.subpaths)
                .all(|(a, b)| a.points.len() == b.points.len());
        assert!(
            !same_structure,
            "inverting should trace the opposite side of the same boundary -- got the same \
             subpath/point structure both times ({} subpaths) for Normal and Inverted",
            path_normal.subpaths.len()
        );
    }

    /// The actual feature request this was all built for: Image Trace's
    /// `Path` output wired into a Path node's own `Reference` input picks
    /// up the traced curve with zero changes to the Path node — it
    /// already reads whatever's at `upstream_node.params["path"]`
    /// (`path::resolve_effective_path`), regardless of the upstream
    /// node's own `kind`.
    #[test]
    fn image_trace_path_output_feeds_a_path_nodes_reference_input() {
        let mut graph = mc_core::Graph::new();
        let shape = graph.add_node(mc_core::Node::new("shape"));
        let tracer = graph.add_node(mc_core::Node::new("image_trace"));
        let path_node = graph.add_node(mc_core::Node::new("path"));
        graph.connect(shape, "output", tracer, "image").unwrap();
        graph
            .connect(tracer, "path", path_node, "reference")
            .unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let effective = path::resolve_effective_path(&graph, path_node);
        assert!(
            !effective.subpaths.is_empty(),
            "the Path node should see Image Trace's own traced subpaths through Reference"
        );

        let traced = graph
            .node(tracer)
            .and_then(|n| n.params.get("path"))
            .and_then(mc_core::ParamValue::as_path)
            .expect("image_trace wrote its own computed path param");
        assert_eq!(
            effective.subpaths.len(),
            traced.subpaths.len(),
            "the Path node (with no points of its own) should pick up exactly Image Trace's \
             subpaths, unmodified"
        );
    }

    /// Path Simplify chained after Image Trace — the actual workflow it
    /// was built for (a dense trace overwhelming the Path node's own
    /// Stamp Spacing floor). Also exercises the same `set_computed_param`
    /// no-redirty guarantee `image_trace_evaluates_through_the_graph_
    /// without_re_dirtying_itself` checks, since this is the second node
    /// kind to use it.
    #[test]
    fn path_simplify_reduces_an_image_trace_result_through_the_real_graph() {
        let mut graph = mc_core::Graph::new();
        let shape = graph.add_node(mc_core::Node::new("shape"));
        let tracer = graph.add_node(mc_core::Node::new("image_trace"));
        let simplifier = graph.add_node(mc_core::Node::new("path_simplify"));
        graph.connect(shape, "output", tracer, "image").unwrap();
        graph.connect(tracer, "path", simplifier, "input").unwrap();
        graph.settings.resolution = 128;
        graph
            .set_param(simplifier, "tolerance", mc_core::ParamValue::Float(0.05))
            .unwrap();

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let traced = graph
            .node(tracer)
            .and_then(|n| n.params.get("path"))
            .and_then(mc_core::ParamValue::as_path)
            .cloned()
            .expect("image_trace wrote its own computed path param");
        let simplified = graph
            .node(simplifier)
            .and_then(|n| n.params.get("path"))
            .and_then(mc_core::ParamValue::as_path)
            .cloned()
            .expect("path_simplify wrote its own computed path param");

        let traced_points: usize = traced.subpaths.iter().map(|sp| sp.points.len()).sum();
        let simplified_points: usize = simplified.subpaths.iter().map(|sp| sp.points.len()).sum();
        assert!(
            simplified_points < traced_points,
            "simplifying at a non-trivial tolerance should reduce the point count: \
             traced {traced_points}, simplified {simplified_points}"
        );
        assert_eq!(
            simplified.subpaths.len(),
            traced.subpaths.len(),
            "simplifying shouldn't merge or drop whole subpaths, just thin their points"
        );

        for id in [shape, tracer, simplifier] {
            assert!(
                !graph.needs_evaluation(id),
                "node {id:?} should be marked up to date after evaluate()"
            );
        }
        evaluate(&mut gpu, &mut pool, &mut graph).expect("second no-op evaluate");
        for id in [shape, tracer, simplifier] {
            assert!(
                !graph.needs_evaluation(id),
                "node {id:?} must stay up to date after a no-op second evaluate()"
            );
        }
    }
}
