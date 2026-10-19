// SPDX-License-Identifier: MIT OR Apache-2.0
//! mc_nodes: node definitions — metadata, params, and the WGSL compute
//! shader that implements each one. [`registry`] bridges mc_core's generic
//! graph model to actually dispatching each node's GPU work.
//!
//! Each node's tests live alongside it in its own `#[cfg(test)] mod tests`,
//! sharing the helpers in `test_support`.

pub mod anisotropic_noise;
pub mod bevel;
pub mod bevel_profile;
pub mod blend;
pub mod block_shift;
pub mod blur;
pub mod channel_pack;
pub mod channel_split;
pub mod checker;
pub mod circular_pattern;
pub mod clamp;
pub mod clouds;
pub mod color_match;
mod connected_components;
pub mod curvature_from_normal;
pub mod curve;
pub mod data_corrupt;
pub mod defs;
pub mod directional_blur;
pub mod directional_warp;
pub mod distance_field;
pub mod distance_spread;
mod dither_common;
pub mod dither_pattern;
pub mod edge_detect;
pub mod edge_wear;
pub mod export;
pub mod fabric_generator;
pub mod fasteners;
pub mod fractal_sum;
pub mod french_knot;
pub mod gabor_noise;
pub mod gradient_linear;
pub mod gradient_map;
pub mod gradient_radial;
pub mod grayscale_conversion;
pub mod height_to_ao;
pub mod height_to_normal;
pub mod histogram_scan;
pub mod hsl_adjust;
pub mod image_import;
pub mod image_trace;
pub mod invert;
pub mod knit;
pub mod levels;
pub mod material_preset;
pub mod ordered_dither;
pub mod palette_dither;
pub mod panel_layout;
pub mod path;
pub mod path_simplify;
pub mod path_to_mask;
pub mod path_to_sdf;
pub mod perlin;
pub mod redistance;
pub mod registry;
pub mod scanline_jitter;
pub mod scattered_streaks;
mod shaders;
pub mod shape;
pub mod shape_boolean;
pub mod simplex_noise;
pub mod slope_blur;
pub mod stitch;
pub mod svg_import;
#[cfg(test)]
mod test_support;
pub mod thread_shape;
pub mod threshold;
pub mod tile_generator;
pub mod tiling;
pub mod transform2d;
pub mod tube_raster;
pub mod type_text;
pub mod uniform_color;
pub mod voronoi;
pub mod warp;
pub mod white_noise;
