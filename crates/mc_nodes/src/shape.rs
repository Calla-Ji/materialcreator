// SPDX-License-Identifier: MIT OR Apache-2.0
//! Generators / Patterns / Shape.
//!
//! Display name: "Shape". Tooltip: "A single disc, square, polygon, or star,
//! centered in the tile."

use crate::shaders::full_source;

const NODE_KIND: &str = "shape";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeKind {
    Disc,
    Square,
    Polygon,
    Star,
    Leaf,
}

impl ShapeKind {
    pub const NAMES: [&'static str; 5] = ["Disc", "Square", "Polygon", "Star", "Leaf"];

    fn as_u32(self) -> u32 {
        match self {
            ShapeKind::Disc => 0,
            ShapeKind::Square => 1,
            ShapeKind::Polygon => 2,
            ShapeKind::Star => 3,
            ShapeKind::Leaf => 4,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Disc" => Some(ShapeKind::Disc),
            "Square" => Some(ShapeKind::Square),
            "Polygon" => Some(ShapeKind::Polygon),
            "Star" => Some(ShapeKind::Star),
            "Leaf" => Some(ShapeKind::Leaf),
            _ => None,
        }
    }
}

pub struct ShapeParams {
    pub kind: ShapeKind,
    pub sides: u32,
    pub size: f32,
    /// Independent X/Y stretch on top of `size`'s own uniform scale,
    /// applied in the shape's own local space *after* `rotation` carries
    /// the query point into that frame (`shape.wgsl`'s own comment on
    /// this has the full reasoning — stretching before rotating a Disc
    /// specifically would be an invisible no-op, since rotation preserves
    /// vector length). The net effect is the usual one: stretching a Disc
    /// into an ellipse and then rotating it tilts the whole ellipse, the
    /// same result every DCC tool gives for a non-uniformly scaled, then
    /// rotated, object. 1.0 is no stretch on that axis; this is how every
    /// shape kind (not just Disc/Square) gets a real elliptical or
    /// rectangular variant, since the stretch happens on the shared input
    /// coordinate every kind's own distance field reads, not duplicated
    /// per-kind math.
    pub scale_x: f32,
    pub scale_y: f32,
    pub softness: f32,
    pub rotation: f32,
    /// Moves the shape's own center away from the tile's, tile units —
    /// wraps (the shape reappears on the opposite edge past 0.5), so this
    /// stays seamless without needing a separate Transform 2D node just to
    /// reposition a single Shape.
    pub offset_x: f32,
    pub offset_y: f32,
    /// Star only: how deep the points cut in, 1.0..0.05. At 1.0 the star's
    /// inner and outer radius match, so it's indistinguishable from a plain
    /// Polygon of the same side count (e.g. a triangle at `sides = 3`);
    /// lowering it "unfolds" that same polygon into sharper and sharper
    /// star points, down to thin spikes near 0.05.
    pub point_depth: f32,
    /// Leaf only: how pointed (vs. rounded) its top tip is.
    pub curvature_top: f32,
    /// Leaf only: how pointed (vs. rounded) its bottom tip is.
    pub curvature_bottom: f32,
    /// Leaf only: how wide it is at its fattest point (the equator), as a
    /// fraction of its own height.
    pub leaf_width: f32,
    /// Leaf only: how pointed (vs. rounded) its sides are through the
    /// middle (the equator), independently from either tip.
    pub curvature_middle: f32,
    /// Leaf only: 0.15..0.85, where the equator sits between the top tip
    /// (0) and bottom tip (1). 0.5 is centered.
    pub middle_position: f32,
}

impl Default for ShapeParams {
    fn default() -> Self {
        Self {
            kind: ShapeKind::Disc,
            sides: 5,
            size: 0.6,
            scale_x: 1.0,
            scale_y: 1.0,
            // The owner's own request: the preset should always start
            // with a hard edge, not the old 0.05 default.
            softness: 0.0,
            rotation: 0.0,
            offset_x: 0.0,
            offset_y: 0.0,
            point_depth: 0.5,
            curvature_top: 0.6,
            curvature_bottom: 0.6,
            leaf_width: 0.55,
            curvature_middle: 0.6,
            middle_position: 0.5,
        }
    }
}

impl ShapeParams {
    pub fn from_graph(params: &std::collections::BTreeMap<String, mc_core::ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            kind: params
                .get("kind")
                .and_then(|v| match v {
                    mc_core::ParamValue::Enum(name) => ShapeKind::from_name(name),
                    _ => None,
                })
                .unwrap_or(defaults.kind),
            sides: params
                .get("sides")
                .and_then(mc_core::ParamValue::as_int)
                .map(|v| v.max(3) as u32)
                .unwrap_or(defaults.sides),
            size: params
                .get("size")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.size),
            scale_x: params
                .get("scale_x")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.scale_x),
            scale_y: params
                .get("scale_y")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.scale_y),
            softness: params
                .get("softness")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.softness),
            rotation: params
                .get("rotation")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.rotation),
            offset_x: params
                .get("offset_x")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.offset_x),
            offset_y: params
                .get("offset_y")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.offset_y),
            point_depth: params
                .get("point_depth")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.point_depth),
            curvature_top: params
                .get("curvature_top")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.curvature_top),
            curvature_bottom: params
                .get("curvature_bottom")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.curvature_bottom),
            leaf_width: params
                .get("leaf_width")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.leaf_width),
            curvature_middle: params
                .get("curvature_middle")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.curvature_middle),
            middle_position: params
                .get("middle_position")
                .and_then(mc_core::ParamValue::as_float)
                .unwrap_or(defaults.middle_position),
        }
    }
}

pub fn dispatch(
    gpu: &mut mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    key: &str,
    resolution: u32,
    params: &ShapeParams,
) {
    let shader_source = full_source("shape.wgsl");
    let pipeline = gpu.compute_pipeline(NODE_KIND, &shader_source, "main");

    let param_bytes = pack_params(resolution, params);
    let param_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shape_params"),
        size: param_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&param_buffer, 0, &param_bytes);

    let view = pool.get_or_create(
        &gpu.device,
        key,
        resolution,
        wgpu::TextureFormat::Rgba16Float,
    );

    let bind_group_layout = pipeline.get_bind_group_layout(0);
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("shape_bind_group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: param_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(view),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("shape_encoder"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shape_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let workgroups = resolution.div_ceil(8);
        pass.dispatch_workgroups(workgroups, workgroups, 1);
    }
    gpu.queue.submit(Some(encoder.finish()));
}

fn pack_params(resolution: u32, params: &ShapeParams) -> [u8; 64] {
    let mut bytes = [0u8; 64];
    bytes[0..4].copy_from_slice(&resolution.to_le_bytes());
    bytes[4..8].copy_from_slice(&params.kind.as_u32().to_le_bytes());
    bytes[8..12].copy_from_slice(&params.sides.to_le_bytes());
    bytes[12..16].copy_from_slice(&params.point_depth.to_le_bytes());
    bytes[16..20].copy_from_slice(&params.size.to_le_bytes());
    bytes[20..24].copy_from_slice(&params.softness.to_le_bytes());
    bytes[24..28].copy_from_slice(&params.rotation.to_le_bytes());
    bytes[28..32].copy_from_slice(&params.curvature_top.to_le_bytes());
    bytes[32..36].copy_from_slice(&params.curvature_bottom.to_le_bytes());
    bytes[36..40].copy_from_slice(&params.leaf_width.to_le_bytes());
    bytes[40..44].copy_from_slice(&params.curvature_middle.to_le_bytes());
    bytes[44..48].copy_from_slice(&params.middle_position.to_le_bytes());
    bytes[48..52].copy_from_slice(&params.scale_x.to_le_bytes());
    bytes[52..56].copy_from_slice(&params.scale_y.to_le_bytes());
    bytes[56..60].copy_from_slice(&params.offset_x.to_le_bytes());
    bytes[60..64].copy_from_slice(&params.offset_y.to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::{dispatch, ShapeKind, ShapeParams};
    use crate::test_support::{assert_tiles_seamlessly, RESOLUTION};

    /// Regression test: a Leaf's distance field used to be exactly 0 at
    /// every point on its own vertical axis (p.x = 0), including far above
    /// or below its actual tips — `abs(0)` divided by anything is still 0 —
    /// painting a spurious line of "inside" pixels running the tile's full
    /// height, regardless of curvature or size.
    #[test]
    fn leaf_has_no_spurious_line_above_or_below_its_tips() {
        let r = RESOLUTION as usize;
        let params = ShapeParams {
            kind: ShapeKind::Leaf,
            ..ShapeParams::default()
        };
        let pixels = render(&params);
        let center = r / 2;
        // Default `size` keeps the tips well short of the tile edge, so the
        // very top/bottom rows are definitely past them.
        for y in [0usize, 1, r - 2, r - 1] {
            let on_axis = pixels[(y * r + center) * 4];
            assert!(
                on_axis < 10,
                "pixel ({center}, {y}) is on the leaf's vertical axis but far past its tips, \
                 should be background (got {on_axis}) — looks like the vertical-seam bug"
            );
        }
    }

    /// Regression test: Polygon and Star used to default to an off-kilter
    /// angle (a raw vertex/tip wherever the angle-folding math happened to
    /// start, not lined up with the tile at all) instead of looking upright.
    /// A shape that's actually upright (one point straight up) is symmetric
    /// about the vertical centerline; checking mirrored columns match is a
    /// simple way to confirm that without hand-deriving exact pixel
    /// positions.
    #[test]
    fn polygon_and_star_default_to_an_upright_point_up_orientation() {
        // A dedicated higher resolution, not the shared (and much coarser)
        // `RESOLUTION` test constant: at low resolution, edge rasterization
        // alone produces single-pixel differences between mirrored columns
        // even at the mathematically exact upright angle, which would make
        // a strict per-pixel check flaky. Summing many probes' differences
        // and checking the *total* against a generous threshold is what
        // actually distinguishes "upright" from "tilted" — at the correct
        // angle this totals a few thousand (antialiasing noise); at a wrong
        // angle it's tens of thousands (see the commit this test landed in).
        let res = 256u32;
        let r = res as usize;
        let center = r / 2;
        for kind in [ShapeKind::Polygon, ShapeKind::Star] {
            let params = ShapeParams {
                kind,
                ..ShapeParams::default()
            };
            let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
            let mut pool = mc_gpu::TexturePool::default();
            dispatch(&mut gpu, &mut pool, "test", res, &params);
            let pixels = gpu.read_rgba8(pool.texture("test").unwrap(), res);

            let mut total_diff: i64 = 0;
            for dx in [center / 8, center / 4, center / 2, center * 3 / 4] {
                for y in 0..r {
                    let left = pixels[(y * r + (center - dx)) * 4] as i64;
                    let right = pixels[(y * r + (center + dx).min(r - 1)) * 4] as i64;
                    total_diff += (left - right).abs();
                }
            }
            assert!(
                total_diff < 10_000,
                "{kind:?}: not symmetric about the vertical centerline (total mirrored-pixel \
                 difference {total_diff}, expected a few thousand at most) — looks tilted, \
                 not upright"
            );
        }
    }

    /// Regression test: moving `middle_position` away from 0.5 must move the
    /// widest point of the outline up or down with it, while keeping both
    /// tips at their intended height (not stretching the shape past `size`).
    #[test]
    fn leaf_middle_position_moves_the_equator() {
        let r = RESOLUTION as usize;
        let center = r / 2;

        // The widest row (max x reaching whichever lit pixel is furthest
        // right) should move toward the top when `middle_position` is low.
        let widest_row = |middle_position: f32| -> usize {
            let params = ShapeParams {
                kind: ShapeKind::Leaf,
                middle_position,
                ..ShapeParams::default()
            };
            let pixels = render(&params);
            (0..r)
                .max_by_key(|&y| {
                    (0..r)
                        .rev()
                        .find(|&x| pixels[(y * r + x) * 4] > 128)
                        .unwrap_or(0)
                })
                .unwrap()
        };
        assert!(
            widest_row(0.2) < widest_row(0.5),
            "a low middle_position should put the widest row above center"
        );
        assert!(
            widest_row(0.8) > widest_row(0.5),
            "a high middle_position should put the widest row below center"
        );

        // Both tips should still reach roughly the same height regardless
        // of where the equator sits — a point near the very top and the
        // very bottom edge of the shape's extent should both be lit.
        let params = ShapeParams {
            kind: ShapeKind::Leaf,
            middle_position: 0.2,
            ..ShapeParams::default()
        };
        let pixels = render(&params);
        let top_tip = pixels[(((center as f32 - 0.28 * r as f32) as usize) * r + center) * 4];
        let bottom_tip = pixels[(((center as f32 + 0.28 * r as f32) as usize) * r + center) * 4];
        assert!(
            top_tip > 100 && bottom_tip > 100,
            "both tips should still reach close to the shape's full height, got \
             top={top_tip} bottom={bottom_tip}"
        );
    }

    /// Regression test: Leaf's top and bottom tips must be independently
    /// shapeable. `curvature_top` low (round) + `curvature_bottom` high
    /// (pointed) should measurably widen the silhouette near the top at a
    /// given distance from its tip compared to the same distance from the
    /// bottom tip.
    #[test]
    fn leaf_curvature_top_and_bottom_are_independent() {
        let r = RESOLUTION as usize;
        let params = ShapeParams {
            kind: ShapeKind::Leaf,
            curvature_top: 0.4,
            curvature_bottom: 3.0,
            ..ShapeParams::default()
        };
        let pixels = render(&params);
        let center = r / 2;
        // Close to each tip (not just partway there) — the exponent that
        // sets top vs. bottom apart is blended in gradually from the middle
        // exponent at the equator up to each tip's own, so the two only
        // diverge clearly near the tips themselves (see `leaf_dist`'s doc
        // comment on the three-way blend).
        let offset = r / 4;
        let probe_x = center + r / 20;
        let top = pixels[((center - offset) * r + probe_x) * 4] as i32;
        let bottom = pixels[((center + offset) * r + probe_x) * 4] as i32;
        assert!(
            top > bottom + 50,
            "round top (curvature 0.4) should be wider than pointed bottom (curvature 3.0) \
             close to its tip, got top={top} bottom={bottom}"
        );
    }

    /// Regression test: `curvature_middle` shapes the sides through the
    /// equator independently from `curvature_top`/`curvature_bottom`, which
    /// only take over near each tip. Right at the equator itself, changing
    /// only the middle curvature (top/bottom held fixed) must still change
    /// the outline.
    #[test]
    fn leaf_curvature_middle_shapes_the_equator_independently() {
        let r = RESOLUTION as usize;
        let center = r / 2;
        let probe_x = center + r / 8;
        let render_at_equator = |curvature_middle: f32| -> i32 {
            let params = ShapeParams {
                kind: ShapeKind::Leaf,
                curvature_middle,
                ..ShapeParams::default()
            };
            render(&params)[(center * r + probe_x) * 4] as i32
        };
        let round = render_at_equator(0.2);
        let pointed = render_at_equator(4.0);
        assert!(
            round > pointed + 50,
            "a rounder middle (curvature 0.2) should be wider at the equator than a pointier \
             one (curvature 4.0), got round={round} pointed={pointed}"
        );
    }

    /// Regression test: an earlier version of Leaf stitched together a
    /// separate width-taper curve for each half with `max()`, which kinks
    /// wherever the two pieces cross — visible as a crease right at the
    /// equator (the widest point) even with very different top/bottom
    /// curvature, and another one near each tip. Checking that the outline
    /// moves roughly the same amount between each pair of adjacent rows
    /// around the equator (no single row where it suddenly jumps) is a
    /// simple way to confirm the boundary is one smooth curve there instead
    /// of two different ones glued together.
    #[test]
    fn leaf_has_no_crease_at_the_equator() {
        // A dedicated higher resolution, not the shared (much coarser)
        // `RESOLUTION` test constant — at low resolution, a few pixels of
        // rasterization wobble near the equator is normal and isn't a real
        // crease, which made this flaky at `RESOLUTION`'s scale.
        let res = 256u32;
        let r = res as usize;
        let params = ShapeParams {
            kind: ShapeKind::Leaf,
            curvature_top: 4.0,
            curvature_bottom: 0.2,
            ..ShapeParams::default()
        };
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", res, &params);
        let pixels = gpu.read_rgba8(pool.texture("test").unwrap(), res);
        let center = r / 2;

        // The boundary's x position (rightmost lit pixel) in each row near
        // the equator.
        let edge_x = |y: usize| -> i64 {
            (0..r)
                .rev()
                .find(|&x| pixels[(y * r + x) * 4] > 128)
                .map_or(center as i64, |x| x as i64)
        };
        let step = |y: usize| (edge_x(y + 1) - edge_x(y)).abs();

        let steps: Vec<i64> = (center - 3..center + 2).map(step).collect();
        let max_step = *steps.iter().max().unwrap();
        // A generous bound, not a tight one — at this test's modest
        // resolution a couple of pixels either way is just rasterization,
        // not a visible crease (confirmed by eye at a much higher
        // resolution when this test was written). What it catches is the
        // old bug's scale: a `max()` of two mismatched curves jumped by
        // dozens of pixels right at the seam, not a handful.
        assert!(
            max_step <= r as i64 / 8,
            "the boundary jumps sideways far more between one pair of rows than its \
             neighbors near the equator (steps: {steps:?}) — looks like a crease, not a \
             smooth curve"
        );
    }

    fn render(params: &ShapeParams) -> Vec<u8> {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        dispatch(&mut gpu, &mut pool, "test", RESOLUTION, params);
        let texture = pool.texture("test").expect("shape texture");
        gpu.read_rgba8(texture, RESOLUTION)
    }

    /// A shape sits fully inside the tile on a flat background, so it
    /// trivially tiles — background meets background at every edge.
    #[test]
    fn tiles_seamlessly_for_every_kind() {
        for kind in [
            ShapeKind::Disc,
            ShapeKind::Square,
            ShapeKind::Polygon,
            ShapeKind::Star,
            ShapeKind::Leaf,
        ] {
            let params = ShapeParams {
                kind,
                ..ShapeParams::default()
            };
            assert_tiles_seamlessly(&render(&params), RESOLUTION);
        }
    }

    #[test]
    fn center_is_inside_every_kind() {
        let r = RESOLUTION as usize;
        for kind in [
            ShapeKind::Disc,
            ShapeKind::Square,
            ShapeKind::Polygon,
            ShapeKind::Star,
            ShapeKind::Leaf,
        ] {
            let params = ShapeParams {
                kind,
                ..ShapeParams::default()
            };
            let pixels = render(&params);
            let center = pixels[(r / 2 * r + r / 2) * 4];
            assert!(
                center > 200,
                "{kind:?}: center should be inside the shape, got {center}"
            );
        }
    }

    #[test]
    fn corner_is_outside_every_kind() {
        let pixels_for = |kind| {
            let params = ShapeParams {
                kind,
                ..ShapeParams::default()
            };
            render(&params)
        };
        for kind in [
            ShapeKind::Disc,
            ShapeKind::Square,
            ShapeKind::Polygon,
            ShapeKind::Star,
            ShapeKind::Leaf,
        ] {
            let pixels = pixels_for(kind);
            let corner = pixels[0];
            assert!(
                corner < 50,
                "{kind:?}: corner should be outside the shape, got {corner}"
            );
        }
    }

    #[test]
    fn is_deterministic() {
        let params = ShapeParams {
            kind: ShapeKind::Star,
            rotation: 0.2,
            ..ShapeParams::default()
        };
        assert_eq!(
            render(&params),
            render(&params),
            "same params produced different pixels"
        );
    }

    #[test]
    fn kind_name_round_trips() {
        for name in ShapeKind::NAMES {
            assert!(
                ShapeKind::from_name(name).is_some(),
                "{name:?} should parse"
            );
        }
    }

    /// Size's own UI minimum is 0.01 (defs.rs) — every kind should still
    /// dispatch cleanly at that floor, not just at the old 0.1 minimum.
    #[test]
    fn renders_without_panicking_at_minimum_size() {
        for kind in [
            ShapeKind::Disc,
            ShapeKind::Square,
            ShapeKind::Polygon,
            ShapeKind::Star,
            ShapeKind::Leaf,
        ] {
            let params = ShapeParams {
                kind,
                size: 0.01,
                ..ShapeParams::default()
            };
            render(&params);
        }
    }

    #[test]
    fn default_softness_is_zero() {
        assert_eq!(ShapeParams::default().softness, 0.0);
    }

    /// Horizontal extent (inside-pixel count along the center row) of a
    /// `scale_x = 2.0` Disc should end up roughly twice its own vertical
    /// extent (inside-pixel count along the center column, `scale_y`
    /// untouched) — a real ellipse, not just a bigger circle.
    #[test]
    fn scale_x_stretches_a_disc_into_an_ellipse() {
        let r = RESOLUTION as usize;
        let params = ShapeParams {
            kind: ShapeKind::Disc,
            size: 0.3,
            scale_x: 2.0,
            ..ShapeParams::default()
        };
        let pixels = render(&params);

        let horizontal_extent = (0..r)
            .filter(|&x| pixels[(r / 2 * r + x) * 4] > 128)
            .count();
        let vertical_extent = (0..r)
            .filter(|&y| pixels[(y * r + r / 2) * 4] > 128)
            .count();

        let ratio = horizontal_extent as f32 / vertical_extent as f32;
        assert!(
            (1.7..=2.3).contains(&ratio),
            "scale_x = 2.0 should roughly double the horizontal extent relative to vertical \
             (horizontal={horizontal_extent}, vertical={vertical_extent}, ratio={ratio})"
        );
    }

    /// Same idea as the Disc case, but for Square — confirms the stretch
    /// is applied uniformly across shape kinds (via the shared input
    /// coordinate every kind's own distance field reads), not special-
    /// cased just for Disc.
    #[test]
    fn scale_y_stretches_a_square_into_a_rectangle() {
        let r = RESOLUTION as usize;
        let params = ShapeParams {
            kind: ShapeKind::Square,
            size: 0.3,
            scale_y: 2.0,
            ..ShapeParams::default()
        };
        let pixels = render(&params);

        let horizontal_extent = (0..r)
            .filter(|&x| pixels[(r / 2 * r + x) * 4] > 128)
            .count();
        let vertical_extent = (0..r)
            .filter(|&y| pixels[(y * r + r / 2) * 4] > 128)
            .count();

        let ratio = vertical_extent as f32 / horizontal_extent as f32;
        assert!(
            (1.7..=2.3).contains(&ratio),
            "scale_y = 2.0 should roughly double the vertical extent relative to horizontal \
             (horizontal={horizontal_extent}, vertical={vertical_extent}, ratio={ratio})"
        );
    }

    /// A stretched ellipse rotated 90 degrees should swap its own extents
    /// — proof the stretch happens in the shape's own local space before
    /// `rotation`, not fixed to the image's x/y axes afterward.
    #[test]
    fn stretch_rotates_with_the_shape() {
        let r = RESOLUTION as usize;
        let params = ShapeParams {
            kind: ShapeKind::Disc,
            size: 0.3,
            scale_x: 2.0,
            rotation: 0.25, // a quarter turn
            ..ShapeParams::default()
        };
        let pixels = render(&params);

        let horizontal_extent = (0..r)
            .filter(|&x| pixels[(r / 2 * r + x) * 4] > 128)
            .count();
        let vertical_extent = (0..r)
            .filter(|&y| pixels[(y * r + r / 2) * 4] > 128)
            .count();

        let ratio = vertical_extent as f32 / horizontal_extent as f32;
        assert!(
            (1.7..=2.3).contains(&ratio),
            "a scale_x = 2.0 ellipse rotated a quarter turn should now be tall, not wide \
             (horizontal={horizontal_extent}, vertical={vertical_extent}, ratio={ratio})"
        );
    }

    /// `offset_x`/`offset_y` move the shape's own center away from the
    /// tile's — the default center should go from shape to background, and
    /// the new center (tile center + offset) should go from background to
    /// shape.
    #[test]
    fn offset_moves_the_shape_away_from_the_tile_center() {
        let r = RESOLUTION as usize;
        let params = ShapeParams {
            kind: ShapeKind::Disc,
            size: 0.3,
            offset_x: 0.25,
            offset_y: 0.0,
            ..ShapeParams::default()
        };
        let pixels = render(&params);
        let tile_center = pixels[(r / 2 * r + r / 2) * 4];
        let new_center_x = (r / 2 + r / 4).min(r - 1);
        let shifted_center = pixels[(r / 2 * r + new_center_x) * 4];
        assert!(
            tile_center < 10,
            "the tile's own center should now be background, got {tile_center}"
        );
        assert!(
            shifted_center > 200,
            "offset_x = 0.25 should put the shape a quarter-tile to the right, got \
             {shifted_center}"
        );
    }

    /// An offset past 0.5 wraps instead of sliding the shape off the tile —
    /// the shape should reappear on the opposite edge, keeping this
    /// seamless with no separate Transform 2D node.
    #[test]
    fn offset_wraps_past_the_tile_edge() {
        let no_offset = ShapeParams {
            kind: ShapeKind::Disc,
            size: 0.3,
            ..ShapeParams::default()
        };
        let want = render(&no_offset);

        let wrapped = ShapeParams {
            kind: ShapeKind::Disc,
            size: 0.3,
            offset_x: 1.0,
            offset_y: 1.0,
            ..ShapeParams::default()
        };
        let got = render(&wrapped);

        assert_eq!(
            got, want,
            "offsetting by exactly one full tile should wrap back to an identical image"
        );
    }

    #[test]
    fn offset_preserves_seamless_tiling() {
        let params = ShapeParams {
            kind: ShapeKind::Disc,
            size: 0.3,
            offset_x: 0.4,
            offset_y: 0.35,
            ..ShapeParams::default()
        };
        let pixels = render(&params);
        assert_tiles_seamlessly(&pixels, RESOLUTION);
    }

    /// Regression test for a real sizing bug found via hands-on testing
    /// (the owner's own report: "the Star/Triangle shape is not exactly
    /// centered"): `polygon_dist`'s own internal angle convention puts its
    /// fold-zero at the edge-midpoint (apothem foot), not the vertex, so
    /// the shape's *true* vertex used to overshoot `Size` by a factor of
    /// `1 / cos(π/sides)` — exactly 2x for a triangle, large enough to push
    /// the apex off-canvas (clipped against the tile edge), which reads as
    /// "not centered." Checked directly: the topmost lit pixel of an
    /// upright triangle at a known `Size` must land within half a pixel of
    /// the mathematically exact vertex position (`0.5 - Size/2`), not the
    /// old, doubled distance. Star's own math already used the opposite
    /// (already-correct) convention — confirmed separately before fixing
    /// anything, by computing each rendered shape's real pixel-weighted
    /// centroid at several resolutions: Star's offset from (0.5, 0.5)
    /// shrank toward zero as resolution increased (ordinary discretization
    /// noise), while Polygon sides=3's did not (a genuine, resolution-
    /// independent bias) — so only `polygon_dist` needed a fix.
    #[test]
    fn polygon_vertex_reaches_exactly_the_configured_size_not_doubled() {
        let resolution = 1024u32;
        for &sides in &[3u32, 4, 5, 6, 8] {
            let size = 0.6f32;
            let params = ShapeParams {
                kind: ShapeKind::Polygon,
                sides,
                size,
                softness: 0.0,
                ..ShapeParams::default()
            };
            let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
            let mut pool = mc_gpu::TexturePool::default();
            dispatch(&mut gpu, &mut pool, "test", resolution, &params);
            let pixels = gpu.read_rgba8(pool.texture("test").unwrap(), resolution);
            let r = resolution as usize;
            let center_col = r / 2;
            let top_lit_row = (0..r)
                .find(|&y| pixels[(y * r + center_col) * 4] > 128)
                .expect("the shape should have a visible top edge somewhere");
            let measured_uv = (top_lit_row as f32 + 0.5) / resolution as f32;
            let expected_uv = 0.5 - size / 2.0;
            assert!(
                (measured_uv - expected_uv).abs() < 2.0 / resolution as f32,
                "sides={sides}: vertex measured at uv={measured_uv:.4}, expected ~{expected_uv:.4} \
                 (size={size}) — a vertex overshooting this means the old apothem/circumradius \
                 mixup is back"
            );
        }
    }
}
