// SPDX-License-Identifier: MIT OR Apache-2.0
// shape.wgsl — Generators / Patterns / Shape
//
// Inputs:  none.
// Output:  `output` — a single shape centered in the tile, white inside and
//          black outside with a soft edge. Trivially tileable: the shape
//          sits fully inside the tile on a flat background, so neighboring
//          tiles meet background-to-background with no seam.
// Params:
//   kind        — 0 Disc, 1 Square, 2 Polygon, 3 Star, 4 Leaf
//   sides       — vertex/point count for Polygon and Star
//   point_depth      — Star only: 1.0..0.05, how deep the points cut in. At
//                      1.0 the star's points vanish and it's exactly a
//                      Polygon of the same side count (a `sides = 3` Star is
//                      then a plain triangle); lowering it unfolds that same
//                      outline into sharper star points. Straight
//                      tip-to-valley edges, not a smoothed curve — see
//                      `star_radius` below.
//   leaf_width       — Leaf only: how wide it is at its fattest point (the
//                      equator), as a fraction of its own height.
//   curvature_top    — Leaf only: how pointed (vs. rounded) its top tip is.
//   curvature_bottom — Leaf only: same, for its bottom tip — independently
//                      controllable from the top, see `leaf_dist` below.
//   curvature_middle — Leaf only: how pointed (vs. rounded) its sides are
//                      through the middle (the equator), independently from
//                      either tip — see `leaf_dist` below.
//   middle_position  — Leaf only: 0.15..0.85, where the equator (widest
//                      point) sits between the top tip (0) and bottom tip
//                      (1). 0.5 is centered (equal top/bottom halves).
//   size             — how large the shape is (0..1, fraction of the tile)
//   scale_x          — independent X stretch on top of `size`, applied
//                      before `rotation` (so a stretched-then-rotated
//                      ellipse/rectangle tilts as a whole). 1.0 = no
//                      stretch.
//   scale_y          — same, for Y.
//   softness         — how soft the edge is (0 = hard edge)
//   rotation         — 0..1, fraction of a full turn, on top of each kind's
//                      own upright default (a Polygon/Star point straight
//                      up, not at some arbitrary angle — see `BASE_OFFSET`)

struct ShapeParams {
    resolution: u32,
    kind: u32,
    sides: u32,
    point_depth: f32,
    size: f32,
    softness: f32,
    rotation: f32,
    curvature_top: f32,
    curvature_bottom: f32,
    leaf_width: f32,
    curvature_middle: f32,
    middle_position: f32,
    scale_x: f32,
    scale_y: f32,
    offset_x: f32,
    offset_y: f32,
}

@group(0) @binding(0) var<uniform> params: ShapeParams;
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, write>;

fn rotate(p: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

// Distance to a regular N-gon's edge along the radial direction — a cheap,
// commonly-used approximation (not an exact SDF), good enough for a mask.
//
// The fold's own zero (`a = 0`) lands on the edge-midpoint (apothem foot)
// direction, not the vertex — `length(p) * cos(a)` alone would therefore
// reach the comparison threshold (`edge` in `main`'s own smoothstep) at the
// *apothem* when `length(p) == edge`, meaning the true vertex (at
// `a = sector/2`) wouldn't reach that same threshold until
// `length(p) == edge / cos(sector/2)` — a real, measured bug (found via
// hands-on testing, not just reasoned about): for a triangle that's exactly
// a 2x overshoot, large enough to push the apex off the tile entirely,
// which read as "not centered" rather than "slightly oversized." Dividing
// by `cos(sector/2)` rescales so the vertex reaches exactly `edge` (matching
// `size`'s own "how large" semantic, the same as every other shape kind
// here) and the apothem correctly sits at `edge * cos(sector/2)`, smaller,
// as a real regular polygon's apothem should be.
fn polygon_dist(p: vec2<f32>, sides: f32) -> f32 {
    let angle = atan2(p.y, p.x);
    let sector = 6.283185307 / sides;
    let a = ((angle % sector) + sector) % sector - sector * 0.5;
    return length(p) * cos(a) / cos(sector * 0.5);
}

// The star's target radius at angle `p`'s direction, as an exact straight
// line (in polar form) through the sector's tip (radius 1, angle 0) and its
// neighboring valley (radius `point_depth`, angle `half_sector`) — not a
// smoothed blend between the two, which is what made the old star look like
// a rounded flower instead of having sharp points. At `point_depth = 1.0`
// the tip and valley radii match, so this collapses to a plain N-gon (e.g. a
// triangle at `sides = 3`) — lowering `point_depth` "unfolds" that same
// outline into sharper star points.
fn star_radius(p: vec2<f32>, sides: f32, point_depth: f32) -> f32 {
    let angle = atan2(p.y, p.x);
    let sector = 6.283185307 / sides;
    let half_sector = sector * 0.5;
    let a = abs(((angle % sector) + sector) % sector - half_sector);
    let inner = clamp(point_depth, 0.05, 1.0);
    let coef_a = inner * sin(half_sector);
    let coef_b = 1.0 - inner * cos(half_sector);
    return coef_a / (coef_a * cos(a) + coef_b * sin(a));
}

// Leaf boundary: a superellipse per half — |x/waist|^n + |y/edge|^n = 1 at
// the boundary — rather than a separate width-taper curve stitched to a
// patched-on "past the tip" fallback. That earlier version pieced two
// different formulas together with `max()`, which kinks whichever one
// dominates where they cross, producing a visible crease near each tip; a
// superellipse is a single smooth formula across the whole shape, so there's
// nothing to stitch. `n` is the exponent: n = 2 is a plain ellipse (the
// roundest this gets), and n < 2 pinches the axis crossing into a
// progressively sharper cusp. (n > 2 would flatten it toward a boxy,
// almost-rectangular cap instead — not what "rounder" should mean here, so
// `curvature` never maps past n = 2.) Each `curvature` param (the
// artist-facing value, high = pointed) maps linearly onto its own exponent
// over just the n <= 2 half: min..max becomes n = 2..0.25.
//
// `n` isn't fixed across a whole half, though — it's itself blended from
// `curvature_middle`'s exponent at the equator (`middle_position` between
// the two tips, not necessarily halfway) up to that half's own tip exponent
// (`curvature_top`/`curvature_bottom`) at its tip, by height fraction. That
// gives three independent controls — how the sides bulge through the middle
// vs. how sharp each individual tip is — while staying one continuous
// formula: right at the equator the blend weight is 0 on *both* sides
// regardless of which tip exponent they'd otherwise head toward, so top and
// bottom still meet using the exact same (middle) exponent there, same as
// every other point on this curve — there's still nothing stitched
// together, just a smoothly varying exponent instead of a fixed one.
fn leaf_exponent(curvature: f32) -> f32 {
    let curvature_min = 0.2;
    let curvature_max = 4.0;
    let unit = clamp(
        (curvature - curvature_min) / (curvature_max - curvature_min),
        0.0,
        1.0,
    );
    return mix(2.0, 0.25, unit);
}

fn leaf_dist(
    p: vec2<f32>,
    edge: f32,
    width: f32,
    curvature_top: f32,
    curvature_bottom: f32,
    curvature_middle: f32,
    middle_position: f32,
) -> f32 {
    let waist = edge * width;
    // Where the equator sits between the top tip (-edge) and bottom tip
    // (+edge) — 0.5 is centered, matching the original fixed-at-0 behavior
    // exactly.
    let equator_y = mix(-edge, edge, clamp(middle_position, 0.05, 0.95));
    let is_top = p.y < equator_y;
    let tip_exponent = leaf_exponent(select(curvature_bottom, curvature_top, is_top));
    let middle_exponent = leaf_exponent(curvature_middle);
    // Each half is measured against its own distance from the equator to
    // its own tip, not a shared `edge` — so moving the equator stretches
    // one half and compresses the other without otherwise changing their
    // shape, instead of (e.g.) the bottom tip creeping past where `size`
    // says the shape should end.
    let half_height = select(edge - equator_y, equator_y + edge, is_top);
    let ax = abs(p.x) / max(waist, 0.0001);
    let ay = abs(p.y - equator_y) / max(half_height, 0.0001);
    let n = mix(middle_exponent, tip_exponent, clamp(ay, 0.0, 1.0));
    let field = pow(ax, n) + pow(ay, n);
    return edge * pow(field, 1.0 / n);
}

// A per-kind rotation baked in on top of `params.rotation`, so a Polygon or
// Star appears upright (a point straight up) at the default rotation = 0
// instead of at whatever arbitrary angle its raw angle-folding math happens
// to start from. Both `polygon_dist` and `star_radius` fold the raw angle
// into one sector and compare it against `sector / 2`, but a Polygon vertex
// sits where that fold hits 0 while a Star tip sits where it hits
// `sector / 2` — so the two needed offsets differ by half a sector, not the
// same amount (verified empirically by rendering and checking for left-right
// symmetry, not just derived on paper — see the PR/commit this landed in).
fn base_offset(kind: u32, sides: f32) -> f32 {
    let quarter_turn = 1.5707963;
    let sector = 6.283185307 / sides;
    switch kind {
        case 2u: {
            // Polygon
            return quarter_turn;
        }
        case 3u: {
            // Star
            return quarter_turn + sector * 0.5;
        }
        default: {
            return 0.0;
        }
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.resolution || id.y >= params.resolution) {
        return;
    }
    let uv = (vec2<f32>(id.xy) + 0.5) / f32(params.resolution);
    // Offset wraps (fract), not clamps, so a shape pushed past one edge
    // reappears on the opposite one instead of sliding off — keeps this
    // seamless without a separate Transform 2D node just to reposition it.
    let shifted = fract(uv - vec2<f32>(params.offset_x, params.offset_y));
    var p = shifted - vec2<f32>(0.5, 0.5);
    let sides_f = f32(params.sides);
    // Rotate the world-space query point into the shape's own local frame
    // first (the existing, already-correct step every kind's rotation
    // already relied on), *then* stretch — not the other way around. For
    // Disc specifically, stretching before rotating would be invisible:
    // `length(rotate(v, angle)) == length(v)` for any angle (rotation
    // preserves vector length), so a stretch-then-rotate ellipse's
    // apparent orientation could never actually change with Rotation.
    // Stretching in the already-local (post-rotate) frame instead makes
    // the stretch's own axes rotate along with the shape, as expected.
    p = rotate(p, params.rotation * 6.283185307 + base_offset(params.kind, sides_f));
    p = p / vec2<f32>(max(params.scale_x, 0.0001), max(params.scale_y, 0.0001));

    let edge = params.size * 0.5;
    let softness = max(params.softness * 0.5, 0.0005);

    var dist: f32;
    switch params.kind {
        case 0u: {
            // Disc
            dist = length(p);
        }
        case 1u: {
            // Square
            dist = max(abs(p.x), abs(p.y));
        }
        case 2u: {
            // Polygon
            dist = polygon_dist(p, sides_f);
        }
        case 3u: {
            // Star — see `star_radius`'s doc comment.
            dist = length(p) / star_radius(p, sides_f, params.point_depth);
        }
        default: {
            // Leaf — see `leaf_dist`'s doc comment.
            dist = leaf_dist(
                p,
                edge,
                params.leaf_width,
                params.curvature_top,
                params.curvature_bottom,
                params.curvature_middle,
                params.middle_position,
            );
        }
    }

    let value = 1.0 - smoothstep(edge - softness, edge + softness, dist);
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(value, value, value, 1.0));
}
