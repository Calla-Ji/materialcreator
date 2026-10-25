// SPDX-License-Identifier: MIT OR Apache-2.0
//! Converters / Distance Field.
//!
//! Turns a greyscale mask into an exact signed distance field of its 0.5
//! iso-line (`docs/hardsurface_nodes/HARDSURFACE_NODES.md` §3.1) — ported
//! field-for-field from the hand-off package's own ground truth,
//! `hs_reference.py`'s `distance_field` function.
//!
//! **CPU-only for now** — see [`dispatch`]'s own doc comment for why, and
//! why that's a deliberate, temporary step rather than this project's
//! general node-evaluation model.

use std::collections::BTreeMap;

use mc_core::ParamValue;

pub struct DistanceFieldParams {
    /// Whether the mask wraps (torus) — distances stay correct up to 0.5
    /// tile across the seam when true. Off treats the mask as a single,
    /// non-repeating island (edge-clamped neighbors/gradients).
    pub wrap: bool,
    /// Shifts which mask value counts as "the edge" — `0.5` is the normal
    /// anti-aliased-mask convention everywhere else in this codebase.
    pub threshold: f32,
    /// Clamp the output to `+-max_distance` tile units. `0.0` (the
    /// default) means no clamp — not a meaningful clamp value on its own
    /// (collapsing everything within the edge band to a single distance
    /// isn't useful), so it doubles safely as the "off" sentinel without a
    /// separate toggle.
    pub max_distance: f32,
}

impl Default for DistanceFieldParams {
    fn default() -> Self {
        Self {
            wrap: true,
            threshold: 0.5,
            max_distance: 0.0,
        }
    }
}

impl DistanceFieldParams {
    pub fn from_graph(params: &BTreeMap<String, ParamValue>) -> Self {
        let defaults = Self::default();
        Self {
            wrap: params
                .get("wrap")
                .and_then(ParamValue::as_bool)
                .unwrap_or(defaults.wrap),
            threshold: params
                .get("threshold")
                .and_then(ParamValue::as_float)
                .unwrap_or(defaults.threshold),
            max_distance: params
                .get("max_distance")
                .and_then(ParamValue::as_float)
                .unwrap_or(defaults.max_distance),
        }
    }
}

/// Reads `mask_key`'s R channel back to the CPU, runs the exact distance
/// transform in plain Rust, and uploads the result to `output_key`.
///
/// Every other node in this codebase evaluates as a pure GPU compute
/// dispatch (CLAUDE.md §3) — this one breaks that, deliberately and
/// temporarily. The hand-off's own build order (§5) splits this node into
/// two gated stages on purpose: get the algorithm byte-exact against the
/// real goldens on the CPU first (this function), then port it to the GPU
/// via jump flooding (JFA+1) as a separate, later step that has to match
/// these same goldens. The CPU round-trip (`read_rgba16f` blocks on the
/// GPU, then `write_rgba16f` uploads back) only runs when this node's own
/// inputs are dirty, same as any other node's cache — slower per
/// evaluation, not slower per frame. This is never meant to be the shipped
/// end state; the GPU version replaces it as the real live-editing path.
pub fn dispatch(
    gpu: &mc_gpu::Context,
    pool: &mut mc_gpu::TexturePool,
    mask_key: &str,
    output_key: &str,
    resolution: u32,
    params: &DistanceFieldParams,
) {
    let texture = pool
        .texture(mask_key)
        .unwrap_or_else(|| panic!("distance_field: no texture at mask key {mask_key:?}"));
    let raw = gpu.read_rgba16f(texture, resolution);
    let mask: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|p| p[0]).collect();

    let sdf = compute_distance_field(&mask, resolution as usize, params);

    // HARDSURFACE_NODES.md §2: "Storage: float32" for the `Sdf` socket —
    // `Rgba16Float`'s ~3-4 significant decimal digits is coarse enough to
    // read back as a visibly wrong sdf once a downstream node (Shape
    // Boolean) compares against a tight, pure-math golden.
    let pixels: Vec<[f32; 4]> = sdf.iter().map(|&d| [d, d, d, 1.0]).collect();
    gpu.write_rgba32f(pool, output_key, resolution, &pixels);
}

/// A cost standing in for "unreachable" in the 1D distance transform below.
/// Real `f32::INFINITY` works almost everywhere, but a column/row with *no*
/// feature pixel at all produces `INFINITY - INFINITY` (`NaN`) in the
/// envelope intersection math — a known sharp edge of the Felzenszwalb–
/// Huttenlocher algorithm's usual infinity-based presentation. A very large
/// *finite* sentinel (astronomically bigger than any real squared distance
/// this ever computes, `res` is at most a few thousand) behaves identically
/// for every comparison that matters here, without ever producing `NaN`.
const UNREACHABLE: f32 = 1e20;

/// `hs_reference.py`'s own `coverage()` (§2's anti-aliasing convention):
/// anti-aliased inside-coverage of an SDF value — 1 inside, 0 outside, a
/// 1-pixel ramp at the edge. Shared by `redistance` (`redistance.rs`,
/// `redistance(sdf) = distance_field(coverage(sdf))`) and Shape Boolean's
/// own `Exact` option, both of which turn an SDF back into a mask this
/// module's own [`compute_distance_field`] can re-run the exact transform
/// on.
pub(crate) fn coverage(sdf: f32, res: f32) -> f32 {
    (0.5 - sdf * res).clamp(0.0, 1.0)
}

/// Ports `hs_reference.py`'s `distance_field` (lines 106–184) exactly:
/// edge-pixel detection (ramp + hard-crossing), subpixel edge points along
/// the mask's own gradient, an exact 2D Euclidean distance transform to the
/// nearest edge *pixel*, then a 3x3 refinement against that pixel's
/// neighbors' subpixel points. `mask` is row-major, `res*res` long, values
/// in roughly `0..1` (an anti-aliased mask's usual range, though nothing
/// here requires it to be strictly clamped).
///
/// `pub(crate)`, not just a `distance_field` dispatch internal: `redistance`
/// and Shape Boolean's `Exact` option both need to re-run this exact same
/// transform on a coverage mask derived from an SDF, not just on a
/// genuinely painted/imported mask.
pub(crate) fn compute_distance_field(
    mask: &[f32],
    res: usize,
    params: &DistanceFieldParams,
) -> Vec<f32> {
    // `m` stays the mask's own, unshifted values throughout: ramp-pixel
    // detection and the gradient are properties of the mask's own
    // anti-aliasing, independent of where the inside/outside line is drawn
    // (a hard 0/1 mask has no real ramp regardless of threshold). Only the
    // inside test and the subpixel-offset *target* below actually care
    // about `threshold` — shifting the whole mask by a constant instead
    // (an earlier version of this function did) corrupts the ramp test for
    // any threshold != 0.5: every background `0.0` pixel becomes a small
    // positive value, which falls inside the `0 < x < 1` ramp range and
    // gets misread as an edge pixel everywhere, not just near the real one.
    let m = mask;
    let inside: Vec<bool> = m.iter().map(|&v| v >= params.threshold).collect();
    let wrap = params.wrap;

    let at = |x: i64, y: i64| -> usize {
        if wrap {
            let xi = x.rem_euclid(res as i64) as usize;
            let yi = y.rem_euclid(res as i64) as usize;
            yi * res + xi
        } else {
            let xi = x.clamp(0, res as i64 - 1) as usize;
            let yi = y.clamp(0, res as i64 - 1) as usize;
            yi * res + xi
        }
    };
    let in_bounds = |x: i64, y: i64| -> bool {
        wrap || (x >= 0 && y >= 0 && (x as usize) < res && (y as usize) < res)
    };

    // Crossing pixels: a pixel whose any 4-neighbor is on the other side of
    // `inside`. Non-wrap neighbors that fall off the edge just repeat this
    // pixel's own side (never a crossing there), matching the reference's
    // own `mode="edge"` padding for the non-wrap case.
    let mut crossing = vec![false; res * res];
    for y in 0..res {
        for x in 0..res {
            let here = inside[y * res + x];
            let mut cross = false;
            for &(dx, dy) in &[(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                if in_bounds(nx, ny) && inside[at(nx, ny)] != here {
                    cross = true;
                    break;
                }
            }
            crossing[y * res + x] = cross;
        }
    }

    // Ramp pixels: the anti-aliased edge itself says where the boundary is.
    let frac: Vec<bool> = m.iter().map(|&v| v > 1e-3 && v < 1.0 - 1e-3).collect();

    // `near_frac`: `frac` dilated into its own 3x3 neighborhood — crossing
    // pixels only count as edge pixels where there's no nearby ramp pixel
    // already covering that edge (hard binary masks only).
    let mut near_frac = vec![false; res * res];
    for y in 0..res {
        for x in 0..res {
            let mut any = false;
            'block: for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                    if in_bounds(nx, ny) && frac[at(nx, ny)] {
                        any = true;
                        break 'block;
                    }
                }
            }
            near_frac[y * res + x] = any;
        }
    }

    let edge: Vec<bool> = (0..res * res)
        .map(|i| frac[i] || (crossing[i] && !near_frac[i]))
        .collect();

    // Gradient via central differences, and the subpixel edge point each
    // edge pixel's own value/gradient implies.
    let mut ex = vec![0.0f32; res * res];
    let mut ey = vec![0.0f32; res * res];
    for y in 0..res {
        for x in 0..res {
            let i = y * res + x;
            let gx = (m[at(x as i64 + 1, y as i64)] - m[at(x as i64 - 1, y as i64)]) * 0.5;
            let gy = (m[at(x as i64, y as i64 + 1)] - m[at(x as i64, y as i64 - 1)]) * 0.5;
            let gl = (gx * gx + gy * gy).sqrt();
            if gl > 1e-6 {
                let t = (params.threshold - m[i]).clamp(-0.5, 0.5);
                ex[i] = gx / gl * t;
                ey[i] = gy / gl * t;
            }
        }
    }

    if !edge.iter().any(|&b| b) {
        let value = if inside.iter().all(|&b| b) { -1.0 } else { 1.0 };
        return vec![value; res * res];
    }

    let (qy0, qx0) = exact_edt_nearest_index(&edge, res, wrap);

    // Refine: the nearest edge *pixel* isn't always the one with the
    // nearest subpixel edge *point* — test its own 3x3 edge neighbors too.
    let mut best_sq = vec![f32::INFINITY; res * res];
    for py in 0..res {
        for px in 0..res {
            let i = py * res + px;
            for oy in -1i64..=1 {
                for ox in -1i64..=1 {
                    let qy = qy0[i] + oy;
                    let qx = qx0[i] + ox;
                    let (sy, sx) = if wrap {
                        (
                            qy.rem_euclid(res as i64) as usize,
                            qx.rem_euclid(res as i64) as usize,
                        )
                    } else {
                        (
                            qy.clamp(0, res as i64 - 1) as usize,
                            qx.clamp(0, res as i64 - 1) as usize,
                        )
                    };
                    let s = sy * res + sx;
                    if !edge[s] {
                        continue;
                    }
                    let dy = (qy as f32 + ey[s]) - py as f32;
                    let dx = (qx as f32 + ex[s]) - px as f32;
                    let d2 = dx * dx + dy * dy;
                    if d2 < best_sq[i] {
                        best_sq[i] = d2;
                    }
                }
            }
        }
    }

    let clamp = params.max_distance;
    best_sq
        .iter()
        .zip(inside.iter())
        .map(|(&d2, &is_inside)| {
            let d = d2.sqrt() / res as f32;
            let signed = if is_inside { -d } else { d };
            if clamp > 0.0 {
                signed.clamp(-clamp, clamp)
            } else {
                signed
            }
        })
        .collect()
}

/// Nearest edge-*pixel* coordinate per pixel, as `(y, x)` — **not**
/// necessarily inside `0..res`: when `wrap` is on, the nearest edge pixel
/// might be a wrapped copy from across the seam, and the caller needs that
/// unwrapped coordinate to measure the correct (non-wrapped) geometric
/// distance. Computed via two separable 1D passes
/// (Felzenszwalb & Huttenlocher), each tracking which input index won
/// (the index *is* the nearest-feature lookup; distance is just a
/// byproduct here, discarded past the first pass).
fn exact_edt_nearest_index(edge: &[bool], res: usize, wrap: bool) -> (Vec<i64>, Vec<i64>) {
    if !wrap {
        let (qy, qx) = exact_edt_nearest_index_plain(edge, res);
        return (
            qy.into_iter().map(|v| v as i64).collect(),
            qx.into_iter().map(|v| v as i64).collect(),
        );
    }

    // Periodic padding by res/2 on each side (matches the reference
    // exactly) so distances across the tile seam stay correct up to 0.5
    // tile: a feature just past one edge is visible, at the right
    // distance, to a query pixel near the opposite edge.
    let p = res / 2;
    let padded = res + 2 * p;
    let mut padded_edge = vec![false; padded * padded];
    for y in 0..padded {
        let src_y = (y as i64 - p as i64).rem_euclid(res as i64) as usize;
        for x in 0..padded {
            let src_x = (x as i64 - p as i64).rem_euclid(res as i64) as usize;
            padded_edge[y * padded + x] = edge[src_y * res + src_x];
        }
    }

    let (qy_full, qx_full) = exact_edt_nearest_index_plain(&padded_edge, padded);
    let mut qy = vec![0i64; res * res];
    let mut qx = vec![0i64; res * res];
    for y in 0..res {
        for x in 0..res {
            let src = (y + p) * padded + (x + p);
            qy[y * res + x] = qy_full[src] as i64 - p as i64;
            qx[y * res + x] = qx_full[src] as i64 - p as i64;
        }
    }
    (qy, qx)
}

/// The same transform as [`exact_edt_nearest_index`], but over a plain
/// (non-wrapped, non-padded) square boolean image — the shared worker both
/// the wrap and non-wrap cases above reduce to.
fn exact_edt_nearest_index_plain(edge: &[bool], size: usize) -> (Vec<usize>, Vec<usize>) {
    // Pass 1: along each column, nearest feature row.
    let mut col_cost = vec![0.0f32; size * size];
    let mut col_best_y = vec![0usize; size * size];
    let mut column = vec![0.0f32; size];
    for x in 0..size {
        for y in 0..size {
            column[y] = if edge[y * size + x] { 0.0 } else { UNREACHABLE };
        }
        let (d, idx) = edt_1d_with_index(&column);
        for y in 0..size {
            col_cost[y * size + x] = d[y];
            col_best_y[y * size + x] = idx[y];
        }
    }

    // Pass 2: along each row, nearest column (by pass-1 cost) — composing
    // with pass 1's own index gives the true nearest feature coordinate.
    let mut qy = vec![0usize; size * size];
    let mut qx = vec![0usize; size * size];
    let mut row = vec![0.0f32; size];
    for y in 0..size {
        for x in 0..size {
            row[x] = col_cost[y * size + x];
        }
        let (_d, idx) = edt_1d_with_index(&row);
        for x in 0..size {
            let winning_x = idx[x];
            qx[y * size + x] = winning_x;
            qy[y * size + x] = col_best_y[y * size + winning_x];
        }
    }
    (qy, qx)
}

/// Felzenszwalb & Huttenlocher's linear-time 1D lower-envelope distance
/// transform, extended to also return which input index won at each
/// output position (the standard presentation only returns the distance;
/// the winning parabola's own vertex location is already sitting right
/// there in `v`, just usually thrown away).
fn edt_1d_with_index(f: &[f32]) -> (Vec<f32>, Vec<usize>) {
    let n = f.len();
    let mut d = vec![0.0f32; n];
    let mut winner = vec![0usize; n];
    let mut v = vec![0usize; n]; // parabola vertex (input index) per envelope entry
    let mut z = vec![0.0f32; n + 1]; // right boundary of envelope entry k
    let mut k = 0usize;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;

    let intersect = |q: usize, vk: usize| -> f32 {
        let fq = f[q] + (q * q) as f32;
        let fv = f[vk] + (vk * vk) as f32;
        (fq - fv) / (2.0 * q as f32 - 2.0 * vk as f32)
    };

    for q in 1..n {
        loop {
            let s = intersect(q, v[k]);
            if s <= z[k] && k > 0 {
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = s;
                z[k + 1] = f32::INFINITY;
                break;
            }
        }
    }

    k = 0;
    for (q, slot) in d.iter_mut().enumerate() {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let vk = v[k];
        let diff = q as f32 - vk as f32;
        *slot = diff * diff + f[vk];
        winner[q] = vk;
    }
    (d, winner)
}

#[cfg(test)]
mod tests {
    use super::{compute_distance_field, edt_1d_with_index, DistanceFieldParams, UNREACHABLE};
    use crate::test_support::{golden_path, read_npy_f32_2d};

    const U: f32 = UNREACHABLE;

    #[test]
    fn edt_1d_single_feature() {
        let (d, idx) = edt_1d_with_index(&[U, 0.0, U, U, U]);
        assert_eq!(d, [1.0, 0.0, 1.0, 4.0, 9.0]);
        assert_eq!(idx, [1, 1, 1, 1, 1]);
    }

    #[test]
    fn edt_1d_two_features() {
        let (d, idx) = edt_1d_with_index(&[0.0, U, U, U, 0.0]);
        assert_eq!(d, [0.0, 1.0, 4.0, 1.0, 0.0]);
        assert_eq!(idx, [0, 0, 0, 4, 4]);
    }

    #[test]
    fn edt_1d_no_features() {
        let (d, _idx) = edt_1d_with_index(&[U, U, U]);
        assert!(d.iter().all(|&v| v >= U));
    }

    fn mean_p99_error_px(got: &[f32], want: &[f32], res: usize) -> (f64, f64) {
        let mut errs: Vec<f64> = got
            .iter()
            .zip(want.iter())
            .map(|(&g, &w)| ((g - w) as f64 * res as f64).abs())
            .collect();
        let mean = errs.iter().sum::<f64>() / errs.len() as f64;
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p99 = errs[(errs.len() as f64 * 0.99) as usize];
        (mean, p99)
    }

    /// HARDSURFACE_NODES.md §6: mean under 0.05 px, p99 under 0.3 px,
    /// against the real reference goldens (both 512x512 anti-aliased
    /// masks). `golden/accuracy.json`'s own measured numbers (circle: mean
    /// 0.0175, p99 0.166; box: mean 0.0112, p99 0.261) are what the Rust
    /// port doesn't need to beat, just stay under the stated ceiling.
    #[test]
    fn df_circle_matches_the_accuracy_table() {
        let mask = read_npy_f32_2d(&golden_path("hardsurface", "df_circle_input_mask.npy"));
        let want = read_npy_f32_2d(&golden_path("hardsurface", "df_circle_sdf.npy"));
        let got = compute_distance_field(&mask, 512, &DistanceFieldParams::default());
        let (mean, p99) = mean_p99_error_px(&got, &want, 512);
        assert!(mean < 0.05, "df_circle: mean error {mean} px");
        assert!(p99 < 0.3, "df_circle: p99 error {p99} px");
    }

    #[test]
    fn df_box_matches_the_accuracy_table() {
        let mask = read_npy_f32_2d(&golden_path("hardsurface", "df_box_input_mask.npy"));
        let want = read_npy_f32_2d(&golden_path("hardsurface", "df_box_sdf.npy"));
        let got = compute_distance_field(&mask, 512, &DistanceFieldParams::default());
        let (mean, p99) = mean_p99_error_px(&got, &want, 512);
        assert!(mean < 0.05, "df_box: mean error {mean} px");
        assert!(p99 < 0.3, "df_box: p99 error {p99} px");
    }

    /// A disc painted centered on the wrap seam (`uv.x == 0`) should read
    /// as one continuous shape across the seam when `wrap` is on — the
    /// leftmost and rightmost columns should see small, consistent
    /// distances to it, not the large "no edge nearby on this side" values
    /// `wrap = false` would produce there.
    #[test]
    fn wrap_true_sees_across_the_seam() {
        const RES: usize = 64;
        let mut mask = vec![0.0f32; RES * RES];
        let r = 8i64;
        for y in 0..RES {
            for x in 0..RES {
                // Distance to (x=0, y=RES/2), wrapped.
                let dx = (x as i64).min(RES as i64 - (x as i64));
                let dy = (y as i64) - (RES as i64 / 2);
                if dx * dx + dy * dy <= r * r {
                    mask[y * RES + x] = 1.0;
                }
            }
        }
        let wrapped = compute_distance_field(
            &mask,
            RES,
            &DistanceFieldParams {
                wrap: true,
                ..DistanceFieldParams::default()
            },
        );
        let unwrapped = compute_distance_field(
            &mask,
            RES,
            &DistanceFieldParams {
                wrap: false,
                ..DistanceFieldParams::default()
            },
        );
        let mid = RES / 2;
        let left_edge = wrapped[mid * RES];
        let right_edge = wrapped[mid * RES + RES - 1];
        assert!(
            left_edge < 0.1 && right_edge < 0.1,
            "wrap=true should see the disc straddling the seam from both sides: \
             left={left_edge} right={right_edge}"
        );
        assert_ne!(
            wrapped, unwrapped,
            "wrap on vs off should disagree near a seam-straddling shape"
        );
    }

    #[test]
    fn sign_is_negative_inside_positive_outside() {
        const RES: usize = 64;
        let mut mask = vec![0.0f32; RES * RES];
        let (cx, cy, r) = (32i64, 32i64, 20i64);
        for y in 0..RES {
            for x in 0..RES {
                let dx = x as i64 - cx;
                let dy = y as i64 - cy;
                if dx * dx + dy * dy <= r * r {
                    mask[y * RES + x] = 1.0;
                }
            }
        }
        let sdf = compute_distance_field(&mask, RES, &DistanceFieldParams::default());
        assert!(
            sdf[32 * RES + 32] < 0.0,
            "center should be inside (negative)"
        );
        assert!(sdf[0] > 0.0, "corner should be outside (positive)");
    }

    /// Regression test for a real bug: thresholding used to shift the
    /// *entire* mask by a constant before running the algorithm, which
    /// corrupted the ramp-pixel test for any threshold != 0.5 — a hard 0/1
    /// mask's background pixels (`0.0`) became small positive values after
    /// shifting, fell inside the `0 < x < 1` ramp-pixel range, and got
    /// misread as edge pixels almost everywhere (reported as "the whole
    /// background renders as the zero contour" once Threshold moved off
    /// its default). A background pixel far from any real edge must read a
    /// real, non-near-zero distance regardless of `threshold`.
    #[test]
    fn non_default_threshold_does_not_corrupt_background_pixels() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "square",
            256,
            &crate::shape::ShapeParams {
                kind: crate::shape::ShapeKind::Square,
                ..crate::shape::ShapeParams::default()
            },
        );
        let raw = gpu.read_rgba16f(pool.texture("square").unwrap(), 256);
        let mask: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
        let sdf = compute_distance_field(
            &mask,
            256,
            &DistanceFieldParams {
                wrap: true,
                threshold: 0.41,
                max_distance: 0.41,
            },
        );
        let corner = sdf[0];
        assert!(
            corner > 0.1,
            "a far-background corner pixel should read a real positive distance, not \
             near-zero: got {corner}"
        );
    }

    /// A row of several repeated discs (like Pattern Generator's Horizontal
    /// Only output) is a real-world multi-feature mask, unlike the
    /// single-shape `df_circle`/`df_box` goldens above — checks the exact
    /// transform stays numerically sane (no `NaN`, no wild non-smooth
    /// jumps) once more than one edge feature is involved, not just that
    /// it matches a golden for one.
    #[test]
    fn multi_feature_mask_stays_numerically_sane() {
        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        crate::shape::dispatch(
            &mut gpu,
            &mut pool,
            "disc",
            256,
            &crate::shape::ShapeParams::default(),
        );
        crate::tile_generator::dispatch(
            &mut gpu,
            &mut pool,
            Some("disc"),
            None,
            "dashes",
            256,
            &crate::tile_generator::TileGeneratorParams {
                rows: 11,
                columns: 11,
                offset: 0.0,
                shape_size: 0.2,
                distribution: crate::tile_generator::PatternDistribution::HorizontalOnly,
                ..crate::tile_generator::TileGeneratorParams::default()
            },
        );
        let raw = gpu.read_rgba16f(pool.texture("dashes").unwrap(), 256);
        let mask: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
        let sdf = compute_distance_field(&mask, 256, &DistanceFieldParams::default());

        assert!(sdf.iter().all(|v| v.is_finite()), "no NaN/infinite values");

        // A scanline far above the dash row (y=20): every step should stay
        // close to its neighbor (a smooth field, even if it wobbles gently
        // from being roughly equidistant between several discs) — a real
        // bug would show a sharp, large jump instead.
        let row: Vec<f32> = (0..256).map(|x| sdf[20 * 256 + x]).collect();
        let max_step = row
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_step < 0.05,
            "expected a smooth scanline, found a jump of {max_step} tile units between \
             adjacent pixels"
        );
    }

    #[test]
    fn from_graph_round_trips() {
        let mut params = std::collections::BTreeMap::new();
        params.insert("wrap".to_string(), mc_core::ParamValue::Bool(false));
        params.insert("threshold".to_string(), mc_core::ParamValue::Float(0.3));
        params.insert("max_distance".to_string(), mc_core::ParamValue::Float(0.1));
        let parsed = DistanceFieldParams::from_graph(&params);
        assert!(!parsed.wrap);
        assert_eq!(parsed.threshold, 0.3);
        assert_eq!(parsed.max_distance, 0.1);

        let defaults = DistanceFieldParams::from_graph(&std::collections::BTreeMap::new());
        assert!(defaults.wrap);
        assert_eq!(defaults.threshold, 0.5);
        assert_eq!(defaults.max_distance, 0.0);
    }
}
