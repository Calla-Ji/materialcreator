// SPDX-License-Identifier: MIT OR Apache-2.0
//! Wrap-aware 4-connected component labeling for Distance's Luminance
//! Jitter (`mc_nodes::distance_spread`).
//!
//! An earlier version of this ran entirely on the GPU: a relax-plus-
//! pointer-jump shader pass, repeated many times. That's a fundamentally
//! *local* operation — each round only ever learns about pixels one texel
//! away — so a single blob winding most of the way across the canvas (easy
//! to get from a low Threshold plus a Pattern Generator mask whose own
//! Edge Darkening never truly reaches pure black) needed a round count
//! approaching the canvas's own diameter to fully converge, not the
//! `O(log(resolution))` a geometric Jump Flooding pass gets away with.
//! Measured concretely: ~65ms at 2048x2048 for a worst-case mask, 13x this
//! project's own <5ms/2048x2px budget (CLAUDE.md §6.4), confirmed to scale
//! with how many rounds ran, not fixed per-call overhead.
//!
//! Union-find connected-component labeling is the standard, textbook
//! solution to this exact problem — a single linear-time CPU pass instead
//! of dozens-to-hundreds of GPU rounds. `mc_nodes::distance_spread::dispatch`
//! reads its own seed pass's output back to the CPU, labels it here, and
//! uploads the result — a synchronous GPU/CPU round trip, but only when
//! Luminance Jitter is actually non-zero (the common case, jitter left at
//! its default 0, never touches this path at all).

/// Labels every connected blob in `occupied` (a `resolution * resolution`,
/// row-major grid — `occupied[y * resolution + x]`, matching the layout
/// `mc_gpu::Context::read_r32i` returns) with a packed coordinate shared by
/// every other pixel in the same 4-connected blob (`wrap_x`/`wrap_y`
/// control whether the left/right and top/bottom edges count as touching).
///
/// Returns a same-size, same-layout buffer: `-1` for an unoccupied pixel,
/// or `rep_x * resolution + rep_y` (matching `distance_spread_seed.wgsl`'s
/// own packed-coordinate convention, not the row-major layout of this
/// function's own input/output buffers — the two are independent) for an
/// occupied one, where `(rep_x, rep_y)` is some single pixel belonging to
/// that blob, consistent for every pixel in it.
///
/// Deterministic for the same `occupied` input (CLAUDE.md §6.3) — a single-
/// threaded scan in a fixed order, no hashing or randomness involved.
pub fn label_components(
    occupied: &[bool],
    resolution: u32,
    wrap_x: bool,
    wrap_y: bool,
) -> Vec<i32> {
    let res = resolution as usize;
    debug_assert_eq!(occupied.len(), res * res);

    let mut parent: Vec<u32> = (0..(res * res) as u32).collect();
    let mut size: Vec<u32> = vec![1; res * res];

    for y in 0..res {
        for x in 0..res {
            if !occupied[y * res + x] {
                continue;
            }
            if let Some(left) = wrap_neighbor(x, res, wrap_x) {
                if occupied[y * res + left] {
                    union(
                        &mut parent,
                        &mut size,
                        (y * res + x) as u32,
                        (y * res + left) as u32,
                    );
                }
            }
            if let Some(up) = wrap_neighbor(y, res, wrap_y) {
                if occupied[up * res + x] {
                    union(
                        &mut parent,
                        &mut size,
                        (y * res + x) as u32,
                        (up * res + x) as u32,
                    );
                }
            }
        }
    }

    let mut labels = vec![-1i32; res * res];
    for y in 0..res {
        for x in 0..res {
            if !occupied[y * res + x] {
                continue;
            }
            let root = find(&mut parent, (y * res + x) as u32) as usize;
            let (rep_x, rep_y) = (root % res, root / res);
            labels[y * res + x] = (rep_x * res + rep_y) as i32;
        }
    }
    labels
}

/// The neighbor index one step before `coord` along an axis of length
/// `size` — wrapped to `size - 1` when `coord == 0` and `wrap` is set,
/// or `None` at the boundary with no wraparound (nothing to union with).
fn wrap_neighbor(coord: usize, size: usize, wrap: bool) -> Option<usize> {
    if coord > 0 {
        Some(coord - 1)
    } else if wrap && size > 1 {
        Some(size - 1)
    } else {
        None
    }
}

/// Finds `i`'s root, path-halving as it goes (every visited node points
/// straight at its own grandparent) — one pass over the chain instead of
/// two full ones, which in practice beats full path compression here: this
/// runs on a plain flat array for a 2048x2048-or-larger grid, so fewer
/// passes over memory matters more than the (already tiny, union-by-size
/// keeps trees shallow) difference in resulting tree depth.
fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        parent[i as usize] = parent[parent[i as usize] as usize];
        i = parent[i as usize];
    }
    i
}

/// Union by size, with a deterministic tiebreak (lower index wins) so the
/// same input always collapses to the same root regardless of visitation
/// order — both just for reproducibility, not correctness, since either
/// choice gives a valid union-find.
fn union(parent: &mut [u32], size: &mut [u32], a: u32, b: u32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra == rb {
        return;
    }
    let (big, small) = if size[ra as usize] > size[rb as usize]
        || (size[ra as usize] == size[rb as usize] && ra < rb)
    {
        (ra, rb)
    } else {
        (rb, ra)
    };
    parent[small as usize] = big;
    size[big as usize] += size[small as usize];
}

#[cfg(test)]
mod tests {
    use super::label_components;

    fn grid(rows: &[&str]) -> (Vec<bool>, u32) {
        let res = rows.len() as u32;
        let mut occupied = vec![false; (res * res) as usize];
        for (y, row) in rows.iter().enumerate() {
            for (x, c) in row.chars().enumerate() {
                occupied[y * res as usize + x] = c == '#';
            }
        }
        (occupied, res)
    }

    #[test]
    fn two_disconnected_blobs_get_different_labels() {
        let (occupied, res) = grid(&["#...", "....", "....", "...#"]);
        let labels = label_components(&occupied, res, false, false);
        let a = labels[0];
        let b = labels[3 * res as usize + 3];
        assert_ne!(a, -1);
        assert_ne!(b, -1);
        assert_ne!(a, b, "two isolated pixels should be separate blobs");
    }

    #[test]
    fn a_winding_connected_blob_gets_one_label_everywhere() {
        // Full rows joined end-to-end by single-pixel connector rows,
        // alternating which edge the connector sits on — one connected
        // blob that still needs many hops to link its first and last row.
        let (occupied, res) = grid(&[
            "########", ".......#", "########", "#.......", "########", ".......#", "########",
            "#.......",
        ]);
        let labels = label_components(&occupied, res, false, false);
        let mut seen = None;
        for (i, &label) in labels.iter().enumerate() {
            if !occupied[i] {
                assert_eq!(label, -1);
                continue;
            }
            match seen {
                None => seen = Some(label),
                Some(expected) => assert_eq!(
                    label, expected,
                    "every pixel of one connected snake should share a label"
                ),
            }
        }
    }

    #[test]
    fn wrap_x_connects_left_and_right_edges() {
        let (occupied, res) = grid(&["#..#", "....", "....", "...."]);
        let no_wrap = label_components(&occupied, res, false, false);
        let with_wrap = label_components(&occupied, res, true, false);
        assert_ne!(
            no_wrap[0], no_wrap[3],
            "without wrap, edge pixels on the same row aren't adjacent"
        );
        assert_eq!(
            with_wrap[0], with_wrap[3],
            "with wrap_x, column 0 and the last column are adjacent"
        );
    }

    #[test]
    fn wrap_y_connects_top_and_bottom_edges() {
        let (occupied, res) = grid(&["#...", "....", "....", "#..."]);
        let no_wrap = label_components(&occupied, res, false, false);
        let with_wrap = label_components(&occupied, res, false, true);
        let top = 0usize;
        let bottom = 3 * res as usize;
        assert_ne!(no_wrap[top], no_wrap[bottom]);
        assert_eq!(with_wrap[top], with_wrap[bottom]);
    }

    #[test]
    fn empty_grid_is_all_unlabeled() {
        let (occupied, res) = grid(&["....", "....", "....", "...."]);
        let labels = label_components(&occupied, res, true, true);
        assert!(labels.iter().all(|&l| l == -1));
    }

    #[test]
    fn is_deterministic() {
        let (occupied, res) = grid(&[
            "##.#....", "#.##..#.", "...#.##.", "##...#..", ".#.##...", "..#.#.##", "#.##..#.",
            "...#.##.",
        ]);
        let a = label_components(&occupied, res, true, true);
        let b = label_components(&occupied, res, true, true);
        assert_eq!(a, b);
    }
}
