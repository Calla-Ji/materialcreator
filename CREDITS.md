# CREDITS.md — Algorithm sources

Every node in MaterialCreator is implemented from scratch in Rust/WGSL against the general,
published technique it's named after — never against another tool's code, documentation, or
internal file formats. This file records which published technique each node builds on, as the
paper trail for that.

## Noise & patterns

- **Perlin Noise** — Ken Perlin's gradient noise, as described in *An Image Synthesizer*
  (SIGGRAPH 1985) and refined in *Improving Noise* (SIGGRAPH 2002). The reference lattice/
  gradient-hash structure is standard and public domain; this implementation is tileable by
  construction (the gradient hash wraps at a fixed period), which the original algorithm
  doesn't guarantee on its own.
- **Simplex Noise** — the triangular-lattice noise Ken Perlin introduced in 2001, implemented
  here following the practical breakdown in Stefan Gustavson's *Simplex Noise Demystified*
  (2005). Tileability is handled the same way as Perlin Noise above: by blending multiple
  period-shifted copies so both tile edges evaluate identically.
- **Voronoi** — cellular/Worley noise, from Steven Worley's *A Cellular Texture Basis
  Function* (SIGGRAPH 1996): distances from each pixel to the nearest of a set of scattered
  feature points, with F1/F2/F2−F1/edges as the standard output modes.
- **Fractal Sum / Clouds** — fractional Brownian motion (fBm): summing several octaves of a
  base noise at increasing frequency and decreasing amplitude, a standard technique with no
  single canonical source (see e.g. Ken Musgrave's writing on fractal terrain synthesis).

## Filters

- **Blur / Directional Blur / Slope Blur** — a standard separable Gaussian blur (textbook
  image-processing technique), wrap-addressed for tiling. Past a 16px radius, Blur switches to
  a standard downsample/blur/upsample pyramid (box-downsample until the effective radius is
  back under 16px, run the same separable Gaussian there, bilinear-upsample back to full
  resolution) — the standard technique for keeping a large-radius blur both alias-free and
  cheap, independently implemented here, not ported from any reference code.
- **Height to Normal**'s gradient estimation — the Sobel operator (Sobel & Feldman, 1968), a
  standard image-gradient kernel.
- **Warp / Directional Warp** — displacing a texture by the gradient of a second ("intensity")
  input, a standard procedural-texturing technique with no single canonical source.
- **Bevel** — box-blur-then-remap: a uniform box blur (not a Gaussian — a box blur of a hard
  step produces an exactly linear ramp over its own width, which reads as a defined chamfer
  rather than a generic soft blur) at a mask/height input's edges, then reshape *how much the
  blur changed each pixel* with a power curve, added back onto the unblurred original. A
  standard, cheap approximation of a distance-based bevel (vs. computing a true distance-to-edge
  field), with no single canonical source — this project's own derivation of the box-blur
  radius/remap exponent parameterization, including the delta-based remap formulation (rather
  than reshaping the blurred value directly) that guarantees a no-bevel setting is an exact
  no-op on any input. The box blur itself is computed with a sliding-window running sum (one
  thread per row/column, add the texel entering the window and subtract the one leaving it) —
  a standard, well-known box-filter technique, chosen here over a fixed-stride subsample after
  the latter produced a visible staircase artifact on hard edges.
- **Distance** — nearest-seed propagation via the Jump Flooding Algorithm, Rong & Tan's *Jump
  Flooding in GPU with Applications to Voronoi Diagram and Distance Transform* (ACM I3D 2006),
  plus the "JFA+1" extra same-step-size refinement pass several later publications on the
  technique add to clean up its own known small approximation error — independently implemented
  here in WGSL, not ported from any reference code. Luminance Jitter's per-blob brightness (each
  disconnected island in Mask gets one consistent random value, not per-pixel noise) needed a
  real connected-component identity, which nearest-seed propagation alone doesn't give — a real
  blob can only be *discovered* one direct-neighbor hop at a time without risking a false merge
  across a gap (unlike Jump Flooding's own geometric hops, fine for nearest-seed search but
  exactly wrong for this), which makes GPU-side relaxation fundamentally local and genuinely slow
  on a large, winding blob (confirmed concretely: ~65ms at 2048x2048 for a real worst-case mask,
  13x this project's own <5ms/2048x2px budget — see CLAUDE.md §6.4). Labeling instead runs as a
  single linear-time union-find connected-component pass (`mc_nodes::connected_components`,
  CPU-side) on the seed pass's own output, read back and re-uploaded only when Luminance Jitter
  is actually non-zero — standard, textbook union-find with union-by-size and path-halving (see
  e.g. Tarjan & van Leeuwen's *Worst-Case Analysis of Set Union Algorithms*, JACM 1984, for the
  general technique), this project's own implementation.
- **Outline** — reuses the same Jump Flooding Algorithm as Distance above (same citation), with
  one change: seeds are placed at selected-interface *midpoints* (a half-pixel position between
  two pixels), packed as doubled integer coordinates rather than mask-pixel coordinates, so a
  half-pixel offset is represented exactly rather than approximated. Rounding is grayscale
  morphological closing (dilation then erosion over a disk-shaped structuring element) — a
  standard image-processing technique, independently implemented here as a brute per-pixel scan,
  not ported from any reference code (see e.g. SciPy's own `grey_closing` documentation for the
  general definition: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_closing.html).

## Stylize & glitch

- **Dither Pattern**'s Bayer 2×2/4×4/8×8 modes — the recursively-defined ordered-dither (Bayer)
  threshold matrix, a public-domain technique (see e.g. Wikipedia's "Ordered dithering"
  article for the general recursive definition); implemented from scratch in
  `shaders/common.wgsl`'s `bayer_threshold`, verified against the standard 2×2 and 4×4
  matrices by this project's own tests.
- **Dither Pattern**'s Blue Noise mode — samples the bundled CC0 blue noise texture
  (Christoph Peters, see ASSETS.md) rather than computing noise procedurally.
- **Dither Pattern**'s Rotation option — a classic halftone "screen angle," implemented via
  an integer-lattice rotation matrix (small (a, b) pairs, e.g. (2,1) for ~26.6°) rather than a
  continuous angle, since that's what keeps it exactly tileable; a standard, independently
  worked-out technique (see e.g. "rational tangent screening" in print-halftone literature),
  not adapted from any specific source.
- **Ordered Dither** and **Palette Dither** — standard ordered-dithering math (quantize, then
  nudge by a threshold mask before rounding), the same public-domain technique Dither Pattern's
  masks are built on, applied here to an actual image instead of just outputting the mask
  itself. Palette Dither's "nearest palette color" step and its Perceptual (luminance-weighted)
  distance option are a standard color-quantization technique, original implementation.
- **Block Shift**, **Channel Split**, **Scanline Jitter**, **Data Corrupt** — original glitch
  nodes built from general, independently-implemented ideas (seeded per-block/per-band hashing,
  wrap-addressed resampling, per-channel UV offsets), with
  [KinoGlitch](https://github.com/keijiro/KinoGlitch) (Keijiro Takahashi, MIT) as inspiration
  for which effects are worth having, not as ported code — this project's versions are WGSL
  compute shaders built against `common.wgsl`'s own hash helper, not a line-by-line port of
  KinoGlitch's Unity HLSL.
- Background reading for the family as a whole (concepts only; no code taken): Surma's
  [*Ditherpunk*](https://surma.dev/things/ditherpunk/) and its
  [sequel](https://www.makeworld.space/2021/02/dithering.html), and
  [KinoGlitch](https://github.com/keijiro/KinoGlitch) (Keijiro Takahashi, MIT) for
  general glitch-node ideas, reimplemented in WGSL rather than ported.

## Hard-surface

- **Panel Layout** — a tileable guillotine subdivision (recursive rectangle splitting, each
  split seeded so it's deterministic and periodic at the tile edge) — a standard CAD/procedural-
  layout technique with no single canonical paper; this project's own formulation of the
  recursion and its seeding.
- **Distance Field** — an exact signed distance field from an arbitrary mask's own 0.5 iso-line,
  computed directly (brute-force nearest-feature search per pixel, not a blurred approximation)
  — the same standard definition of an SDF used throughout computer graphics, independently
  implemented.
- **Shape Boolean** — unions/subtracts/intersects two SDFs with Inigo Quilez's widely-published
  smooth-minimum formula (see [iquilezles.org](https://iquilezles.org/articles/smin/)) for the
  smooth blend mode, and round/chamfer/groove boolean operators in the style described by
  [hg_sdf](https://mercury.sexy/hg_sdf/) (Mercury Demogroup, MIT-licensed) — see
  [THIRD_PARTY.md](THIRD_PARTY.md). Both are referenced as published formulas, not copied shader
  code: this project's own WGSL is an independent implementation.
- **Redistance** — rebuilds an exact distance field from any SDF's own zero contour (a thin
  wrapper reusing Distance Field above), erasing the false interior seams a boolean or warp
  otherwise leaves for a later Bevel Profile to expose — a standard SDF-hygiene technique, no
  single canonical source.
- **Bevel Profile** — shapes an exact SDF into a height map through a chosen edge profile (flat,
  round, chamfer) directly from the known distance-to-edge, rather than the usual blur-then-
  curve-then-histogram approximation a height map alone would need — this project's own
  derivation, since it depends on Distance Field/Shape Boolean already existing upstream.
- **Fasteners** — procedural rivet/bolt/screw head shapes (6 kinds), stamped along a Path —
  original geometry, no external technique.
- **Edge Wear** — wear on convex edges and a cavity mask for dirt, either directly from an SDF's
  own curvature (exact) or, with only a height map available, from a Laplacian-of-Gaussian
  curvature estimate — LoG (the "Marr-Hildreth" edge/blob operator) is a standard, widely-taught
  image-processing technique; the wear/cavity thresholding built on top of it is original to
  this project.

## Fabric & yarn

- **Fabric Generator** — Plain Weave/Twill/Rib Knit/Stockinette structure (which thread crosses
  over which, per weave type) is original compositing logic written for this project, built on
  top of Thread's own geometry below — no external technique.
- **Thread** (`mc_nodes::thread_shape`) — plied-yarn geometry (ply packing, the S/Z helix twist
  convention, balanced-vs-crepe fiber twist, cable/helix-of-helices construction) derived from
  the owner's own textile reference book's yarn-construction chapter, not from any other
  software tool's node or source. The owner wrote a design brief (`THREAD_NODE_SPEC.md`) and a
  small Python/numpy ground-truth reference renderer (`thread_reference.py`) deriving the exact
  formulas from that book — both kept in the private notes repo (not this public one) alongside
  `CLAUDE.md`. This project's WGSL implementation is an independent, from-scratch port of that
  derivation: the per-pixel tube/plane-intersection quadratic (a tilted cylinder's cross-section
  in a fixed plane is an ellipse — standard analytic geometry, independently derived rather than
  taken from a graphics-literature source) and the top-surface-height occlusion test (the
  highest tube wins at each pixel, the same idea general-purpose rendering calls a z-buffer) are
  both original to this project. Verified by rendering all of the reference script's own test
  cases and comparing pixel-for-pixel against its output (`crates/mc_cli/tests/thread_golden.rs`),
  not just checked for internal consistency.

## 3D preview

- **Tangent-space basis computation** (`compute_tangents` in `mc_app::viewport3d`) — Eric
  Lengyel's method for deriving a per-vertex tangent from a triangle's position/UV gradient,
  as published in *Foundations of Game Engine Development, Volume 2: Rendering* and his
  earlier widely-circulated article *Computing Tangent Space Basis Vectors for an Arbitrary
  Mesh*.
- **PBR shading** (`viewport3d.wgsl`) — the metallic-roughness GGX microfacet model used by
  virtually every modern physically-based renderer (Blender's Principled BSDF, Unreal, Godot,
  glTF's own material model included): Brent Burley's *Physically Based Shading at Disney*
  (SIGGRAPH 2012 course notes) and Brian Karis's *Real Shading in Unreal Engine 4* (SIGGRAPH
  2013 course notes), both public and widely cited.
- **GPU vertex-pulling tessellation** (`viewport3d.wgsl`'s `vs_main_tess`, `mc_app::tess_mesh`/
  `tess_gpu`) — subdivides a small fixed base mesh into detail triangles directly in the vertex
  shader (decode a sub-triangle lattice index, evaluate position/normal/UV procedurally, draw
  instanced with no CPU-built vertex buffer), used because WebGPU/wgpu has no hardware
  tessellation stage (see [gpuweb/gpuweb#445](https://github.com/gpuweb/gpuweb/issues/445) and
  [gfx-rs/wgpu#222](https://github.com/gfx-rs/wgpu/issues/222)). Vertex pulling/sub-triangle
  lattice decoding is a published real-time-graphics technique with no single canonical paper;
  this project's own implementation, independently authored from a from-scratch spec and a
  Python/numpy ground-truth reference, golden-tested against it
  (`crates/mc_app/testdata/tess_golden/`) — not derived from any commercial tool's own code.

## Shape math

The Shape node's Star (an exact straight-edge star-polygon distance field), Leaf (an
asymmetric superellipse with independently blended top/middle/bottom exponents), and the
per-kind upright-orientation offsets are original derivations written for this project —
worked out and verified against this project's own test suite and rendered output, not
adapted from any other tool.

---

If you add a node, add an entry here naming the published technique (or "original
derivation, see `<path>`" if there isn't one) before merging.
