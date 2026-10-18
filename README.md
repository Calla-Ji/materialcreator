# MaterialCreator

A free, open-source, native node-based procedural material authoring tool, built in Rust.

Connect nodes — noises, patterns, filters, blends — into a graph that outputs a seamless,
tileable PBR texture set, previews it live in 2D and 3D, and exports maps ready for
**Blender** and **Unreal Engine**.

> Status: v0.1 — first public release. New to the workspace? See the navigation/shortcuts
> reference for your OS:
> [macOS](NAVIGATION-macOS.md) · [Windows](NAVIGATION-Windows.md) · [Linux](NAVIGATION-Linux.md).

![The full app: node graph, 2D view, 3D view, and parameter panel](assets/screenshots/hero.png)

## Features

- **Real-time GPU node graph.** Every node is a WGSL compute shader; editing a param or
  rewiring a connection re-evaluates only what's downstream and dirty, live.
- **Live 2D and 3D preview.** The 2D view shows any selected node's output at full
  resolution, with tiling and single-channel views. The 3D view renders the graph's output
  maps on a sphere, cube, plane, rounded cube, or rounded cylinder under a GGX
  metallic-roughness PBR shader, with real per-vertex tangents and height-driven
  displacement on every mesh — not just a flat plane. The Environment panel lets you pick
  between 4 bundled CC0 HDRIs (Poly Haven) for the sky and reflections, or load your own
  `.hdr`/`.exr` file, to see how a material reads under a different lighting mood — a small
  tonemapped preview of the active HDRI shows right under the picker. Channel
  textures get a real generated mip chain and 16x anisotropic filtering, with an Unlit Base
  Color debug mode, an Exposure slider, and a Pixel/Nearest filtering toggle for inspecting
  dither/pixel-art materials crisply. The Alpha output channel drives real transparency in the
  preview, not just an exported mask.

  ![3D viewport under a PBR material, HDRI-lit](assets/screenshots/3d-viewport.png)

- **Seamless tiling, guaranteed — with an escape hatch.** Every generator is periodic and every
  filter samples with wrap-around addressing by default; each node ships with its own automated
  tiling test. Nodes that read past a tile's edge (an offset, a blur, a noise lattice, ...)
  also expose a per-axis Tiling Mode, for the times you want that edge to clip instead of wrap.
  Resolution-independent by design — a graph looks the same at 512px and 4096px, detail
  aside.

  ![2D view with the 2x2 tiling toggle on, seam-free](assets/screenshots/tiling.png)

- **Deterministic.** Same graph, same seed, same pixels — every time, on the same GPU.
- **A custom node-graph canvas** — pan, zoom (including trackpad pinch-to-zoom, working
  anywhere on the canvas), box-select, duplicate, undo/redo, and a category-colored,
  alphabetically-sorted, keyboard-navigable (arrow keys + Enter) searchable add-node menu —
  built directly on top of the UI toolkit's own primitives rather than a third-party graph
  widget. Every node card is color-coded by category, matching a dedicated Node Library panel
  along the canvas's left edge for browsing and adding nodes by icon. New nodes always land
  wherever you last clicked on the canvas.

  ![Node graph canvas, category-colored nodes and wires](assets/screenshots/node-graph.png)

- **Crash-safe autosave and error toasts.** The current graph is snapshotted to a recovery file
  every 30 seconds; if MaterialCreator doesn't shut down cleanly, the next launch offers to
  restore it. Failed opens/saves and graph evaluation errors surface as a dismissable
  notification instead of only reaching the terminal log.
- **Export** to PNG (8-bit and 16-bit) and 32-bit float EXR, at resolutions from 256px up to
  4096px (8192px where VRAM allows), with ready-made naming presets for Blender and for
  Unreal Engine (including an ORM-packed texture: AO/Roughness/Metallic in one file).
- **Save/load** as a human-readable, versioned `.mcgraph` JSON file.

## Nodes

63 nodes, each with its own tiling and determinism tests.

### Generators — Noise
- **Perlin Noise** — classic gradient noise, smooth and organic.
- **Simplex Noise** — gradient noise on a triangular lattice, fewer directional artifacts
  than Perlin.
- **Voronoi** — cellular noise: distances to a scattered set of points (F1, F2, F2−F1, edges).
- **White Noise** — fully uncorrelated random cells, no interpolation.
- **Fractal Sum** — several octaves of a chosen base noise, summed (fBm).
- **Clouds** — an fBm preset tuned for a cloud-like look.
- **Anisotropic Noise** — directional, stretched noise for brushed-metal/wood-grain streaks.
- **Gabor Noise** — oriented, band-limited noise for fine, controllable grain/fabric-like
  texture.
- **Streaks** — scattered directional streak strokes, for scratches and fine fibrous detail.

### Generators — Patterns
- **Pattern Generator** — a brick or grid pattern, with gap, edge darkening (a flat luminance
  fade toward the gap), per-tile size/rotation/luminance jitter, and an optional connected
  Shape to stamp instead of the built-in tile.
- **Shape** — a disc, square, polygon, star, or leaf, centered in the tile. The star and leaf
  are original constructions, not smoothed approximations: the star's points are exact
  straight edges that can "unfold" a plain polygon into a star via one setting, and the leaf
  is a single continuous curve with independently shapeable top, middle, and bottom
  curvature, plus control over where its widest point sits.
- **Circular Pattern** — shapes or noise arranged radially around a center, for dials, gears,
  and radial ornament.
- **Gradient Linear** — a ramp from black to white, rotatable in quarter turns.
- **Gradient Radial** — a ramp from the center outward to a given radius.
- **Checker** — an alternating black/white grid.
- **Uniform Color** — a flat, solid color fill.
- **Image Import** — bring in your own PNG/JPEG/EXR as a node input.
- **Dither Pattern** — a Bayer, blue noise, or halftone dot mask, for use in Blend, Threshold,
  and similar nodes.
- **Type** — render real text from any installed font, for embossed/engraved labels and
  embroidery-style graphs.
- **Panel Layout** — a seamless, tileable guillotine subdivision into armor/hard-surface
  panels, with per-panel SDF, ID, and a fastener placement path — no flood fill, so no seams
  at the tile edge.
- **Fabric Generator** — Plain Weave, Twill, Rib Knit, or Stockinette cloth structure, with an
  optional Thread input for a custom fiber shape and its own Ply ID output for coloring
  individual plies.
- **Thread** — genuine plied-yarn geometry: round tube plies on a helical centerline, with a
  real top-surface-height occlusion test deciding which ply shows where (not a blend). Outputs
  Height, Ply ID (for per-ply coloring via Gradient Map), Fiber (groove detail), and Mask.
- **Knit** — a single-ply stockinette/rib knit loop structure, independent of the Fabric
  Generator + Thread combination above.
- **Stitch** — a repeating hand/machine-stitch mark pattern along a Path.

### Filters — Adjustments
- **Levels** — remap an input/output range, with a gamma curve in between.
- **Curve** — a 5-point adjustable response curve.
- **Invert** and **Clamp** — the usual tone-mapping building blocks.
- **HSL Adjust** — shift hue, scale saturation, shift lightness.
- **Gradient Map** — map a grayscale input through a color gradient.
- **Grayscale Conversion** — Color to Gray by luminance.

### Filters — Blending & Masking
- **Blend** — 9 blend modes (Copy, Add, Subtract, Multiply, Screen, Overlay, Max, Min,
  Difference), opacity, and an optional mask.
- **Value Range** — threshold/window an input around a position.
- **Threshold** — a hard black/white cutoff at a given value.

### Filters — Stylize
- **Ordered Dither** — reduce an image to a few tones with a Bayer or blue noise dither
  pattern; Per Channel or hue-preserving Luminance mode.
- **Palette Dither** — dither a Color image down to a 2–8 color palette, with RGB or
  perceptual (luminance-weighted) nearest-color matching.

### Filters — Glitch
- **Block Shift** — seeded random blocks shift sideways.
- **Channel Split** — sample each RGB channel at its own offset, for a chromatic-aberration
  look.
- **Scanline Jitter** — seeded random horizontal or vertical bands shift sideways.
- **Data Corrupt** — posterize, channel-swap, or invert random blocks for a "corrupted file"
  look.

### Filters — Spatial
- **Blur** and **Directional Blur** — separable Gaussian blur, omnidirectional or along one
  fixed angle.
- **Warp** and **Push Warp** — distort an input by the gradient of an intensity map,
  omnidirectional or along a fixed angle.
- **Flow Blur** — blur along a control map's own gradient, strength scaled by its steepness.
- **Transform 2D** — offset, rotate, scale, and flip horizontal/vertical, with wrap-around
  tiling.
- **Edge Detect** — highlight where an input changes sharply.
- **Bevel** — rounds a mask or height input's edges into a beveled height map, with a
  Convexity control from a plain chamfer through a rounded dome to a concave scoop. Chain into
  Height to Normal for the usual beveled-edge normal look.

### Generators — Path
- **Path** — an authored or procedural vector path, the shared input every node below builds on.
- **Image Trace** — extract a path from an image's own edges.
- **Path Simplify** — reduce a path's point count while preserving its shape.
- **Path to Mask** and **Path to SDF** — rasterize a path to a filled mask or an exact signed
  distance field.
- **Fasteners** — rivets, bolts, or screws placed evenly along a path in 6 head shapes, seated
  on an optional base height.

### Converters
- **Height to Normal** — a tangent-space normal map from a height map, Sobel-based, with an
  OpenGL/DirectX toggle.
- **Height to AO** — a screen-space-style ambient occlusion pass from height.
- **Curvature from Normal** — convex edges read bright, concave crevices read dark.
- **Distance Field** — an exact signed distance field from any mask's own edge, not a blurred
  approximation.
- **Shape Boolean** — union/subtract/intersect two SDFs with a real boolean, in 5 edge-blend
  modes (hard, smooth, round, chamfer, groove).
- **Redistance** — rebuilds an exact distance field from any SDF's zero contour, erasing the
  hidden internal seams a boolean or warp can otherwise leave for a later Bevel to expose.
- **Bevel Profile** — shapes an exact SDF into a real height map through a chosen edge profile
  (flat, round, chamfer), instead of the usual blur-then-curve-then-histogram approximation.
- **Edge Wear** — chipped-paint wear on convex edges and a cavity mask for dirt/recesses, from
  an SDF (exact) or a height map (curvature-estimated) fallback.
- **Channel Packer** — pack up to 4 grayscale inputs into one RGBA texture's own channels.

### Outputs
One pass-through marker node per PBR channel: Base Color, Normal, Roughness, Metallic,
Height, Ambient Occlusion, Emissive, Alpha. Alpha drives real transparency in the 3D preview
(not just an exported mask) and exports as its own standalone grayscale map.

## Building from source

Requires a stable Rust toolchain ([rustup.rs](https://rustup.rs)) and a GPU with a
Vulkan/Metal/DX12 driver (`wgpu`'s supported backends — no OpenGL fallback).

- **macOS**: install the Xcode Command Line Tools if you haven't already
  (`xcode-select --install`). No other setup needed; `wgpu` uses Metal.
- **Windows**: rustup's default `stable-x86_64-pc-windows-msvc` toolchain is enough; it
  prompts to install the Visual Studio Build Tools (the "Desktop development with C++"
  workload) the first time you build if they're missing. `wgpu` uses DX12.
- **Linux**: `wgpu` uses Vulkan, so you need a Vulkan-capable driver (Mesa's RADV/ANV on
  AMD/Intel, or the proprietary driver on Nvidia), your distro's windowing dev packages, and
  `fontconfig` (the Type node's own font search) — on Debian/Ubuntu: `apt install
  libxkbcommon-dev libxkbcommon-x11-dev libx11-dev libxcursor-dev libxi-dev libwayland-dev
  mesa-vulkan-drivers libfontconfig1-dev`. (`libxkbcommon-x11-dev`/`libxcursor-dev`/`libxi-dev`
  are easy to miss and don't get pulled in by the rest — confirmed by actually running a
  packaged build on a bare Ubuntu container with only a shorter, previously-documented set
  installed: missing `libxkbcommon-x11` crashed with a real panic/backtrace, not just a clean
  error, the first time winit tried to set up its X11 keyboard handling.)

  **Downloaded the prebuilt release instead of building from source, and it won't start at
  all** (a terminal window flashing a "cannot open shared object file" error or a Rust panic,
  or nothing visible happening when you double-click it)? You're very likely missing one of
  the runtime libraries above — install the *runtime* package names instead of their `-dev`
  counterparts: `apt install libxkbcommon0 libxkbcommon-x11-0 libx11-6 libxcursor1 libxi6
  libwayland-client0 libwayland-cursor0 mesa-vulkan-drivers libfontconfig1`, then try again.
  Most desktop Linux installs already have all of these; this mainly comes up on a
  minimal/server-style install.

```sh
cargo run -p mc_app   # launch the desktop app
cargo run -p mc_cli   # headless: regenerates the bundled example graphs and exports into ./out

# headless: render any single saved graph file
cargo run -p mc_cli -- render path/to/material.mcgraph --preset blender --out ./out

# render N independent re-rolls of the graph's own Seed params, for
# quickly previewing how its jitter/seed-driven look varies
cargo run -p mc_cli -- render path/to/material.mcgraph --preset blender --out ./out --variations 5
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) — short version: sign off your commits (`git commit
-s`), keep `cargo fmt`/`cargo clippy`/`cargo test` clean, and don't submit anything derived
from Adobe Substance.

## License

Dual-licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

## Inspiration

MaterialCreator is an independent implementation, built from scratch for this project. It does
not use, reference, or reproduce any other procedural texturing tool's code, file formats,
trademarked names, or UI design — generic, industry-standard terms for node types (Levels,
Blend, Blur, Height to Normal, and so on) are the only overlap you should ever find with any
other tool in this space.

Its broader motivation — that ambitious, professional-grade creative software is worth
building in the open, for artists — takes a cue from [ArtCraft](https://github.com/storytold/artcraft),
an open-source creative tool for artists and filmmakers.

---

MaterialCreator is an independent open-source project and is not affiliated with, endorsed by,
or sponsored by Adobe. Adobe and Adobe Substance 3D are either registered trademarks or
trademarks of Adobe in the United States and/or other countries.
