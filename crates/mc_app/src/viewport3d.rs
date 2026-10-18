// SPDX-License-Identifier: MIT OR Apache-2.0
//! The 3D preview (CLAUDE.md §9's "3D View"): a render pipeline (as opposed
//! to the compute pipelines everywhere else in the app) that draws a sphere,
//! cube, or plane lit by one directional light plus a procedural sky,
//! textured from the graph's own Output nodes — falling back to flat
//! defaults for any channel the graph doesn't produce.
//!
//! Blender/Unreal analogy: this is a tiny built-in viewport renderer, the
//! same idea as Blender's Material Preview shading mode or Unreal's material
//! preview sphere, just scoped to exactly what CLAUDE.md asks for.

use std::collections::HashMap;

use glam::{Mat4, Vec3};

use crate::{tess_gpu, tess_mesh};

const VERTEX_ATTRS: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x3];

struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    /// Direction of increasing U across the surface — filled in by
    /// `compute_tangents` after a mesh's positions/normals/UVs are built, not
    /// set by hand per-generator. Lets the shader build a real tangent-space
    /// basis for normal mapping instead of faking one from the normal alone,
    /// which only looked right on the Plane (see `compute_tangents`'s doc
    /// comment) by coincidence.
    tangent: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MeshKind {
    Sphere,
    Cube,
    Plane,
    RoundedCube,
    RoundedCylinder,
}

impl MeshKind {
    pub const ALL: [MeshKind; 5] = [
        MeshKind::Sphere,
        MeshKind::Cube,
        MeshKind::Plane,
        MeshKind::RoundedCube,
        MeshKind::RoundedCylinder,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MeshKind::Sphere => "Sphere",
            MeshKind::Cube => "Cube",
            MeshKind::Plane => "Plane",
            MeshKind::RoundedCube => "Rounded Cube",
            MeshKind::RoundedCylinder => "Rounded Cylinder",
        }
    }
}

/// Which of the two render paths the current mesh needs: a plain indexed
/// draw from a CPU-built vertex/index buffer (every mesh kind, and Cube/
/// Rounded Cube with no Height node connected), or the GPU vertex-pulling
/// tessellation path (Cube/Rounded Cube with Height connected — see
/// `viewport3d.wgsl`'s `vs_main_tess` module doc comment). Built by
/// `build_and_upload_mesh`.
enum CurrentMesh {
    Uniform {
        vertex_buffer: wgpu::Buffer,
        index_buffer: wgpu::Buffer,
        index_count: u32,
    },
    Tessellated {
        // Kept alive for as long as `bind_group` references them — never
        // read from again directly once the bind group exists.
        _vertex_storage: wgpu::Buffer,
        _tri_storage: wgpu::Buffer,
        _inst_tri_storage: wgpu::Buffer,
        bind_group: wgpu::BindGroup,
        /// `3 * n_global^2` — the non-indexed per-instance vertex count
        /// `draw()` needs.
        vertices_per_instance: u32,
        /// One instance per base triangle (uniform tessellation: every
        /// instance subdivides to the same `n_global`, so there's no
        /// per-level draw splitting to do — see `tess_gpu::identity_inst_tri`).
        instance_count: u32,
        /// `tessellation`, after `tess_gpu::budget_n` reduces it to stay
        /// within `MAX_TESS_TRIANGLES` — written into the uniform buffer
        /// every frame (`vs_main_tess` needs it, and it can differ from the
        /// artist's own typed-in Tessellation value).
        n_global: u32,
        tex_size: u32,
        mip_count: u32,
    },
}

/// The vertex-pulling tessellation path's own pipelines and bind group
/// layout — same fragment shader and pixel-level behavior as `pipeline`/
/// `pipeline_transparent`, just a different vertex stage (`vs_main_tess`, no
/// vertex buffer) and an extra bind group (see `viewport3d.wgsl`'s own
/// module doc comment on `vs_main_tess`). `None` on a device that can't
/// supply `TESS_STORAGE_BUFFERS_NEEDED` VERTEX-stage storage buffers — see
/// `device_supports_tess`.
struct TessResources {
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    pipeline_transparent: wgpu::RenderPipeline,
}

/// The vertex-pulling tessellation bind group (`storage_entry(0..3)`, see
/// its own doc comment) needs exactly this many VERTEX-stage storage
/// buffers. Some real GPU/driver combinations report 0 here — confirmed on
/// an older Intel Haswell iGPU, under both its Vulkan (Mesa ANV) and GL
/// fallback paths — and even just constructing a `BindGroupLayout` asking
/// for more than a device's own limit panics inside wgpu's own validation,
/// regardless of whether anything ever actually uses it.
const TESS_STORAGE_BUFFERS_NEEDED: u32 = 3;

fn device_supports_tess(limits: &wgpu::Limits) -> bool {
    limits.max_storage_buffers_per_shader_stage >= TESS_STORAGE_BUFFERS_NEEDED
}

/// The 7 PBR maps the shader reads, each falling back to a flat default
/// texture when the graph has no node for that Output kind.
struct FallbackTextures {
    base_color: wgpu::TextureView,
    normal: wgpu::TextureView,
    roughness: wgpu::TextureView,
    metallic: wgpu::TextureView,
    ao: wgpu::TextureView,
    height: wgpu::TextureView,
    alpha: wgpu::TextureView,
}

pub struct Viewport3d {
    pipeline: wgpu::RenderPipeline,
    pipeline_transparent: wgpu::RenderPipeline,
    tess: Option<TessResources>,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    sampler_smooth: wgpu::Sampler,
    sampler_pixel: wgpu::Sampler,
    // Built lazily and rebuilt whenever `(mesh_kind, tessellation)` changes,
    // rather than every `MeshKind` pre-built once at a fixed quality — see
    // `tessellated_mesh`.
    current_mesh: CurrentMesh,
    /// `(mesh_kind, tessellation, height_evaluation_hash)` — the third
    /// field is what makes a Cube/Rounded Cube's tessellated mesh (see
    /// `build_and_upload_mesh`) rebuild when the Height content
    /// itself changes, not just when `mesh_kind`/`tessellation` do; `None`
    /// when the graph has no Output: Height node at all (the plain
    /// uniform-grid path, same as every other mesh kind).
    current_mesh_key: (MeshKind, u32, Option<u64>),
    fallback: FallbackTextures,
    environment_view: wgpu::TextureView,
    environment_ambient_color: Vec3,
    pub environment_source: EnvironmentSource,
    /// Set whenever the environment changes; `render()` regenerates and
    /// (re-)registers `environment_thumbnail_id` the next time it runs, the
    /// same lazy-rebuild-on-change pattern as `current_mesh_key`. Starts
    /// `true` so the default environment's thumbnail exists from the first
    /// frame without `new()` needing an `egui_wgpu::RenderState` of its own.
    environment_thumbnail_dirty: bool,
    environment_thumbnail_id: Option<egui::TextureId>,

    color_view: Option<wgpu::TextureView>,
    depth_view: Option<wgpu::TextureView>,
    texture_id: Option<egui::TextureId>,
    size: (u32, u32),

    pub mesh_kind: MeshKind,
    pub height_scale: f32,
    /// At least `TESSELLATION_MIN` — the exact subdivision count along a
    /// Cube/Rounded Cube edge (other mesh kinds scale proportionally, see
    /// `tessellated_mesh`), or the requested per-base-triangle lattice
    /// density `N` for the GPU vertex-pulling tessellation path when a
    /// Height node is connected (see `build_and_upload_mesh`/
    /// `tess_gpu::budget_n` — the actual `N` used can be lower, reduced to
    /// stay within `MAX_TESS_TRIANGLES`).
    pub tessellation: u32,
    pub shading_mode: ShadingMode,
    pub exposure: f32,
    /// Scales just the environment's contribution (sky reflection + ambient
    /// fill), independent of `exposure` above — lets a too-bright HDRI be
    /// dimmed without affecting direct light.
    pub environment_intensity: f32,
    /// 0 = exact mirror reflection of the environment; 1 = heavily blurred.
    /// Independent of surface roughness (which already dampens reflection
    /// *strength* — see `fs_main`'s `specular_strength`): a perfectly smooth
    /// metal has roughness ~0 and so no damping at all, which is exactly the
    /// case this is for — letting the artist soften that mirror without
    /// having to fake it by raising roughness.
    pub environment_blur: f32,
    pub texture_filter: TextureFilter,
    yaw: f32,
    pitch: f32,
    distance: f32,
    /// World-space point the camera orbits around and looks at — shifted by
    /// right-click-drag panning (`handle_input`); the mesh itself always
    /// stays at the origin, only the camera moves.
    target: Vec3,
}

/// "Smooth" is trilinear + 16x anisotropic — the normal case. "Pixel"
/// disables all interpolation (texel and mip), for inspecting dither/
/// pixel-art materials without any blending softening the exact pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextureFilter {
    #[default]
    Smooth,
    Pixel,
}

/// Which HDRI is currently lighting the scene — a bundled preset, or a file
/// the user loaded themselves. Kept on `Viewport3d` (rather than recomputed
/// each frame) purely so the Environment panel can show which one is active;
/// the texture/ambient color it produced are what actually drive rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvironmentSource {
    Preset(crate::hdri::HdriPreset),
    /// The file name only (not the full path) — just for display, since
    /// nothing needs to re-read the file after it's loaded.
    Custom(String),
}

/// A debug ground-truth view (CLAUDE.md §8's linear/sRGB discipline):
/// bypasses PBR shading and exposure/tonemapping entirely, outputting Base
/// Color with just the sRGB encode applied — for comparing pixel-for-pixel
/// against the 2D View and an exported PNG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShadingMode {
    #[default]
    Lit,
    UnlitBaseColor,
}

impl Viewport3d {
    pub fn new(gpu: &mc_gpu::Context) -> Self {
        let device = &gpu.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport3d"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/viewport3d.wgsl").into()),
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewport3d uniforms layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("viewport3d textures layout"),
                entries: &[
                    texture_entry(0, wgpu::ShaderStages::FRAGMENT),
                    texture_entry(1, wgpu::ShaderStages::FRAGMENT),
                    texture_entry(2, wgpu::ShaderStages::FRAGMENT),
                    texture_entry(3, wgpu::ShaderStages::FRAGMENT),
                    texture_entry(4, wgpu::ShaderStages::FRAGMENT),
                    // height_tex and tex_sampler need VERTEX visibility too
                    // — `vs_main` samples height there for real vertex
                    // displacement (see `viewport3d.wgsl`'s module doc
                    // comment).
                    texture_entry(5, wgpu::ShaderStages::VERTEX_FRAGMENT),
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    texture_entry(7, wgpu::ShaderStages::FRAGMENT),
                    texture_entry(8, wgpu::ShaderStages::FRAGMENT),
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport3d pipeline layout"),
            bind_group_layouts: &[
                Some(&uniform_bind_group_layout),
                Some(&texture_bind_group_layout),
            ],
            immediate_size: 0,
        });

        // Four pipeline variants: {plain vertex-buffer, GPU vertex-pulling
        // tessellation} x {opaque, transparent} — the latter pair differing
        // only in depth write, same reasoning as the former (see
        // `render()`'s doc comment on why a graph with an Alpha node needs
        // it off). Built from a closure rather than four near-copies of the
        // same ~30-line descriptor.
        let build_pipeline = |label: &'static str,
                              layout: &wgpu::PipelineLayout,
                              entry_point: &'static str,
                              buffers: &[Option<wgpu::VertexBufferLayout>],
                              depth_write_enabled: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        // Real alpha blending, not a flat overwrite, so the
                        // Output: Alpha channel (see viewport3d.wgsl's
                        // module doc comment) actually shows the background
                        // through a low-alpha area instead of being
                        // silently discarded. A graph with no Alpha node
                        // falls back to a fully opaque 1.0 (see
                        // `FallbackTextures::alpha`), which blends
                        // identically to the old `REPLACE` behavior, so
                        // this is a no-op for every existing graph.
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    // Preview-only: not worth getting every mesh's winding
                    // order exactly right when there's no performance need
                    // to cull.
                    cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(depth_write_enabled),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let vertex_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRS,
        })];
        let pipeline = build_pipeline(
            "viewport3d pipeline (opaque)",
            &pipeline_layout,
            "vs_main",
            &vertex_buffers,
            true,
        );
        // With `cull_mode: None`, several faces of a closed mesh (e.g. a
        // cube's far side) can land on the same screen pixel — depth write
        // is what makes that resolve correctly for fully-opaque geometry.
        // But once alpha blending is actually doing something (a real Alpha
        // node wired in), keeping depth write on breaks it: whichever
        // partially-transparent face happens to be drawn first still writes
        // full depth, so a face behind it that should show through gets
        // rejected by the depth test regardless of how transparent the
        // first one was — seen live as background "partially" showing
        // through in a way that depends on draw order, not on the actual
        // alpha value. Turning depth write off for this pipeline variant
        // fixes that; the known tradeoff (noted in viewport3d.wgsl's module
        // doc comment) is that it can't correctly sort two *different*
        // transparent layers of the same mesh against each other, only
        // against the opaque background — an acceptable simplification for
        // a material preview, not a compositor.
        let pipeline_transparent = build_pipeline(
            "viewport3d pipeline (transparent)",
            &pipeline_layout,
            "vs_main",
            &vertex_buffers,
            false,
        );
        // Gated on `device_supports_tess`: some GPU/driver combinations
        // (confirmed on an older Intel Haswell iGPU, under both Vulkan and
        // its GL fallback) report 0 VERTEX-stage storage buffers, and even
        // just constructing this bind group layout on such a device panics
        // inside wgpu's own validation — see `storage_entry`'s own doc
        // comment for what the 3 bindings are. `build_and_upload_mesh`
        // checks `self.tess.is_some()` before ever trying to use any of
        // this, falling back to the plain uniform-grid mesh the same way it
        // already does for every mesh kind that doesn't use tessellation at
        // all.
        let tess = if device_supports_tess(&device.limits()) {
            let tess_bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("viewport3d tess layout"),
                    entries: &[storage_entry(0), storage_entry(1), storage_entry(2)],
                });
            let pipeline_layout_tess =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("viewport3d tess pipeline layout"),
                    bind_group_layouts: &[
                        Some(&uniform_bind_group_layout),
                        Some(&texture_bind_group_layout),
                        Some(&tess_bind_group_layout),
                    ],
                    immediate_size: 0,
                });
            let pipeline = build_pipeline(
                "viewport3d tess pipeline (opaque)",
                &pipeline_layout_tess,
                "vs_main_tess",
                &[],
                true,
            );
            let pipeline_transparent = build_pipeline(
                "viewport3d tess pipeline (transparent)",
                &pipeline_layout_tess,
                "vs_main_tess",
                &[],
                false,
            );
            Some(TessResources {
                bind_group_layout: tess_bind_group_layout,
                pipeline,
                pipeline_transparent,
            })
        } else {
            None
        };

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport3d uniforms"),
            size: UNIFORM_SIZE as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport3d uniforms bind group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Anisotropic filtering needs mag/min/mipmap filter all `Linear` to
        // take effect at all (a wgpu validation requirement) — sharpens
        // textures viewed at a glancing angle, where trilinear alone still
        // blurs heavily since it can only pick an isotropic (square) mip
        // footprint for a genuinely elongated one.
        let sampler_smooth = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewport3d sampler (smooth)"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });
        // No interpolation at all, mip or texel — for inspecting dither/
        // pixel-art materials crisply, where any blending (even between
        // adjacent texels at mip 0) muddies the exact pattern.
        let sampler_pixel = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("viewport3d sampler (pixel)"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let initial_tessellation = 64;
        let current_mesh = upload_uniform_mesh(
            device,
            tessellated_mesh(MeshKind::RoundedCube, initial_tessellation),
        );
        let current_mesh_key = (MeshKind::RoundedCube, initial_tessellation, None);

        let fallback = FallbackTextures {
            base_color: solid_view(
                device,
                &gpu.queue,
                "fallback base color",
                [140, 140, 140, 255],
            ),
            normal: solid_view(device, &gpu.queue, "fallback normal", [128, 128, 255, 255]),
            // Fully rough (and, via `metallic` below, fully non-metal) by
            // default, so connecting a node to either Output clearly reads
            // as a change from a neutral matte baseline rather than from an
            // arbitrary partial value.
            roughness: solid_view(
                device,
                &gpu.queue,
                "fallback roughness",
                [255, 255, 255, 255],
            ),
            metallic: solid_view(device, &gpu.queue, "fallback metallic", [0, 0, 0, 255]),
            ao: solid_view(device, &gpu.queue, "fallback ao", [255, 255, 255, 255]),
            height: solid_view(device, &gpu.queue, "fallback height", [128, 128, 128, 255]),
            // Fully opaque by default — a graph with no Alpha node should
            // render exactly as it did before Alpha existed.
            alpha: solid_view(device, &gpu.queue, "fallback alpha", [255, 255, 255, 255]),
        };

        let initial_preset = crate::hdri::HdriPreset::StudioSmall03;
        let (environment_view, environment_ambient_color) =
            crate::hdri::load_preset(device, &gpu.queue, initial_preset);
        let environment_ambient_color = Vec3::from(environment_ambient_color);

        Self {
            pipeline,
            pipeline_transparent,
            tess,
            uniform_buffer,
            uniform_bind_group,
            texture_bind_group_layout,
            sampler_smooth,
            sampler_pixel,
            current_mesh,
            current_mesh_key,
            fallback,
            environment_view,
            environment_ambient_color,
            environment_source: EnvironmentSource::Preset(initial_preset),
            environment_thumbnail_dirty: true,
            environment_thumbnail_id: None,
            color_view: None,
            depth_view: None,
            texture_id: None,
            size: (0, 0),
            mesh_kind: MeshKind::RoundedCube,
            height_scale: 0.15,
            tessellation: initial_tessellation,
            shading_mode: ShadingMode::default(),
            exposure: 1.0,
            environment_intensity: 1.0,
            environment_blur: 0.0,
            texture_filter: TextureFilter::default(),
            yaw: 0.6,
            pitch: 0.35,
            distance: 3.0,
            target: Vec3::ZERO,
        }
    }

    /// `false` on a device that can't run the GPU vertex-pulling
    /// tessellation path at all (see `device_supports_tess`) — `mc_app::main`
    /// checks this right after construction to let the artist know why
    /// Cube/Rounded Cube's Height detail looks flat-faceted instead of
    /// silently downgrading with no explanation.
    pub fn tess_supported(&self) -> bool {
        self.tess.is_some()
    }

    /// Switches the sky/reflection environment to a bundled preset. Infallible
    /// — presets are checked into the repo and covered by a decode test, so
    /// any failure here would mean a corrupted build, not bad user input.
    pub fn set_environment_preset(
        &mut self,
        gpu: &mc_gpu::Context,
        preset: crate::hdri::HdriPreset,
    ) {
        let (view, ambient) = crate::hdri::load_preset(&gpu.device, &gpu.queue, preset);
        self.environment_view = view;
        self.environment_ambient_color = Vec3::from(ambient);
        self.environment_source = EnvironmentSource::Preset(preset);
        self.environment_thumbnail_dirty = true;
    }

    /// Switches the sky/reflection environment to a user-supplied `.hdr`/
    /// `.exr` file. Unlike [`Self::set_environment_preset`], every failure
    /// mode here is reachable from an arbitrary file the user picked, so the
    /// current environment is left untouched on `Err` — the caller is
    /// expected to surface it as a toast rather than crash the app.
    pub fn set_environment_custom(
        &mut self,
        gpu: &mc_gpu::Context,
        path: &std::path::Path,
    ) -> Result<(), String> {
        let (view, ambient) = crate::hdri::load_custom(&gpu.device, &gpu.queue, path)?;
        self.environment_view = view;
        self.environment_ambient_color = Vec3::from(ambient);
        self.environment_source = EnvironmentSource::Custom(
            path.file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.display().to_string()),
        );
        self.environment_thumbnail_dirty = true;
        Ok(())
    }

    /// The active environment's preview thumbnail, registered as an egui
    /// texture — `None` only for the first frame or two after startup or an
    /// environment change, before `render()` has run once to build it.
    pub fn environment_thumbnail(&self) -> Option<egui::TextureId> {
        self.environment_thumbnail_id
    }

    /// Orbit (left-drag), pan (right-drag), and zoom (scroll, while hovered)
    /// camera controls.
    pub fn handle_input(&mut self, ui: &egui::Ui, response: &egui::Response) {
        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            // Pan speed scales with distance so a dragged-screen-pixel keeps
            // tracking roughly the same world-space point regardless of
            // zoom level — fixed FOV means more world space fits in a pixel
            // the farther the camera is.
            let pan_speed = self.distance * 0.0015;
            let forward = -self.orbit_offset().normalize_or_zero();
            let right = forward.cross(Vec3::Y).normalize_or_zero();
            let up = right.cross(forward).normalize_or_zero();
            // Move the camera opposite the drag so the *content* tracks the
            // cursor (the same "grab the canvas" convention `canvas.rs`'s 2D
            // pan uses) — screen Y grows downward, world up doesn't, hence
            // the sign flip on the vertical term.
            self.target -= right * delta.x * pan_speed;
            self.target += up * delta.y * pan_speed;
        } else if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            self.yaw -= delta.x * 0.01;
            self.pitch = (self.pitch - delta.y * 0.01).clamp(-1.4, 1.4);
        }
        if response.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.distance = (self.distance - scroll * 0.01).clamp(1.5, 10.0);
            }
        }
    }

    /// The camera's offset from `target` — orbit radius/angles only, with no
    /// pan applied. `eye_position` below is just this plus `target`; pan
    /// (`handle_input`) only ever moves `target`, never this offset.
    fn orbit_offset(&self) -> Vec3 {
        Vec3::new(
            self.distance * self.pitch.cos() * self.yaw.sin(),
            self.distance * self.pitch.sin(),
            self.distance * self.pitch.cos() * self.yaw.cos(),
        )
    }

    fn eye_position(&self) -> Vec3 {
        self.target + self.orbit_offset()
    }

    fn recreate_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport3d color target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport3d depth target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.color_view = Some(color_texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.depth_view = Some(depth_texture.create_view(&wgpu::TextureViewDescriptor::default()));
    }

    fn build_texture_bind_group(
        &self,
        gpu: &mut mc_gpu::Context,
        pool: &mc_gpu::TexturePool,
        graph: &mc_core::Graph,
    ) -> wgpu::BindGroup {
        // Resolves `kind`'s channel to a view spanning its *entire* mip
        // chain (regenerating that chain first), or the 1x1 fallback —
        // which has only one mip to begin with, so it's used as-is.
        // `wgpu::TextureView` is a cheap `Clone` (a GPU-side handle, not a
        // borrow of `pool`), so there's no need to keep anything alive
        // beyond this closure call.
        let mut channel_view = |kind: &str, is_normal_map: bool, fallback: &wgpu::TextureView| {
            let Some(key) = resolve_channel_key(graph, kind) else {
                return fallback.clone();
            };
            // This bind group's own sampler (binding 6) always filters —
            // every real channel texture is `Rgba16Float`, which supports
            // that, *except* a node whose own output is deliberately
            // higher-precision (so far just Gabor Noise's `raw`, stored as
            // `Rgba32Float`, which WebGPU never allows filtered sampling
            // of without an opt-in device feature this app doesn't
            // request). Falling back here — treating it the same as
            // nothing meaningfully connected — is the same call its own
            // doc comment already makes ("for math nodes downstream, not
            // direct display"); the alternative (a second, non-filtering
            // bind group layout + pipeline variant just for this one
            // case) is a lot of machinery for a channel that was never a
            // sensible thing to view directly in the first place.
            let Some(texture) = pool.texture(&key) else {
                return fallback.clone();
            };
            if texture.format() != wgpu::TextureFormat::Rgba16Float {
                return fallback.clone();
            }
            crate::mipgen::generate_mip_chain(gpu, pool, &key, is_normal_map);
            crate::mipgen::full_mip_view(pool, &key).unwrap_or_else(|| fallback.clone())
        };

        let base_color = channel_view("output_base_color", false, &self.fallback.base_color);
        let normal = channel_view("output_normal", true, &self.fallback.normal);
        let roughness = channel_view("output_roughness", false, &self.fallback.roughness);
        let metallic = channel_view("output_metallic", false, &self.fallback.metallic);
        let ao = channel_view("output_ambient_occlusion", false, &self.fallback.ao);
        let height = channel_view("output_height", false, &self.fallback.height);
        let alpha = channel_view("output_alpha", false, &self.fallback.alpha);

        let device = &gpu.device;
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport3d textures bind group"),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_color),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&normal),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&roughness),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&metallic),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&ao),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&height),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(match self.texture_filter {
                        TextureFilter::Smooth => &self.sampler_smooth,
                        TextureFilter::Pixel => &self.sampler_pixel,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&self.environment_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&alpha),
                },
            ],
        })
    }

    /// Builds whatever mesh `mesh_kind` needs at the given `tessellation`,
    /// using the GPU vertex-pulling tessellation path (see
    /// `viewport3d.wgsl`'s `vs_main_tess` module doc comment) for Cube/
    /// Rounded Cube when the graph actually has an Output: Height node —
    /// real vertex displacement needs real geometry to displace, and a
    /// uniform grid's own fixed vertex budget can't resolve a hard height
    /// edge (text, a crisp cutout) as finely as this can, without the
    /// CPU-built-mesh problems `mc_app::adaptive_mesh` (since removed)
    /// eventually hit: a hard wgpu per-buffer size ceiling, and a
    /// multi-second UI freeze rebuilding/re-uploading the mesh on every
    /// Tessellation/Resolution change. Every other mesh kind, and Cube/
    /// Rounded Cube with no Height connected at all, keep using the plain
    /// uniform grid — not worth a base-mesh rebuild for a flat fallback
    /// texture that was never going to show any detail anyway.
    fn build_and_upload_mesh(
        &self,
        mesh_kind: MeshKind,
        tessellation: u32,
        gpu: &mut mc_gpu::Context,
        pool: &mc_gpu::TexturePool,
        graph: &mc_core::Graph,
    ) -> CurrentMesh {
        use wgpu::util::DeviceExt;

        let radius = match mesh_kind {
            MeshKind::Cube => 0.0,
            MeshKind::RoundedCube => 0.25,
            _ => {
                return upload_uniform_mesh(&gpu.device, tessellated_mesh(mesh_kind, tessellation))
            }
        };
        // No tess resources on this device (see `device_supports_tess`) —
        // same plain-grid fallback as every other case here, just driven by
        // hardware instead of the graph.
        let Some(tess) = &self.tess else {
            return upload_uniform_mesh(&gpu.device, tessellated_mesh(mesh_kind, tessellation));
        };
        let Some(height_key) = resolve_channel_key(graph, "output_height") else {
            return upload_uniform_mesh(&gpu.device, tessellated_mesh(mesh_kind, tessellation));
        };
        if pool.texture(&height_key).is_none() {
            return upload_uniform_mesh(&gpu.device, tessellated_mesh(mesh_kind, tessellation));
        }

        // Reuses the same real, box-filtered GPU mip chain every channel the
        // viewport reads already gets (`mipgen.rs`) — matching exactly what
        // this technique's own lam/lod math assumes (see
        // `tess_mesh::build_mips`'s doc comment) — so no separate CPU
        // readback or texture is needed just for tessellation.
        crate::mipgen::generate_mip_chain(gpu, pool, &height_key, false);
        let texture = pool
            .texture(&height_key)
            .expect("just confirmed present above");
        let tex_size = texture.width();
        let mip_count = texture.mip_level_count();

        let base = tess_mesh::make_rounded_cube(BASE_FILLET_SEGMENTS, radius, 2.0);
        let base_tri_count = base.tri.len() as u32;
        let mesh = tess_mesh::prepare(base, tex_size);
        // Reduces a large typed-in Tessellation value (up to the graph's own
        // output Resolution, e.g. 4096) so the resulting instanced draw
        // stays within a real GPU time-per-frame budget — see
        // `tess_gpu::budget_n`'s own doc comment on why there's no longer a
        // hard memory wall to avoid instead.
        let n_global = tess_gpu::budget_n(tessellation, base_tri_count, MAX_TESS_TRIANGLES);
        let nt = vec![n_global; base_tri_count as usize];
        let edge_n = vec![[n_global; 3]; base_tri_count as usize];

        let device = &gpu.device;
        let vertex_storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewport3d tess vertices"),
            contents: &tess_gpu::pack_vertex_buffer(&mesh),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let tri_storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewport3d tess triangles"),
            contents: &tess_gpu::pack_tri_buffer(&mesh, &nt, &edge_n),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let inst_tri_bytes: Vec<u8> = tess_gpu::identity_inst_tri(base_tri_count)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let inst_tri_storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("viewport3d tess inst_tri"),
            contents: &inst_tri_bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("viewport3d tess bind group"),
            layout: &tess.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: vertex_storage.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tri_storage.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: inst_tri_storage.as_entire_binding(),
                },
            ],
        });

        CurrentMesh::Tessellated {
            _vertex_storage: vertex_storage,
            _tri_storage: tri_storage,
            _inst_tri_storage: inst_tri_storage,
            bind_group,
            vertices_per_instance: 3 * n_global * n_global,
            instance_count: base_tri_count,
            n_global,
            tex_size,
            mip_count,
        }
    }

    /// Records the current mesh's draw call — picks the plain-vertex-buffer
    /// or vertex-pulling-tessellation pipeline/bind-group-2 based on which
    /// variant `self.current_mesh` currently is (see [`CurrentMesh`]).
    /// Shared between `render()` and the test module's own
    /// `render_offscreen`, which do identical GPU work otherwise.
    fn record_draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        texture_bind_group: &wgpu::BindGroup,
        transparent: bool,
    ) {
        pass.set_bind_group(0, &self.uniform_bind_group, &[]);
        pass.set_bind_group(1, texture_bind_group, &[]);
        match &self.current_mesh {
            CurrentMesh::Uniform {
                vertex_buffer,
                index_buffer,
                index_count,
            } => {
                pass.set_pipeline(if transparent {
                    &self.pipeline_transparent
                } else {
                    &self.pipeline
                });
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*index_count, 0, 0..1);
            }
            CurrentMesh::Tessellated {
                bind_group,
                vertices_per_instance,
                instance_count,
                ..
            } => {
                // Only ever constructed by `build_and_upload_mesh`'s own
                // `self.tess` branch, so `self.tess` existing here is
                // guaranteed, not assumed.
                let tess = self
                    .tess
                    .as_ref()
                    .expect("CurrentMesh::Tessellated implies tess resources exist");
                pass.set_pipeline(if transparent {
                    &tess.pipeline_transparent
                } else {
                    &tess.pipeline
                });
                pass.set_bind_group(2, bind_group, &[]);
                pass.draw(0..*vertices_per_instance, 0..*instance_count);
            }
        }
    }

    /// Renders the current mesh lit by the current graph's outputs into an
    /// offscreen target sized `width x height`, and returns the egui texture
    /// id it's registered under (stable across frames and resizes, so the UI
    /// side can just keep displaying the same id).
    pub fn render(
        &mut self,
        gpu: &mut mc_gpu::Context,
        pool: &mc_gpu::TexturePool,
        graph: &mc_core::Graph,
        render_state: &egui_wgpu::RenderState,
        width: u32,
        height: u32,
    ) -> egui::TextureId {
        let width = width.max(1);
        let height = height.max(1);
        if self.size != (width, height) || self.color_view.is_none() {
            self.recreate_targets(&gpu.device, width, height);
            self.size = (width, height);
        }

        if self.environment_thumbnail_dirty {
            let thumbnail_view = crate::hdri::render_thumbnail(gpu, &self.environment_view);
            self.environment_thumbnail_id = Some(match self.environment_thumbnail_id {
                Some(texture_id) => {
                    render_state
                        .renderer
                        .write()
                        .update_egui_texture_from_wgpu_texture(
                            &gpu.device,
                            &thumbnail_view,
                            wgpu::FilterMode::Linear,
                            texture_id,
                        );
                    texture_id
                }
                None => render_state.renderer.write().register_native_texture(
                    &gpu.device,
                    &thumbnail_view,
                    wgpu::FilterMode::Linear,
                ),
            });
            self.environment_thumbnail_dirty = false;
        }

        let texture_bind_group = self.build_texture_bind_group(gpu, pool, graph);

        let desired_key = (
            self.mesh_kind,
            self.tessellation,
            height_evaluation_hash(graph),
        );
        if desired_key != self.current_mesh_key {
            self.current_mesh =
                self.build_and_upload_mesh(self.mesh_kind, self.tessellation, gpu, pool, graph);
            self.current_mesh_key = desired_key;
        }
        let tess = match &self.current_mesh {
            CurrentMesh::Uniform { .. } => (0.0, 0.0, 0.0),
            CurrentMesh::Tessellated {
                n_global,
                tex_size,
                mip_count,
                ..
            } => (*n_global as f32, *tex_size as f32, *mip_count as f32),
        };

        let aspect = width as f32 / height as f32;
        let eye = self.eye_position();
        let view = glam::camera::rh::view::look_at_mat4(eye, self.target, Vec3::Y);
        let proj =
            glam::camera::rh::proj::directx::perspective(60f32.to_radians(), aspect, 0.1, 100.0);
        let light_dir = Vec3::new(-0.4, -1.0, -0.3).normalize();

        let bytes = uniform_bytes(
            proj * view,
            eye,
            light_dir,
            self.environment_ambient_color,
            RenderSettings {
                height_scale: self.height_scale,
                shading_mode: self.shading_mode,
                exposure: self.exposure,
                environment_intensity: self.environment_intensity,
                environment_blur: self.environment_blur,
                tess,
            },
        );
        gpu.queue.write_buffer(&self.uniform_buffer, 0, &bytes);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viewport3d encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport3d pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: self.color_view.as_ref().expect("just created"),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(crate::theme::NEUTRON_STAR_WGPU),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.depth_view.as_ref().expect("just created"),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            // See `pipeline_transparent`'s doc comment in `new()`: depth
            // write has to turn off once a real Alpha node is actually
            // driving transparency, or partially-transparent faces fight
            // the depth buffer in a way that depends on draw order rather
            // than the alpha value itself.
            let transparent = resolve_channel_key(graph, "output_alpha").is_some();
            self.record_draw(&mut pass, &texture_bind_group, transparent);
        }
        gpu.queue.submit(Some(encoder.finish()));

        let color_view = self.color_view.as_ref().expect("just created");
        match self.texture_id {
            Some(texture_id) => {
                render_state
                    .renderer
                    .write()
                    .update_egui_texture_from_wgpu_texture(
                        &gpu.device,
                        color_view,
                        wgpu::FilterMode::Linear,
                        texture_id,
                    );
                texture_id
            }
            None => {
                let texture_id = render_state.renderer.write().register_native_texture(
                    &gpu.device,
                    color_view,
                    wgpu::FilterMode::Linear,
                );
                self.texture_id = Some(texture_id);
                texture_id
            }
        }
    }
}

fn texture_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// A read-only storage buffer entry, VERTEX-only — the tess bind group's
/// own 3 bindings (`tess_verts`/`tess_tris`/`tess_inst_tri` in
/// `viewport3d.wgsl`) are all this same shape.
fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The graph's texture pool key for one Output kind (e.g.
/// `"output_base_color"`), if the graph has that Output node and it
/// resolves to something. Mirrors `mc_nodes::export::read_output`'s lookup,
/// minus the readback.
fn resolve_channel_key(graph: &mc_core::Graph, kind: &str) -> Option<String> {
    let id = graph
        .nodes()
        .find(|(_, n)| n.kind == kind)
        .map(|(id, _)| id)?;
    mc_nodes::registry::resolve_texture_key(graph, id)
}

/// A hash that changes exactly when the graph's Height output would
/// render differently — `None` if the graph has no Output: Height node at
/// all. Hashing the Output node itself (rather than, say, its upstream
/// source) is enough: `Graph::evaluation_hash` already recurses into
/// whatever's connected to it, so any upstream param edit changes this
/// too. Used to decide when a Cube/Rounded Cube's adaptive mesh (see
/// `generate_rounded_cube_adaptive`) actually needs rebuilding — a real
/// GPU readback plus quadtree build, too expensive to redo every frame
/// just because the camera moved.
fn height_evaluation_hash(graph: &mc_core::Graph) -> Option<u64> {
    let id = graph
        .nodes()
        .find(|(_, n)| n.kind == "output_height")
        .map(|(id, _)| id)?;
    graph.evaluation_hash(id)
}

/// The base mesh's own per-face fillet resolution for Cube/Rounded Cube's
/// tessellated path (`Viewport3d::build_and_upload_mesh`) — fixed,
/// independent of the user's own Tessellation slider. Unlike a flat cube
/// face, the rounded fillet's curve has to already be resolved at the base-
/// mesh level: vertex-pulling tessellation only adds *height-detail*
/// density within an existing base triangle via barycentric interpolation,
/// which stays exactly planar (3 corners' affine combinations never leave
/// their own plane) — it can't add curvature a low-poly base mesh doesn't
/// already have. 12 segments/face (1,728 base triangles for the 6-face
/// cube) is comfortably smooth for this fillet's radius at preview scale,
/// while leaving most of `MAX_TESS_TRIANGLES`'s budget for the user's own
/// Tessellation value to spend on actual height detail.
const BASE_FILLET_SEGMENTS: u32 = 12;

/// Upper bound on `base_triangles * n_global^2` for the tessellated path —
/// see `tess_gpu::budget_n`. Not a hardware/memory limit (the whole point
/// of this technique is that nothing scales a CPU-built buffer anymore);
/// this is purely a real-time-rendering-budget guess. With
/// `BASE_FILLET_SEGMENTS` at 1,728 base triangles, this caps `n_global`
/// around 190 (effective resolution ~2,300 across a face) — real,
/// measured-responsive on real hardware at the time this was tuned, but
/// adjust by feel: lower it if a high Tessellation value stutters, raise it
/// if the ceiling itself is visibly limiting on faster hardware. Uniform
/// tessellation (every part of the surface gets the same density) has a
/// fundamentally lower practical ceiling than the content-adaptive CPU mesh
/// this replaced (which only spent vertices where Height actually had
/// detail) — per-pixel normal mapping is what carries most of the
/// perceived fine detail regardless (see `viewport3d.wgsl`'s own module
/// doc comment on `vs_main_tess`), so this matters most for how crisp a
/// hard silhouette edge (text, a crisp cutout) looks, not overall surface
/// detail.
const MAX_TESS_TRIANGLES: u32 = 64_000_000;

fn solid_view(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    color: [u8; 4],
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &color,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn upload_uniform_mesh(
    device: &wgpu::Device,
    (vertices, indices): (Vec<Vertex>, Vec<u32>),
) -> CurrentMesh {
    let mut vertex_bytes = Vec::with_capacity(vertices.len() * std::mem::size_of::<Vertex>());
    for v in &vertices {
        for c in v.position {
            vertex_bytes.extend_from_slice(&c.to_le_bytes());
        }
        for c in v.normal {
            vertex_bytes.extend_from_slice(&c.to_le_bytes());
        }
        for c in v.uv {
            vertex_bytes.extend_from_slice(&c.to_le_bytes());
        }
        for c in v.tangent {
            vertex_bytes.extend_from_slice(&c.to_le_bytes());
        }
    }
    let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewport3d mesh vertices"),
        size: vertex_bytes.len() as u64,
        usage: wgpu::BufferUsages::VERTEX,
        mapped_at_creation: true,
    });
    vertex_buffer
        .slice(..)
        .get_mapped_range_mut()
        .expect("buffer was just created mapped")
        .copy_from_slice(&vertex_bytes);
    vertex_buffer.unmap();

    let mut index_bytes = Vec::with_capacity(indices.len() * 4);
    for i in &indices {
        index_bytes.extend_from_slice(&i.to_le_bytes());
    }
    let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("viewport3d mesh indices"),
        size: index_bytes.len() as u64,
        usage: wgpu::BufferUsages::INDEX,
        mapped_at_creation: true,
    });
    index_buffer
        .slice(..)
        .get_mapped_range_mut()
        .expect("buffer was just created mapped")
        .copy_from_slice(&index_bytes);
    index_buffer.unmap();

    CurrentMesh::Uniform {
        vertex_buffer,
        index_buffer,
        index_count: indices.len() as u32,
    }
}

/// `view_proj`(mat4) + `camera_pos`(vec4) + `light_dir`(vec4) +
/// `ambient_color`(vec4) + 5 scalars + 3 tess-only scalars = 144 bytes,
/// matching `Uniforms` in viewport3d.wgsl. The last 3 (`tess_n_global`/
/// `tess_tex_size`/`tess_mip_count`) are read only by `vs_main_tess` —
/// harmless zeros whenever `self.current_mesh` is `CurrentMesh::Uniform`.
const UNIFORM_SIZE: usize = 144;

/// The render-settings scalars that make up `uniform_bytes`' tail — grouped
/// into one struct rather than more positional args, which pushed that
/// function over clippy's `too_many_arguments` threshold.
struct RenderSettings {
    height_scale: f32,
    shading_mode: ShadingMode,
    exposure: f32,
    environment_intensity: f32,
    environment_blur: f32,
    /// `(n_global, tex_size, mip_count)` from the current
    /// `CurrentMesh::Tessellated`, or `(0.0, 0.0, 0.0)` when the mesh is a
    /// plain `CurrentMesh::Uniform` (unread by `vs_main` either way).
    tess: (f32, f32, f32),
}

fn uniform_bytes(
    view_proj: Mat4,
    camera_pos: Vec3,
    light_dir: Vec3,
    ambient_color: Vec3,
    settings: RenderSettings,
) -> [u8; UNIFORM_SIZE] {
    let mut bytes = [0u8; UNIFORM_SIZE];
    let mut offset = 0;
    let put = |value: f32, bytes: &mut [u8; UNIFORM_SIZE], offset: &mut usize| {
        bytes[*offset..*offset + 4].copy_from_slice(&value.to_le_bytes());
        *offset += 4;
    };
    for component in view_proj.to_cols_array() {
        put(component, &mut bytes, &mut offset);
    }
    for component in [camera_pos.x, camera_pos.y, camera_pos.z, 1.0] {
        put(component, &mut bytes, &mut offset);
    }
    for component in [light_dir.x, light_dir.y, light_dir.z, 0.0] {
        put(component, &mut bytes, &mut offset);
    }
    for component in [ambient_color.x, ambient_color.y, ambient_color.z, 0.0] {
        put(component, &mut bytes, &mut offset);
    }
    put(settings.height_scale, &mut bytes, &mut offset);
    let shading_mode_value = match settings.shading_mode {
        ShadingMode::Lit => 0.0,
        ShadingMode::UnlitBaseColor => 1.0,
    };
    put(shading_mode_value, &mut bytes, &mut offset);
    put(settings.exposure, &mut bytes, &mut offset);
    put(settings.environment_intensity, &mut bytes, &mut offset);
    put(settings.environment_blur, &mut bytes, &mut offset);
    put(settings.tess.0, &mut bytes, &mut offset);
    put(settings.tess.1, &mut bytes, &mut offset);
    put(settings.tess.2, &mut bytes, &mut offset);
    bytes
}

/// The Tessellation slider's own floor — matching artists coming from
/// Substance Designer picking an explicit subdivision count (16, 32...)
/// rather than tuning a vague 0..1 "density" value, but as a free
/// drag-or-type number (`egui::Slider`'s own `DragValue` already supports
/// typing an exact value, no separate widget needed) rather than a fixed
/// list.
pub const TESSELLATION_MIN: u32 = 4;

/// The Tessellation slider's own ceiling — deliberately *not* tied to the
/// graph's output Resolution anymore (an earlier version used
/// `graph.settings.resolution` directly, on the reasoning that tessellating
/// finer than the Height texture's own resolution can't reveal more real
/// *data*; still true, but it meant a 1024-res graph silently capped the
/// slider at 1024 even though a crisp height edge's own *silhouette* can
/// still visibly facet well past that, which is the whole reason this
/// slider exists). Safe to request regardless of Resolution: the
/// tessellated path (`build_and_upload_mesh`) reduces whatever's actually
/// achievable via `tess_gpu::budget_n`/`MAX_TESS_TRIANGLES`, a real-time-
/// rendering-time budget independent of this number, and the plain
/// uniform-grid path (every other mesh kind, and Cube/Rounded Cube with no
/// Height connected) separately clamps to `UNIFORM_MESH_SEGMENTS_MAX`
/// regardless of what this slider allows. 4096 is the highest value
/// actually measured (crash-free, ~35-120ms worst-case mesh rebuild, no
/// per-frame lag) against a real logo graph — raise it further only after
/// the same kind of hands-on check.
pub const TESSELLATION_MAX: u32 = 4096;

/// `tessellation` scaled by `ratio`, for mesh kinds whose own segment
/// counts were historically set relative to Cube/Rounded Cube's own
/// (rather than equal to it) — see `tessellated_mesh`'s own per-arm
/// comments for where each ratio comes from.
fn scaled(tessellation: u32, ratio: f32) -> u32 {
    ((tessellation as f32 * ratio).round() as u32).max(1)
}

/// Builds the given mesh kind's plain *uniform* grid at `tessellation`
/// segments (Cube/Rounded Cube) or a proportional multiple of it (every
/// other kind — see each match arm) — used for every mesh kind except
/// Cube/Rounded Cube with a Height node actually connected, which instead
/// gets the GPU vertex-pulling tessellation path from
/// `Viewport3d::build_and_upload_mesh` (see that method's own doc comment
/// for why). This segment count
/// bounds how finely a hard height edge can bend without faceting
/// wherever this uniform path is still used (every other mesh kind, and
/// Cube/Rounded Cube with no Height connected at all, where there's
/// nothing to displace anyway).
/// Safety ceiling for every *uniform*-grid mesh kind (every kind except
/// Cube/Rounded Cube with a Height node connected, which instead takes the
/// budget-protected GPU tessellation path — see `build_and_upload_mesh`).
/// This path builds an actual CPU vertex/index buffer sized directly by
/// `tessellation` with no other protection, so letting the Tessellation
/// slider's own range grow (it now goes well past this, independent of the
/// graph's output Resolution — see `main.rs`'s own slider) must not also
/// let this path ask for an unbounded buffer. 256 comfortably covers this
/// path's own historical range (segment counts here historically topped
/// out around 128-256 even before Tessellation became a free-typed value)
/// — there's no real visual reason to go higher on a *uniform* grid with
/// no Height to resolve finely in the first place.
const UNIFORM_MESH_SEGMENTS_MAX: u32 = 256;

fn tessellated_mesh(kind: MeshKind, tessellation: u32) -> (Vec<Vertex>, Vec<u32>) {
    let tessellation = tessellation.min(UNIFORM_MESH_SEGMENTS_MAX);
    match kind {
        // Historically ranged 8..128 (rings) / 16..192 (equator segments)
        // against Cube/Rounded Cube's own 4..128 — ratios 1x and 1.5x.
        MeshKind::Sphere => generate_sphere(tessellation, scaled(tessellation, 1.5)),
        MeshKind::Cube => generate_rounded_cube(tessellation, 0.0),
        // Historically ranged 8..256 against Cube/Rounded Cube's 4..128 —
        // ratio 2x.
        MeshKind::Plane => generate_plane(scaled(tessellation, 2.0)),
        MeshKind::RoundedCube => generate_rounded_cube(tessellation, 0.25),
        // Historically ranged 16..192 (radial) / 2..48 (fillet) / 2..96
        // (wall) against Cube/Rounded Cube's 4..128 — ratios 1.5x, 0.375x,
        // 0.75x.
        MeshKind::RoundedCylinder => generate_rounded_cylinder(
            scaled(tessellation, 1.5),
            scaled(tessellation, 0.375),
            scaled(tessellation, 0.75),
            0.3,
        ),
    }
}

fn generate_sphere(rings: u32, segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    for ring in 0..=rings {
        let v = ring as f32 / rings as f32;
        let phi = v * std::f32::consts::PI;
        let y = phi.cos();
        let r = phi.sin();
        for seg in 0..=segments {
            let u = seg as f32 / segments as f32;
            let theta = u * std::f32::consts::TAU;
            let x = r * theta.cos();
            let z = r * theta.sin();
            let position = [x, y, z];
            // At a pole, every vertex around this "ring" sits at the exact
            // same 3D point despite having a different u — height
            // displacement moves each by however much *its own* u happens
            // to sample, which tears that single point into a star/rosette.
            // Giving them all the same u instead means they displace
            // together, by construction. Same pinch every UV sphere/cone has
            // at its poles, just put to use here rather than worked around.
            let pole_safe_u = if r.abs() < 1e-5 { 0.0 } else { u };
            vertices.push(Vertex {
                position,
                normal: position,
                uv: [pole_safe_u, v],
                tangent: [0.0; 3],
            });
        }
    }

    let stride = segments + 1;
    let mut indices = Vec::new();
    for ring in 0..rings {
        for seg in 0..segments {
            let a = ring * stride + seg;
            let b = a + stride;
            let c = a + 1;
            let d = b + 1;
            indices.extend_from_slice(&[a, b, c, c, b, d]);
        }
    }
    compute_tangents(&mut vertices, &indices);
    (vertices, indices)
}

/// A cube with filleted edges/corners: each face is a `segments`×`segments`
/// grid on the unit cube, then every vertex is pulled outward from an
/// "inner" box (half-extent `1 - radius`) by `radius` along the direction
/// it was clamped in. On the flat middle of a face that direction is just
/// the face normal (so the vertex doesn't move at all); near an edge or
/// corner it's the diagonal away from the inner box, tracing out exactly the
/// quarter-cylinder/eighth-sphere fillet a real rounded box has there.
///
/// `radius: 0.0` degenerates cleanly to a plain sharp-edged cube (`clamped`
/// never actually clamps anything, so every vertex just stays where the grid
/// put it) — `MeshKind::Cube` uses exactly that rather than a separate
/// unsubdivided cube generator, reusing this one generator for both. Each
/// face's own shared-edge vertices get welded to their neighbor's
/// afterward (see [`weld_seam_vertices`]) — needed because height moves
/// vertices: without it, two faces' independently-generated copies of a
/// shared edge point (same position, different UVs) displace to different
/// final positions and tear the mesh open there.
fn generate_rounded_cube(segments: u32, radius: f32) -> (Vec<Vertex>, Vec<u32>) {
    let radius = radius.clamp(0.0, 1.0);
    let inner = 1.0 - radius;
    let faces: [([f32; 3], [f32; 3], [f32; 3]); 6] = [
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ];

    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for (normal, right, up) in faces {
        let normal = Vec3::from(normal);
        let right = Vec3::from(right);
        let up = Vec3::from(up);
        let base = vertices.len() as u32;
        for j in 0..=segments {
            let v = j as f32 / segments as f32 * 2.0 - 1.0;
            for i in 0..=segments {
                let u = i as f32 / segments as f32 * 2.0 - 1.0;
                let p = normal + right * u + up * v;
                let clamped = p.clamp(Vec3::splat(-inner), Vec3::splat(inner));
                let delta = p - clamped;
                let n = if delta.length_squared() > 1e-12 {
                    delta.normalize()
                } else {
                    normal
                };
                let position = clamped + n * radius;
                vertices.push(Vertex {
                    position: position.to_array(),
                    normal: n.to_array(),
                    uv: [i as f32 / segments as f32, 1.0 - j as f32 / segments as f32],
                    tangent: [0.0; 3],
                });
            }
        }
        let stride = segments + 1;
        for j in 0..segments {
            for i in 0..segments {
                let a = base + j * stride + i;
                let b = a + 1;
                let c = a + stride;
                let d = c + 1;
                indices.extend_from_slice(&[a, b, d, a, d, c]);
            }
        }
    }
    compute_tangents(&mut vertices, &indices);
    weld_seam_vertices(&vertices, &mut indices);
    (vertices, indices)
}

/// Welds vertices that land at the same physical position but came from
/// different faces — every edge of this per-face-UV-mapped cube has
/// exactly that (two faces' independently generated vertex grids meeting
/// at the same edge, but on opposite sides of a UV seam). Needed because
/// height actually displaces vertices (`viewport3d.wgsl`'s `vs_main`):
/// two different UVs sampling two different heights would otherwise pull
/// a shared edge point to two different final positions, tearing the mesh
/// open there. Welding forces both sides to share one vertex — one UV,
/// one height sample, one final position — so there's no seam left for
/// that to happen across, at the cost of a thin, one-row sliver where the
/// *losing* face's own edge row shows the winning face's color instead of
/// its own. Call after `compute_tangents`, not before — each vertex's
/// tangent should come from its own face's UV gradient, not get confused
/// by a weld that's about to make two differently-UV'd vertices share an
/// index.
fn weld_seam_vertices(vertices: &[Vertex], indices: &mut [u32]) {
    // Quantized to 1e-6 — far finer than any real edge length this mesh
    // ever produces (even at the maximum 128 segments, the shortest edge
    // is still several orders of magnitude longer), so this only ever
    // merges points that are the same position up to float rounding, never
    // two distinct-but-nearby vertices.
    const SCALE: f32 = 1_000_000.0;
    let key = |p: [f32; 3]| -> (i64, i64, i64) {
        (
            (p[0] * SCALE).round() as i64,
            (p[1] * SCALE).round() as i64,
            (p[2] * SCALE).round() as i64,
        )
    };

    let mut canonical: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut remap = vec![0u32; vertices.len()];
    for (i, v) in vertices.iter().enumerate() {
        let canonical_index = *canonical.entry(key(v.position)).or_insert(i as u32);
        remap[i] = canonical_index;
    }
    for idx in indices.iter_mut() {
        *idx = remap[*idx as usize];
    }
}

/// One point of the 2D (radial-distance, height) profile that
/// [`revolve`] sweeps around the Y axis to build a surface of revolution —
/// `(nr, ny)` is that profile curve's own outward normal, not a 3D one.
struct ProfilePoint {
    r: f32,
    y: f32,
    nr: f32,
    ny: f32,
}

/// Sweeps a profile curve around the Y axis in `segments` angular steps,
/// the same ring-by-ring construction [`generate_sphere`] uses for its
/// lines of latitude — a cylinder, cone, torus, or (as used by
/// [`generate_rounded_cylinder`]) a rounded cylinder are all just different
/// profiles through this one function.
fn revolve(profile: &[ProfilePoint], segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    for (ring, p) in profile.iter().enumerate() {
        let v = ring as f32 / (profile.len() - 1) as f32;
        for seg in 0..=segments {
            let u = seg as f32 / segments as f32;
            let theta = u * std::f32::consts::TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            // Same pole pinch as `generate_sphere`: at the center of a flat
            // cap (`p.r == 0`), every vertex around this ring is the same 3D
            // point — give them the same u so they all get the same height
            // displacement instead of tearing that point into a star.
            let pole_safe_u = if p.r.abs() < 1e-5 { 0.0 } else { u };
            vertices.push(Vertex {
                position: [p.r * cos_t, p.y, p.r * sin_t],
                normal: [p.nr * cos_t, p.ny, p.nr * sin_t],
                uv: [pole_safe_u, v],
                tangent: [0.0; 3],
            });
        }
    }

    let stride = segments + 1;
    let mut indices = Vec::new();
    for ring in 0..profile.len() as u32 - 1 {
        for seg in 0..segments {
            let a = ring * stride + seg;
            let b = a + stride;
            let c = a + 1;
            let d = b + 1;
            indices.extend_from_slice(&[a, b, c, c, b, d]);
        }
    }
    compute_tangents(&mut vertices, &indices);
    (vertices, indices)
}

/// A cylinder with filleted top/bottom rim edges, built as a surface of
/// revolution: flat top cap → quarter-circle fillet → straight side wall →
/// quarter-circle fillet → flat bottom cap.
fn generate_rounded_cylinder(
    radial_segments: u32,
    fillet_segments: u32,
    wall_segments: u32,
    fillet_radius: f32,
) -> (Vec<Vertex>, Vec<u32>) {
    let radius = 1.0_f32;
    let half_height = 1.0_f32;
    let fillet = fillet_radius.clamp(0.0, radius.min(half_height));
    let flat_radius = radius - fillet;
    let flat_half_height = half_height - fillet;

    // No flat caps: the mesh is open-ended, just the rounded rim curving
    // into the side wall and back out the other end — a flat cap built from
    // a fan of triangles meeting at the center point is exactly what was
    // causing the mapping problems (the center point is one 3D position
    // shared by every vertex around that fan, each sampling a different
    // part of the height/normal textures, which either tore the fan apart
    // under displacement or, pinned to a single UV to stop that, left the
    // whole cap without real texture detail). Simplest fix is to not have
    // that fan at all.
    let mut profile = Vec::new();
    profile.push(ProfilePoint {
        r: flat_radius,
        y: half_height,
        nr: 0.0,
        ny: 1.0,
    });
    for i in 1..=fillet_segments {
        let t = i as f32 / fillet_segments as f32 * std::f32::consts::FRAC_PI_2;
        let (nr, ny) = (t.sin(), t.cos());
        profile.push(ProfilePoint {
            r: flat_radius + fillet * nr,
            y: flat_half_height + fillet * ny,
            nr,
            ny,
        });
    }
    // The straight side wall, same radius/normal throughout — subdivided
    // along its height so vertex displacement has enough geometry to
    // actually trace the height map's detail. A single edge from top to
    // bottom can only bend the whole wall as one flat ruled strip between
    // two displaced rings, which isn't fine enough to show anything but
    // the broadest height variation.
    for i in 1..wall_segments {
        let t = i as f32 / wall_segments as f32;
        profile.push(ProfilePoint {
            r: radius,
            y: flat_half_height - t * (2.0 * flat_half_height),
            nr: 1.0,
            ny: 0.0,
        });
    }
    profile.push(ProfilePoint {
        r: radius,
        y: -flat_half_height,
        nr: 1.0,
        ny: 0.0,
    });
    for i in 1..=fillet_segments {
        let t = i as f32 / fillet_segments as f32 * std::f32::consts::FRAC_PI_2;
        let (nr, ny) = (t.cos(), -t.sin());
        profile.push(ProfilePoint {
            r: flat_radius + fillet * nr,
            y: -flat_half_height + fillet * ny,
            nr,
            ny,
        });
    }
    // The lower fillet loop above already ends exactly on the bottom rim —
    // no closing cap here either, same reasoning as the top.

    revolve(&profile, radial_segments)
}

fn generate_plane(segments: u32) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    for j in 0..=segments {
        let v = j as f32 / segments as f32;
        let z = v * 2.0 - 1.0;
        for i in 0..=segments {
            let u = i as f32 / segments as f32;
            let x = u * 2.0 - 1.0;
            vertices.push(Vertex {
                position: [x, 0.0, z],
                normal: [0.0, 1.0, 0.0],
                uv: [u, 1.0 - v],
                tangent: [0.0; 3],
            });
        }
    }

    let stride = segments + 1;
    let mut indices = Vec::new();
    for j in 0..segments {
        for i in 0..segments {
            let a = j * stride + i;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    compute_tangents(&mut vertices, &indices);
    (vertices, indices)
}

/// Computes a per-vertex tangent (the direction UV's U axis actually runs
/// across the surface) from each triangle's position/UV gradient — the
/// standard technique (Lengyel), averaged across every triangle a vertex
/// touches and re-orthogonalized against its normal. This is what lets the
/// shader build a real tangent-space basis for normal mapping instead of
/// faking one from the normal alone (which only happened to look right on
/// the Plane, where the normal is constant and the UV layout lines up with
/// it by coincidence — see `viewport3d.wgsl`).
///
/// Deliberately driven only by `position`/`normal`/`uv`, not by any
/// knowledge of which primitive generated them, so it works unchanged for
/// every mesh here *and* for a future user-imported mesh (CLAUDE.md's
/// mesh-import punch list item) — that code will just need to call this too.
fn compute_tangents(vertices: &mut [Vertex], indices: &[u32]) {
    let mut accumulated = vec![Vec3::ZERO; vertices.len()];

    for tri in indices.as_chunks::<3>().0 {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = Vec3::from(vertices[i0].position);
        let p1 = Vec3::from(vertices[i1].position);
        let p2 = Vec3::from(vertices[i2].position);
        let uv0 = vertices[i0].uv;
        let uv1 = vertices[i1].uv;
        let uv2 = vertices[i2].uv;

        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let duv1 = [uv1[0] - uv0[0], uv1[1] - uv0[1]];
        let duv2 = [uv2[0] - uv0[0], uv2[1] - uv0[1]];

        let denom = duv1[0] * duv2[1] - duv2[0] * duv1[1];
        if denom.abs() < 1e-12 {
            // Degenerate UV triangle (e.g. zero UV area) — contributes no
            // usable tangent direction; its vertices still get one from
            // whichever other triangles they're part of.
            continue;
        }
        let r = 1.0 / denom;
        let tangent = (edge1 * duv2[1] - edge2 * duv1[1]) * r;
        if tangent.is_finite() {
            accumulated[i0] += tangent;
            accumulated[i1] += tangent;
            accumulated[i2] += tangent;
        }
    }

    for (vertex, accumulated_tangent) in vertices.iter_mut().zip(accumulated) {
        let normal = Vec3::from(vertex.normal);
        let tangent =
            (accumulated_tangent - normal * normal.dot(accumulated_tangent)).normalize_or_zero();
        let tangent = if tangent != Vec3::ZERO {
            tangent
        } else {
            // No triangle contributed a usable tangent (e.g. a sphere's pole,
            // where every triangle degenerates to a point) — any direction
            // perpendicular to the normal is as good as another there.
            let arbitrary = if normal.x.abs() < 0.9 {
                Vec3::X
            } else {
                Vec3::Y
            };
            (arbitrary - normal * normal.dot(arbitrary)).normalize()
        };
        vertex.tangent = tangent.to_array();
    }
}

#[cfg(test)]
mod tests {
    use super::{
        device_supports_tess, height_evaluation_hash, tessellated_mesh, uniform_bytes, CurrentMesh,
        MeshKind, RenderSettings, ShadingMode, Viewport3d, TESSELLATION_MAX,
        TESS_STORAGE_BUFFERS_NEEDED, UNIFORM_MESH_SEGMENTS_MAX,
    };

    /// Pins the exact threshold behind the real crash this guards against
    /// (confirmed on an older Intel Haswell iGPU reporting 0 VERTEX-stage
    /// storage buffers under both Vulkan and GL): `Viewport3d::new` must
    /// skip building the tess bind group layout on a device reporting fewer
    /// than `TESS_STORAGE_BUFFERS_NEEDED`, and build it normally once the
    /// device reports enough.
    #[test]
    fn device_supports_tess_matches_the_needed_storage_buffer_count() {
        let limits_with = |max_storage_buffers_per_shader_stage| wgpu::Limits {
            max_storage_buffers_per_shader_stage,
            ..wgpu::Limits::default()
        };
        assert!(!device_supports_tess(&limits_with(0)));
        assert!(!device_supports_tess(&limits_with(
            TESS_STORAGE_BUFFERS_NEEDED - 1
        )));
        assert!(device_supports_tess(&limits_with(
            TESS_STORAGE_BUFFERS_NEEDED
        )));
        assert!(device_supports_tess(&limits_with(
            TESS_STORAGE_BUFFERS_NEEDED + 5
        )));
    }

    /// `TESSELLATION_MAX` (4096) is deliberately independent of the graph's
    /// output Resolution now, but every *uniform*-grid mesh kind (no Height
    /// connected, or any mesh kind other than Cube/Rounded Cube) has no
    /// other protection against building an enormous CPU vertex/index
    /// buffer directly from that number — confirms `tessellated_mesh`
    /// actually clamps to `UNIFORM_MESH_SEGMENTS_MAX` regardless, for every
    /// mesh kind, rather than silently trusting the slider's own range.
    #[test]
    fn uniform_mesh_clamps_segment_count_even_at_the_sliders_own_max() {
        for kind in MeshKind::ALL {
            let (vertices, _) = tessellated_mesh(kind, TESSELLATION_MAX);
            let max_plausible_vertices = 6 * (UNIFORM_MESH_SEGMENTS_MAX as usize + 1).pow(2) * 4;
            assert!(
                vertices.len() <= max_plausible_vertices,
                "{kind:?}: {} vertices at Tessellation={TESSELLATION_MAX} — UNIFORM_MESH_SEGMENTS_MAX \
                 doesn't seem to be clamping this mesh kind",
                vertices.len()
            );
        }
    }

    /// Renders one frame of `viewport` (whatever `mesh_kind`/`height_scale`/
    /// camera it's already configured with) to an offscreen `width`×`height`
    /// `Rgba8Unorm` target and reads it back — the same GPU work
    /// `Viewport3d::render` does, minus the `egui_wgpu::RenderState`
    /// bits (registering the result as a displayable egui texture), which
    /// only matter for showing it in a window, not for verifying the
    /// shader's own math. Lets a regression test actually look at rendered
    /// pixels instead of guessing from a screenshot.
    fn render_offscreen(
        viewport: &mut Viewport3d,
        gpu: &mut mc_gpu::Context,
        pool: &mc_gpu::TexturePool,
        graph: &mc_core::Graph,
        resolution: u32,
    ) -> Vec<u8> {
        let color_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test color target"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test depth target"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let texture_bind_group = viewport.build_texture_bind_group(gpu, pool, graph);

        let desired_key = (
            viewport.mesh_kind,
            viewport.tessellation,
            height_evaluation_hash(graph),
        );
        if desired_key != viewport.current_mesh_key {
            viewport.current_mesh = viewport.build_and_upload_mesh(
                viewport.mesh_kind,
                viewport.tessellation,
                gpu,
                pool,
                graph,
            );
            viewport.current_mesh_key = desired_key;
        }
        let tess = match &viewport.current_mesh {
            CurrentMesh::Uniform { .. } => (0.0, 0.0, 0.0),
            CurrentMesh::Tessellated {
                n_global,
                tex_size,
                mip_count,
                ..
            } => (*n_global as f32, *tex_size as f32, *mip_count as f32),
        };

        let aspect = 1.0;
        let eye = viewport.eye_position();
        let view = glam::camera::rh::view::look_at_mat4(eye, viewport.target, glam::Vec3::Y);
        let proj =
            glam::camera::rh::proj::directx::perspective(60f32.to_radians(), aspect, 0.1, 100.0);
        let light_dir = glam::Vec3::new(-0.4, -1.0, -0.3).normalize();
        let bytes = uniform_bytes(
            proj * view,
            eye,
            light_dir,
            viewport.environment_ambient_color,
            RenderSettings {
                height_scale: viewport.height_scale,
                shading_mode: ShadingMode::Lit,
                exposure: viewport.exposure,
                environment_intensity: viewport.environment_intensity,
                environment_blur: viewport.environment_blur,
                tess,
            },
        );
        gpu.queue.write_buffer(&viewport.uniform_buffer, 0, &bytes);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("test encoder"),
            });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("test pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            viewport.record_draw(&mut pass, &texture_bind_group, false);
        }
        gpu.queue.submit(Some(encoder.finish()));

        gpu.read_rgba8(&color_texture, resolution)
    }

    /// Regression test for a crash found by manual testing: wiring Gabor
    /// Noise's `raw` output (deliberately `Rgba32Float`, not the usual
    /// `Rgba16Float` every other node output uses — see `gabor_noise.rs`'s
    /// own doc comment on why) into Output: Height and looking at the 3D
    /// viewport crashed with a wgpu validation error ("Texture binding 5
    /// expects sample type Float { filterable: true }, but was given a
    /// view with format Rgba32Float (sample type Float { filterable:
    /// false })"). `Rgba32Float` can't be sampled with filtering without
    /// an opt-in device feature this app doesn't request — fixed by having
    /// `build_texture_bind_group`'s own `channel_view` fall back to the
    /// placeholder texture for any channel whose resolved format isn't
    /// filterable, the same as if nothing were connected, rather than
    /// trying to bind it anyway.
    #[test]
    fn height_wired_to_a_non_filterable_format_falls_back_instead_of_crashing() {
        let mut graph = mc_core::Graph::new();
        let gabor = graph.add_node(mc_core::Node::new("gabor_noise"));
        let output = graph.add_node(mc_core::Node::new("output_height"));
        graph.connect(gabor, "raw", output, "input").unwrap();
        graph.settings.resolution = 64;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        mc_nodes::registry::evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let viewport = Viewport3d::new(&gpu);
        // Would panic inside wgpu validation before the fix.
        let _bind_group = viewport.build_texture_bind_group(&mut gpu, &pool, &graph);
    }

    /// Confirms Height actually displaces real geometry again (reverted
    /// from a per-pixel parallax shift — see `viewport3d.wgsl`'s module
    /// doc comment for why: parallax never moves the mesh's own
    /// silhouette, only which texel is shown where, which read as the
    /// texture visibly swimming/crawling as the camera orbited — a real
    /// per-vertex offset doesn't have that problem, because it's actual
    /// geometry). A Plane with Height = 1.0 across its far half (raised
    /// toward the camera, which sits above it) should cover visibly more
    /// of the frame at a large Height Scale than at zero — a change in
    /// silhouette area is only possible with real displacement; a pure
    /// shading effect like parallax can never change it at all.
    #[test]
    fn height_scale_visibly_changes_the_meshs_own_silhouette() {
        let mut graph = mc_core::Graph::new();
        let base_color_source = graph.add_node(mc_core::Node::new("uniform_color"));
        let base_color_output = graph.add_node(mc_core::Node::new("output_base_color"));
        graph
            .connect(base_color_source, "output", base_color_output, "input")
            .unwrap();
        let height_source = graph.add_node(mc_core::Node::new("uniform_color"));
        let height_output = graph.add_node(mc_core::Node::new("output_height"));
        graph
            .connect(height_source, "output", height_output, "input")
            .unwrap();
        let resolution = 128;
        graph.settings.resolution = resolution;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        mc_nodes::registry::evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        // A bright, solid Base Color against the viewport's own dark clear
        // color, and a Height field that's fully raised (1.0) across the
        // texture's far (v < 0.5) half — `generate_plane`'s own UV
        // convention maps that to the edge of the plane farthest from the
        // default camera, which sits above and in front of it, so raising
        // that edge moves it up and toward the camera.
        let base_color_key =
            mc_nodes::registry::resolve_texture_key(&graph, base_color_output).unwrap();
        let bright = vec![[1.0f32, 1.0, 1.0, 1.0]; (resolution * resolution) as usize];
        gpu.write_rgba16f(&mut pool, &base_color_key, resolution, &bright);

        let height_key = mc_nodes::registry::resolve_texture_key(&graph, height_output).unwrap();
        let half_raised: Vec<[f32; 4]> = (0..resolution * resolution)
            .map(|i| {
                let y = i / resolution;
                if y < resolution / 2 {
                    [1.0, 1.0, 1.0, 1.0]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                }
            })
            .collect();
        gpu.write_rgba16f(&mut pool, &height_key, resolution, &half_raised);

        let mut viewport = Viewport3d::new(&gpu);
        viewport.mesh_kind = super::MeshKind::Plane;
        viewport.shading_mode = ShadingMode::UnlitBaseColor;

        let count_mesh_pixels = |pixels: &[u8]| -> usize {
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|p| p[0] > 128)
                .count()
        };

        viewport.height_scale = 0.0;
        let flat = render_offscreen(&mut viewport, &mut gpu, &pool, &graph, resolution);
        let flat_count = count_mesh_pixels(&flat);

        viewport.height_scale = 0.8;
        let raised = render_offscreen(&mut viewport, &mut gpu, &pool, &graph, resolution);
        let raised_count = count_mesh_pixels(&raised);

        assert!(
            flat_count > 0,
            "the flat plane should cover a visible part of the frame at all"
        );
        assert_ne!(
            flat_count, raised_count,
            "raising half the plane by a large Height Scale should visibly change how much of \
             the frame its silhouette covers — if it doesn't, Height isn't actually displacing \
             geometry"
        );
    }

    /// End-to-end exercise of the GPU vertex-pulling tessellation path
    /// (`build_and_upload_mesh`'s `CurrentMesh::Tessellated` branch,
    /// `vs_main_tess`) through the real `Viewport3d` API — not just
    /// `tess_gpu`'s own lower-level harness — for Rounded Cube with a real
    /// Height node connected. Confirms it renders without error at a
    /// Tessellation value well past `BASE_FILLET_SEGMENTS`'s own base
    /// triangle count (so this is genuinely exercising per-base-triangle
    /// subdivision, not accidentally staying at N=1), covers a real part
    /// of the frame, and that Height Scale visibly changes the mesh's own
    /// silhouette here too (the same displacement proof
    /// `height_scale_visibly_changes_the_meshs_own_silhouette` already
    /// established for the plain-vertex-buffer Plane path).
    #[test]
    fn tessellated_rounded_cube_renders_and_displaces() {
        let mut graph = mc_core::Graph::new();
        let base_color_source = graph.add_node(mc_core::Node::new("uniform_color"));
        let base_color_output = graph.add_node(mc_core::Node::new("output_base_color"));
        graph
            .connect(base_color_source, "output", base_color_output, "input")
            .unwrap();
        let height_source = graph.add_node(mc_core::Node::new("uniform_color"));
        let height_output = graph.add_node(mc_core::Node::new("output_height"));
        graph
            .connect(height_source, "output", height_output, "input")
            .unwrap();
        let resolution = 64;
        graph.settings.resolution = resolution;

        let mut gpu = mc_gpu::Context::new_headless().expect("headless GPU context");
        let mut pool = mc_gpu::TexturePool::default();
        mc_nodes::registry::evaluate(&mut gpu, &mut pool, &mut graph).expect("graph evaluates");

        let base_color_key =
            mc_nodes::registry::resolve_texture_key(&graph, base_color_output).unwrap();
        let bright = vec![[1.0f32, 1.0, 1.0, 1.0]; (resolution * resolution) as usize];
        gpu.write_rgba16f(&mut pool, &base_color_key, resolution, &bright);

        // A checkerboard, not a flat value — flat Height produces zero
        // contrast (the earlier `adaptive_mesh` bug class relied on real
        // contrast to show up at all) and this exercises many base
        // triangles sampling genuinely different heights, not just one.
        let height_key = mc_nodes::registry::resolve_texture_key(&graph, height_output).unwrap();
        let checker: Vec<[f32; 4]> = (0..resolution * resolution)
            .map(|i| {
                let (x, y) = (i % resolution, i / resolution);
                if (x / 8 + y / 8) % 2 == 0 {
                    [1.0, 1.0, 1.0, 1.0]
                } else {
                    [0.0, 0.0, 0.0, 1.0]
                }
            })
            .collect();
        gpu.write_rgba16f(&mut pool, &height_key, resolution, &checker);

        let mut viewport = Viewport3d::new(&gpu);
        viewport.mesh_kind = super::MeshKind::RoundedCube;
        viewport.shading_mode = ShadingMode::UnlitBaseColor;
        // Comfortably past `BASE_FILLET_SEGMENTS` (12) and within
        // `MAX_TESS_TRIANGLES`'s budget for a 1,728-base-triangle mesh.
        viewport.tessellation = 32;

        let count_mesh_pixels = |pixels: &[u8]| -> usize {
            pixels
                .as_chunks::<4>()
                .0
                .iter()
                .filter(|p| p[0] > 128)
                .count()
        };

        viewport.height_scale = 0.0;
        let flat = render_offscreen(&mut viewport, &mut gpu, &pool, &graph, resolution);
        let flat_count = count_mesh_pixels(&flat);
        assert!(
            flat_count > 0,
            "the rounded cube should cover a visible part of the frame at all"
        );
        assert!(
            matches!(viewport.current_mesh, CurrentMesh::Tessellated { .. }),
            "Rounded Cube with a Height node connected should take the tessellated path, not the \
             plain uniform-grid one"
        );

        viewport.height_scale = 0.3;
        let raised = render_offscreen(&mut viewport, &mut gpu, &pool, &graph, resolution);
        let raised_count = count_mesh_pixels(&raised);
        assert_ne!(
            flat_count, raised_count,
            "raising Height Scale on a checkerboard height map should visibly change the \
             tessellated mesh's own silhouette — if it doesn't, vs_main_tess isn't actually \
             displacing geometry"
        );
    }
}
