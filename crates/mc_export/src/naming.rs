// SPDX-License-Identifier: MIT OR Apache-2.0
//! Export naming presets (CLAUDE.md §8): fixed for v0.1 rather than the
//! fully editable templates the spec eventually wants, but using the same
//! `{name}` substitution so making them editable later is a small change.

/// Which target application's conventions to export for. "Painter" here
/// names a third-party target app a user might drop these textures into —
/// same nominative use as "Blender"/"Unreal Engine" above, not an internal
/// mode name of this app's own (see INDEPENDENCE.md's actual objection,
/// which is to naming *this app's own* UI modes "Designer"/"Painter" to
/// mirror Adobe's product split — a different concern from this).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Blender,
    Unreal,
    Painter,
}

impl Preset {
    pub const NAMES: [&'static str; 3] = ["Blender", "Unreal Engine", "Painter"];

    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Blender" => Some(Self::Blender),
            "Unreal Engine" => Some(Self::Unreal),
            "Painter" => Some(Self::Painter),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Preset::Blender => "Blender",
            Preset::Unreal => "Unreal Engine",
            Preset::Painter => "Painter",
        }
    }

    /// The normal map convention this preset expects. Fixed for
    /// Blender/Unreal; Painter has no fixed convention of its own (its
    /// users pick OpenGL or DirectX per-project), so this is just its
    /// sensible default — the actual choice for a given export is a
    /// runtime option, not something this method alone determines.
    pub fn normal_format(self) -> mc_core::NormalFormat {
        match self {
            Preset::Blender => mc_core::NormalFormat::OpenGl,
            Preset::Unreal => mc_core::NormalFormat::DirectX,
            Preset::Painter => mc_core::NormalFormat::OpenGl,
        }
    }

    /// `ext` ("png"/"tga") is only consulted for Unreal — Blender and
    /// Painter always write `.png` (Painter's own spec: BaseColor is always
    /// an 8-bit sRGB PNG, no format choice).
    pub fn base_color_file_name(self, graph_name: &str, ext: &str) -> String {
        match self {
            Preset::Blender => format!("{graph_name}_BaseColor.png"),
            Preset::Unreal => format!("T_{graph_name}_BC.{ext}"),
            Preset::Painter => format!("{graph_name}_BaseColor.png"),
        }
    }

    /// `normal_format` is only consulted for Painter, whose convention is a
    /// runtime choice rather than fixed per preset (see
    /// [`Self::normal_format`]) — Blender/Unreal ignore it and use their own
    /// fixed convention instead, same as they already ignore `ext` in some
    /// arms. Painter embeds which convention was used directly in the file
    /// name (`_Normal_OpenGL`/`_Normal_DirectX`) so it matches whatever the
    /// user picked when creating their Painter project.
    pub fn normal_file_name(
        self,
        graph_name: &str,
        ext: &str,
        normal_format: mc_core::NormalFormat,
    ) -> String {
        match self {
            Preset::Blender => format!("{graph_name}_Normal_GL.png"),
            Preset::Unreal => format!("T_{graph_name}_N.{ext}"),
            Preset::Painter => {
                let convention = match normal_format {
                    mc_core::NormalFormat::OpenGl => "OpenGL",
                    mc_core::NormalFormat::DirectX => "DirectX",
                };
                format!("{graph_name}_Normal_{convention}.png")
            }
        }
    }

    /// `ext` is only consulted for Painter, whose Height can be a 16-bit PNG
    /// (default) or a 32-bit EXR (`ext` = "exr") — Blender/Unreal ignore it
    /// and always write `.png` (TGA has no clean 16-bit single-channel
    /// equivalent, so neither offers a format choice here).
    pub fn height_file_name(self, graph_name: &str, ext: &str) -> String {
        match self {
            Preset::Blender => format!("{graph_name}_Height.png"),
            Preset::Unreal => format!("T_{graph_name}_H.png"),
            Preset::Painter => format!("{graph_name}_Height.{ext}"),
        }
    }

    /// `None` for Unreal — it packs these into one ORM texture instead, via
    /// [`Self::orm_file_name`].
    pub fn roughness_file_name(self, graph_name: &str) -> Option<String> {
        match self {
            Preset::Blender => Some(format!("{graph_name}_Roughness.png")),
            Preset::Unreal => None,
            Preset::Painter => Some(format!("{graph_name}_Roughness.png")),
        }
    }

    pub fn metallic_file_name(self, graph_name: &str) -> Option<String> {
        match self {
            Preset::Blender => Some(format!("{graph_name}_Metallic.png")),
            Preset::Unreal => None,
            Preset::Painter => Some(format!("{graph_name}_Metallic.png")),
        }
    }

    /// Painter spells this channel out in full (`AmbientOcclusion`) rather
    /// than Blender's abbreviated `AO`, matching Painter's own channel
    /// vocabulary from the spec.
    pub fn ambient_occlusion_file_name(self, graph_name: &str) -> Option<String> {
        match self {
            Preset::Blender => Some(format!("{graph_name}_AO.png")),
            Preset::Unreal => None,
            Preset::Painter => Some(format!("{graph_name}_AmbientOcclusion.png")),
        }
    }

    /// `None` for Blender and Painter — Painter never packs channels at all
    /// (its whole point is one plain image per channel, for dropping
    /// straight into a fill layer's own channel slots).
    pub fn orm_file_name(self, graph_name: &str, ext: &str) -> Option<String> {
        match self {
            Preset::Blender => None,
            Preset::Unreal => Some(format!("T_{graph_name}_ORM.{ext}")),
            Preset::Painter => None,
        }
    }

    /// Painter has no "always-on flat alpha" concept like Blender's — its
    /// orchestration uses [`Self::opacity_mask_file_name`] directly instead
    /// (Opacity is optional for Painter, like every other channel), so this
    /// arm is never actually called in practice; it delegates there anyway
    /// so the two names can't drift apart, and to keep this match
    /// exhaustive.
    pub fn alpha_file_name(self, graph_name: &str) -> String {
        match self {
            Preset::Blender => format!("{graph_name}_Alpha.png"),
            Preset::Unreal => format!("T_{graph_name}_A.png"),
            Preset::Painter => self
                .opacity_mask_file_name(graph_name, "png")
                .expect("Painter always has an opacity file name"),
        }
    }

    /// Emissive's own filename — every preset can produce this file (unlike
    /// most of the methods above, which are genuinely preset-specific);
    /// whether it's actually *written* is a per-export toggle decided by
    /// the caller, not something this method knows about. `ext` is ignored
    /// by Painter, which always writes an 8-bit sRGB PNG for Emissive.
    pub fn emissive_file_name(self, graph_name: &str, ext: &str) -> String {
        match self {
            Preset::Blender => format!("{graph_name}_Emissive.png"),
            Preset::Unreal => format!("T_{graph_name}_E.{ext}"),
            Preset::Painter => format!("{graph_name}_Emissive.png"),
        }
    }

    /// `None` for Blender, which has no "separate opacity mask" concept —
    /// only its own always-on [`Self::alpha_file_name`]. `Some` for Unreal,
    /// used only when opacity is exported as its own file rather than
    /// folded into Base Color's alpha channel. Always `Some` for Painter,
    /// which never folds channels together and always writes Opacity (when
    /// present) as its own file; `ext` is ignored, always PNG.
    pub fn opacity_mask_file_name(self, graph_name: &str, ext: &str) -> Option<String> {
        match self {
            Preset::Blender => None,
            Preset::Unreal => Some(format!("T_{graph_name}_M.{ext}")),
            Preset::Painter => Some(format!("{graph_name}_Opacity.png")),
        }
    }
}

/// Strips everything except ASCII letters/digits/`_`/`-` from a material
/// name, so it's always safe to drop directly into a file name — used by
/// the Painter export preset, where the material name is user-editable
/// rather than fixed to the graph's own file name. Falls back to
/// `"Material"` if nothing survives (e.g. an all-emoji or all-punctuation
/// input).
pub fn sanitize_material_name(input: &str) -> String {
    let cleaned: String = input
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    if cleaned.is_empty() {
        "Material".to_string()
    } else {
        cleaned
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blender_names_match_spec() {
        assert_eq!(
            Preset::Blender.base_color_file_name("bricks", "png"),
            "bricks_BaseColor.png"
        );
        assert_eq!(
            Preset::Blender.normal_file_name("bricks", "png", mc_core::NormalFormat::OpenGl),
            "bricks_Normal_GL.png"
        );
        assert_eq!(
            Preset::Blender.roughness_file_name("bricks"),
            Some("bricks_Roughness.png".to_string())
        );
        assert_eq!(
            Preset::Blender.metallic_file_name("bricks"),
            Some("bricks_Metallic.png".to_string())
        );
        assert_eq!(
            Preset::Blender.height_file_name("bricks", "png"),
            "bricks_Height.png"
        );
        assert_eq!(
            Preset::Blender.ambient_occlusion_file_name("bricks"),
            Some("bricks_AO.png".to_string())
        );
        assert_eq!(Preset::Blender.orm_file_name("bricks", "png"), None);
        assert_eq!(
            Preset::Blender.alpha_file_name("bricks"),
            "bricks_Alpha.png"
        );
        assert_eq!(
            Preset::Blender.emissive_file_name("bricks", "png"),
            "bricks_Emissive.png"
        );
        assert_eq!(
            Preset::Blender.opacity_mask_file_name("bricks", "png"),
            None
        );
        assert_eq!(
            Preset::Blender.normal_format(),
            mc_core::NormalFormat::OpenGl
        );
        // `ext` is ignored for Blender, which always writes PNG.
        assert_eq!(
            Preset::Blender.base_color_file_name("bricks", "tga"),
            "bricks_BaseColor.png"
        );
    }

    #[test]
    fn unreal_names_match_spec() {
        assert_eq!(
            Preset::Unreal.base_color_file_name("bricks", "png"),
            "T_bricks_BC.png"
        );
        assert_eq!(
            Preset::Unreal.normal_file_name("bricks", "png", mc_core::NormalFormat::DirectX),
            "T_bricks_N.png"
        );
        assert_eq!(
            Preset::Unreal.height_file_name("bricks", "png"),
            "T_bricks_H.png"
        );
        assert_eq!(
            Preset::Unreal.orm_file_name("bricks", "png"),
            Some("T_bricks_ORM.png".to_string())
        );
        assert_eq!(Preset::Unreal.roughness_file_name("bricks"), None);
        assert_eq!(Preset::Unreal.metallic_file_name("bricks"), None);
        assert_eq!(Preset::Unreal.ambient_occlusion_file_name("bricks"), None);
        assert_eq!(Preset::Unreal.alpha_file_name("bricks"), "T_bricks_A.png");
        assert_eq!(
            Preset::Unreal.emissive_file_name("bricks", "png"),
            "T_bricks_E.png"
        );
        assert_eq!(
            Preset::Unreal.opacity_mask_file_name("bricks", "png"),
            Some("T_bricks_M.png".to_string())
        );
        assert_eq!(
            Preset::Unreal.normal_format(),
            mc_core::NormalFormat::DirectX
        );
        assert_eq!(Preset::Unreal.name(), "Unreal Engine");
    }

    /// TGA is Unreal-only — every `ext`-taking method actually honors it.
    #[test]
    fn unreal_names_honor_tga_extension() {
        assert_eq!(
            Preset::Unreal.base_color_file_name("bricks", "tga"),
            "T_bricks_BC.tga"
        );
        assert_eq!(
            Preset::Unreal.normal_file_name("bricks", "tga", mc_core::NormalFormat::DirectX),
            "T_bricks_N.tga"
        );
        assert_eq!(
            Preset::Unreal.orm_file_name("bricks", "tga"),
            Some("T_bricks_ORM.tga".to_string())
        );
        assert_eq!(
            Preset::Unreal.emissive_file_name("bricks", "tga"),
            "T_bricks_E.tga"
        );
        assert_eq!(
            Preset::Unreal.opacity_mask_file_name("bricks", "tga"),
            Some("T_bricks_M.tga".to_string())
        );
        // Height stays PNG regardless — Unreal ignores `ext` entirely.
        assert_eq!(
            Preset::Unreal.height_file_name("bricks", "png"),
            "T_bricks_H.png"
        );
    }

    #[test]
    fn preset_name_round_trips() {
        for name in Preset::NAMES {
            assert!(Preset::from_name(name).is_some(), "{name:?} should parse");
        }
    }

    #[test]
    fn sanitize_material_name_strips_spaces_and_special_chars() {
        assert_eq!(sanitize_material_name("My Brick Wall!"), "MyBrickWall");
        assert_eq!(sanitize_material_name("rust_metal-02"), "rust_metal-02");
        assert_eq!(sanitize_material_name("Café Tile #3"), "CafTile3");
    }

    #[test]
    fn sanitize_material_name_falls_back_when_nothing_survives() {
        assert_eq!(sanitize_material_name("!!!"), "Material");
        assert_eq!(sanitize_material_name(""), "Material");
        assert_eq!(sanitize_material_name("已朽木材"), "Material");
    }

    #[test]
    fn painter_names_match_spec() {
        assert_eq!(
            Preset::Painter.base_color_file_name("bricks", "png"),
            "bricks_BaseColor.png"
        );
        assert_eq!(
            Preset::Painter.normal_file_name("bricks", "png", mc_core::NormalFormat::OpenGl),
            "bricks_Normal_OpenGL.png"
        );
        assert_eq!(
            Preset::Painter.normal_file_name("bricks", "png", mc_core::NormalFormat::DirectX),
            "bricks_Normal_DirectX.png"
        );
        assert_eq!(
            Preset::Painter.height_file_name("bricks", "png"),
            "bricks_Height.png"
        );
        assert_eq!(
            Preset::Painter.height_file_name("bricks", "exr"),
            "bricks_Height.exr"
        );
        assert_eq!(
            Preset::Painter.roughness_file_name("bricks"),
            Some("bricks_Roughness.png".to_string())
        );
        assert_eq!(
            Preset::Painter.metallic_file_name("bricks"),
            Some("bricks_Metallic.png".to_string())
        );
        assert_eq!(
            Preset::Painter.ambient_occlusion_file_name("bricks"),
            Some("bricks_AmbientOcclusion.png".to_string())
        );
        assert_eq!(Preset::Painter.orm_file_name("bricks", "png"), None);
        assert_eq!(
            Preset::Painter.emissive_file_name("bricks", "png"),
            "bricks_Emissive.png"
        );
        assert_eq!(
            Preset::Painter.opacity_mask_file_name("bricks", "png"),
            Some("bricks_Opacity.png".to_string())
        );
        assert_eq!(
            Preset::Painter.normal_format(),
            mc_core::NormalFormat::OpenGl
        );
        assert_eq!(Preset::Painter.name(), "Painter");
        // `ext` is ignored everywhere except Height — Painter always
        // writes PNG for every other channel.
        assert_eq!(
            Preset::Painter.base_color_file_name("bricks", "exr"),
            "bricks_BaseColor.png"
        );
    }
}
