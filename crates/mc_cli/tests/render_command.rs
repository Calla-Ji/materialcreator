// SPDX-License-Identifier: MIT OR Apache-2.0
//! Exercises `mc_cli render` (CLAUDE.md §8) end to end as a real subprocess
//! — the same black-box level the user actually runs it at — rather than
//! calling internal functions directly, since `mc_cli` is a binary crate
//! with no `lib` target for an integration test to link against anyway.

use std::path::Path;
use std::process::Command;

fn run_render(args: &[&str]) {
    let status = Command::new(env!("CARGO_BIN_EXE_mc_cli"))
        .arg("render")
        .args(args)
        .status()
        .expect("failed to spawn mc_cli");
    assert!(status.success(), "mc_cli render {args:?} failed");
}

#[test]
fn variation_zero_matches_a_plain_render_exactly() {
    let dir = tempdir();
    run_render(&[
        "../../examples/graphs/bricks.mcgraph",
        "--preset",
        "blender",
        "--res",
        "64",
        "--out",
        dir.to_str().unwrap(),
    ]);
    run_render(&[
        "../../examples/graphs/bricks.mcgraph",
        "--preset",
        "blender",
        "--res",
        "64",
        "--out",
        dir.to_str().unwrap(),
        "--variations",
        "3",
    ]);

    let plain = std::fs::read(dir.join("bricks_BaseColor.png")).unwrap();
    let var0 = std::fs::read(dir.join("bricks_var0_BaseColor.png")).unwrap();
    assert_eq!(
        plain, var0,
        "variation 0 (no re-roll) should match a plain render exactly"
    );

    let var1 = std::fs::read(dir.join("bricks_var1_BaseColor.png")).unwrap();
    let var2 = std::fs::read(dir.join("bricks_var2_BaseColor.png")).unwrap();
    assert_ne!(var0, var1, "each variation should re-roll every Seed param");
    assert_ne!(var1, var2, "each variation should re-roll every Seed param");

    cleanup(&dir);
}

#[test]
fn variations_are_deterministic_across_runs() {
    let dir_a = tempdir();
    let dir_b = tempdir();
    for dir in [&dir_a, &dir_b] {
        run_render(&[
            "../../examples/graphs/bricks.mcgraph",
            "--preset",
            "blender",
            "--res",
            "64",
            "--out",
            dir.to_str().unwrap(),
            "--variations",
            "2",
        ]);
    }

    for file in ["bricks_var0_BaseColor.png", "bricks_var1_BaseColor.png"] {
        let a = std::fs::read(dir_a.join(file)).unwrap();
        let b = std::fs::read(dir_b.join(file)).unwrap();
        assert_eq!(
            a, b,
            "{file} should render identically across separate runs"
        );
    }

    cleanup(&dir_a);
    cleanup(&dir_b);
}

/// The Painter preset's own end-to-end acceptance test: export bricks at
/// 512px and check file names, bit depths, channel counts, and that no
/// placeholder files appear for outputs the graph doesn't have (bricks has
/// no Emissive or Alpha node).
#[test]
fn painter_preset_exports_one_file_per_channel_with_correct_formats() {
    let dir = tempdir();
    run_render(&[
        "../../examples/graphs/bricks.mcgraph",
        "--preset",
        "painter",
        "--res",
        "512",
        "--out",
        dir.to_str().unwrap(),
    ]);

    let material_dir = dir.join("bricks");
    for name in [
        "bricks_BaseColor.png",
        "bricks_Normal_OpenGL.png",
        "bricks_Height.png",
        "bricks_Roughness.png",
        "bricks_Metallic.png",
        "bricks_AmbientOcclusion.png",
        "README_Painter.txt",
    ] {
        assert!(material_dir.join(name).exists(), "missing {name}");
    }
    // bricks.mcgraph has no Emissive/Alpha output nodes -- Painter must
    // never write placeholder files for channels the graph doesn't have.
    assert!(!material_dir.join("bricks_Emissive.png").exists());
    assert!(!material_dir.join("bricks_Opacity.png").exists());

    let base_color = image::open(material_dir.join("bricks_BaseColor.png")).unwrap();
    assert_eq!(base_color.color(), image::ColorType::Rgba8);
    assert_eq!(base_color.width(), 512);

    let roughness = image::open(material_dir.join("bricks_Roughness.png")).unwrap();
    assert_eq!(
        roughness.color(),
        image::ColorType::L8,
        "Roughness should be single-channel 8-bit by default"
    );

    let height = image::open(material_dir.join("bricks_Height.png")).unwrap();
    assert_eq!(
        height.color(),
        image::ColorType::L16,
        "Height should be single-channel 16-bit by default"
    );

    cleanup(&dir);
}

fn tempdir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mc_cli_render_test_{}_{}",
        std::process::id(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

fn cleanup(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
}
