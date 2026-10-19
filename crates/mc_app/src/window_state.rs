// SPDX-License-Identifier: MIT OR Apache-2.0
//! Remembers the native window's own size/position/maximized state across
//! launches — a friend's hands-on testing report: on a 2560x1600 display,
//! the window still opened at some small default size with the top-right
//! toolbar (the Examples button) cut off until manually resized, despite
//! `main()`'s own `ViewportBuilder::with_maximized(true)` — `eframe`'s
//! built-in equivalent (`NativeOptions::persist_window`) needs its
//! `"persistence"` Cargo feature, which pulls in `ron` and `serde`'s own
//! derive machinery as new dependencies; this is 5 numbers, so a plain
//! comma-separated line avoids that entirely rather than asking to add a
//! dependency for something this small.

use std::path::PathBuf;

/// Same per-user app data directory `autosave.rs` uses, different file —
/// genuinely unrelated data (this is UI chrome, not graph content), so its
/// own file rather than folding into the recovery snapshot.
fn path() -> PathBuf {
    let base = dirs::data_dir().unwrap_or_else(std::env::temp_dir);
    base.join("MaterialCreator").join("window_state.txt")
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowState {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub maximized: bool,
}

impl WindowState {
    /// `None` on first launch, a corrupted file, or any read/parse
    /// failure — every one of those just falls back to `main()`'s own
    /// existing maximized-on-first-launch default, never a hard error.
    pub fn load() -> Option<Self> {
        let text = std::fs::read_to_string(path()).ok()?;
        let mut fields = text.trim().split(',');
        let x: f32 = fields.next()?.parse().ok()?;
        let y: f32 = fields.next()?.parse().ok()?;
        let width: f32 = fields.next()?.parse().ok()?;
        let height: f32 = fields.next()?.parse().ok()?;
        let maximized: bool = fields.next()?.parse().ok()?;
        Some(Self {
            x,
            y,
            width,
            height,
            maximized,
        })
    }

    /// Best-effort, same reasoning as `Autosave::maybe_save` — a failed
    /// write here (no permission, full disk) just means next launch falls
    /// back to the default, not worth interrupting shutdown over.
    pub fn save(&self) {
        let path = path();
        if let Some(parent) = path.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                return;
            }
        }
        let text = format!(
            "{},{},{},{},{}",
            self.x, self.y, self.width, self.height, self.maximized
        );
        let _ = std::fs::write(path, text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test, not several — every case here touches the same fixed
    // `path()` file, same reasoning as `autosave.rs`'s own single-test
    // comment: separate `#[test]` fns would race under cargo test's
    // default parallel runner.
    #[test]
    fn save_load_and_corrupted_file_handling() {
        let state = WindowState {
            x: 12.5,
            y: -3.0,
            width: 1920.0,
            height: 1080.0,
            maximized: true,
        };
        state.save();
        assert_eq!(WindowState::load(), Some(state));

        std::fs::write(path(), "not,a,valid,window,state,at,all").expect("test write");
        assert_eq!(
            WindowState::load(),
            None,
            "a malformed file should parse as None, not panic"
        );

        let _ = std::fs::remove_file(path());
        assert_eq!(
            WindowState::load(),
            None,
            "a missing file should parse as None"
        );
    }
}
