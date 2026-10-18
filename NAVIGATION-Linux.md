# MaterialCreator — Navigation & Shortcuts (Linux)

**MaterialCreator won't open at all** — a terminal error mentioning "cannot open shared
object file" or a Rust panic, or nothing visibly happens when you double-click it? You're
missing a runtime library most desktop Linux installs already have (this mainly comes up on a
minimal/server-style install). From a terminal, in this folder:

```sh
sudo apt install libxkbcommon0 libxkbcommon-x11-0 libx11-6 libxcursor1 libxi6 \
  libwayland-client0 libwayland-cursor0 mesa-vulkan-drivers libfontconfig1
./MaterialCreator
```

(Other distros: the same packages under Fedora/Arch's own naming — ask your distro's package
manager to search for `xkbcommon-x11`, `libxcursor`, `libXi`, `fontconfig`, and a
Vulkan driver package if `apt` isn't available.)

A quick reference for finding your way around the workspace. Nothing here needs to be
memorized up front — the canvas and panels are discoverable by just clicking around, but
these are the controls that aren't obvious from the UI alone.

## Node Graph (center panel)

| Action | Control |
|---|---|
| Pan the canvas | Left-drag on empty canvas background |
| Pan (mouse wheel/touchpad) | Two-finger scroll, or the mouse wheel |
| Zoom | Touchpad pinch (if your desktop environment supports it), or scroll while holding Ctrl |
| Box-select | Shift + left-drag on empty canvas |
| Select a node | Left-click it (replaces the current selection) |
| Preview a node in the 2D View/Parameters panel without changing the canvas selection | Click the light dot in its header |
| Add/remove a node from selection | Shift + left-click it |
| Select all | Ctrl+A |
| Move selected node(s) | Left-drag any selected node — the whole selection moves together |
| Delete selected node(s) | Delete or Backspace |
| Delete a single node | Right-click it → "Delete Node", or click the red dot in its header |
| Duplicate selected node(s) | Click the blue dot in any selected node's header, or right-click a node → "Duplicate Node" |
| Connect a wire | Left-drag from one pin to a compatible pin |
| Disconnect a wire | Left-click directly on the wire |
| Rewire (grab a connected end) | Ctrl + left-drag starting on an existing wire |
| Open the add-node menu | Right-click empty canvas |
| Add a node from a dropped wire | Drag a wire onto empty canvas and release — the add-node menu opens pre-filtered to compatible node kinds |
| Clear the current selection | Left-click empty canvas |

New nodes always land wherever you last clicked on the canvas — including ones added from the
Node Library panel, so click roughly where you want the node before adding it.

## Add-Node Menu

| Action | Control |
|---|---|
| Move the highlight | ↑ / ↓ |
| Add the highlighted node | Enter |
| Close without adding | Esc, or click outside the menu |
| Filter the list | Just start typing |

## 3D Viewport (top-right panel)

| Action | Control |
|---|---|
| Orbit the camera | Left-drag inside the viewport |
| Zoom / dolly | Scroll while hovering the viewport |

The toolbar on the viewport's left edge opens slide-out panels for Mesh, Material, and
Environment settings — click an icon to open it, click again to close it.

## File Menu

| Action | Shortcut |
|---|---|
| New | Ctrl+N |
| Open | Ctrl+O |
| Save | Ctrl+S |
| Save As | Ctrl+Shift+S |
| Export… | No shortcut — use the toolbar button |

**Undo/Redo don't have keyboard shortcuts yet** — use the Undo/Redo buttons in the toolbar.

## Panels

Drag the thin divider between any two panels (2D View / 3D View / Node Graph / Parameters /
Node Library) to resize it.

## A note on this build

`mc_app` needs a Vulkan-capable driver already installed on your system (Mesa RADV/ANV on
AMD/Intel, or the proprietary driver on Nvidia) — this isn't something the app bundles. If it
fails to launch at all, that's the first thing to check before anything else.
