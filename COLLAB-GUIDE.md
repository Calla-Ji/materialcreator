# MaterialCreator — Collaboration Guide

Quick reference for building, running, and syncing the project.

---

## Setup (one-time)

```bash
# Clone the repo
cd /Users/stellaachenbach/development
git clone https://github.com/Calla-Ji/materialcreator.git

# Verify
ls materialcreator
# You should see Cargo.toml, README.md, crates/, etc.
```

---

## Building & Running

```bash
# Go to the project root
cd /Users/stellaachenbach/development/materialcreator

# Build and run the GUI app (first build is slow — 5-15 min)
cargo run --package mc_app

# Rebuild after code changes (cached builds are fast)
cargo run --package mc_app

# Run the CLI (headless, no GUI)
cargo run --package mc_cli

# Run tests
cargo test
```

---

## Syncing with GitHub

```bash
cd /Users/stellaachenbach/development/materialcreator

# Pull the latest from GitHub (after I push changes)
git pull

# Check your local status
git status

# View recent commits
git log --oneline -10
```

---

## Making Changes (Optional)

If you want to experiment locally:

```bash
# Create a new branch for your changes
git checkout -b my-change

# Edit files...

# Commit your changes
git add .
git commit -m "Description of changes"

# Push to GitHub so I can pull them
git push origin my-change
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `cargo: command not found` | Run `source ~/.cargo/env` or add Cargo to PATH |
| Build fails on macOS | Run `xcode-select --install` for Metal support |
| App won't start / GPU error | Make sure you have a Mac with Metal support (2018+) |
| Merge conflict on `git pull` | Run `git status`, edit conflicted files, then `git commit` |

---

---

## Hermes Session Startup Guide

### After updating or restarting your Mac

1. Keep your Mac plugged in with the lid open.
2. Open **oMLX**.
3. Open **Terminal** and paste:

```bash
caffeinate -i ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R 172.16.0.1:18000:127.0.0.1:8000 \
  root@179.197.80.104
```

4. Enter your VPS root password if asked.
5. Leave the Terminal window open — staying quiet without returning to the prompt is normal.
6. Open the **Hermes dashboard** and send a message.
   No Hostinger console, reinstallation, or provider setup is normally needed.

---

## Key Paths

- **Project root:** `/Users/stellaachenbach/development/materialcreator/`
- **Source code:** `crates/` (mc_core, mc_app, mc_gpu, mc_nodes, mc_export, mc_cli)
- **Remote repo:** https://github.com/Calla-Ji/materialcreator
