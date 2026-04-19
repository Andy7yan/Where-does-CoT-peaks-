# AGENTS.md

This repository expects coding agents to ground themselves in the local project docs before making changes.

## Always Read First

- Read this file at the start of every task.
- Then read the relevant files under `.agents/` for the area you are touching.

## Required Doc Lookups

- Katana, PBS, scratch paths, run layout, or upload/sync workflow:
  read `.agents/Katana-guide.md`
- Naming, directory layout, public artifact names, or "what should not leak from internal docs":
  read `.agents/project-structure.md`
- Original Stage 1 difficulty-group semantics:
  read `.agents/stage1-spec-v6.md`
- Per-question medium/hard branch semantics:
  read `.agents/stage1-spec-v6-to-v7.md`

## Repository Rules

- Large regenerable artifacts on Katana belong under `${SCRATCH}/runs/...`, not under the repo home tree.
- When a source run is needed on Katana, prefer resolving it from `${SCRATCH}/runs/<run-name>`.
- Names from internal `.agents/` documents must not leak into active code, script, config, artifact, or schema names.
- When giving Katana instructions, assume the code tree at `/home/${USER}/peak-CoT` is manually synced; do not assume `git pull` is available.
- Before suggesting a Katana rerun after a code change, account for whether the changed files have been uploaded.
- If the task asks for Katana commands, PBS, Bash submission helpers, or any action that depends on run outputs, first re-check `.agents/Katana-guide.md`.
- Before printing Katana commands to the user, provide a minimal file-version checklist that states which local and Katana-side files must match for the command to be valid.
- Do not execute file renames, file deletions, or file moves on the user's behalf.
- For rename, delete, or move operations, state explicitly which path should become which new path, which path should be deleted, or which path should be moved where, and let the user perform that filesystem operation manually.
- Do not simulate renames by deleting an old file and creating a new one when the user's real intent is a rename or move.
