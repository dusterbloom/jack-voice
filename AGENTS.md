# Repository Guidelines

## Project Structure & Module Organization
This repository is a Rust workspace with two library crates:
- `jack-voice/`: main voice pipeline components (audio, VAD, STT, TTS, turn detection, model management).
- `supertonic/`: standalone Supertonic TTS implementation (core engine, phonemizer, voice style handling).

Primary entry points are `jack-voice/src/lib.rs` and `supertonic/src/lib.rs`. Most code is organized as module-per-domain under each crate’s `src/`. Test fixtures (WAV samples) live in `jack-voice/src/fixtures/`. Build artifacts are generated in `target/` and should not be edited.

## Build, Test, and Development Commands
- `cargo check --workspace`: fast compile checks for all crates.
- `cargo build --workspace`: full workspace build.
- `cargo test --workspace`: run all unit tests.
- `cargo test -p jack-voice` / `cargo test -p supertonic`: run crate-specific tests.
- `cargo fmt --all`: apply standard Rust formatting.
- `cargo clippy --workspace --all-targets -- -D warnings`: lint strictly.

Feature builds for `jack-voice`:
- `cargo build -p jack-voice --features cuda`
- `cargo build -p jack-voice --features directml`

## Coding Style & Naming Conventions
Use idiomatic Rust and keep code `rustfmt`-clean (default 4-space indentation). Follow standard naming:
- `snake_case` for files, modules, and functions.
- `CamelCase` for structs, enums, and traits.
- `SCREAMING_SNAKE_CASE` for constants.

Keep modules focused by subsystem and re-export stable API surfaces from `lib.rs`. Prefer explicit error types (`thiserror`) and return `Result` with clear context.

## Testing Guidelines
Tests are mostly inline module tests under `#[cfg(test)]`. Keep new tests close to implementation and name them `test_<behavior>`. Favor deterministic unit tests using synthetic data; avoid network/model download paths in unit coverage. Reuse fixtures in `jack-voice/src/fixtures/` when audio files are needed.

## Commit & Pull Request Guidelines
This repository currently has no commit history, so conventions are bootstrapped here:
- Write imperative commit subjects with optional scope (example: `supertonic: tighten gap detection thresholds`).
- Keep commits small and behavior-focused; include relevant tests.
- PRs should include a concise summary, affected crate(s), commands run (check/test/lint), and any model/runtime prerequisites.
