# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Added `SPEC.md` defining the V1 cross-CLI integration architecture: subprocess bridge, NDJSON protocol, and one-line VAD/STT/TTS SDK calls.
- Added `ROADMAP.md` with calendar-dated milestones from 2026-02-16 through planned GA on 2026-05-22.
- Added `PLAN.md` with sub-agent workstream ownership, checkpoints, and quality gates.
- Added workspace crate `jack-voice-bridge` implementing NDJSON stdio methods:
  `runtime.hello`, `models.status`, `models.ensure`, `vad.detect`, `stt.transcribe`, `tts.synthesize`, `runtime.shutdown`.
- Added prototype TypeScript SDK under `sdk/ts/jack-voice-sdk-ts` (subprocess client with `connect`, `vad`, `stt`, `tts`, `close`).
- Added prototype Python SDK under `sdk/python/jack_voice_sdk` (subprocess client with `connect`, `vad`, `stt`, `tts`, `close`).
- Added `jack-voice-bridge/scripts/smoke_test.py` for repeatable local end-to-end protocol checks.
- Added `scripts/verify_local.sh` to run end-to-end local validation (Rust tests + bridge smoke + Python SDK smoke + Node bridge smoke) in one command.
- Added automatic runtime bootstrap in SDKs to discover `jack-voice-bridge` and configure platform-specific loader paths without manual env setup.
- Added `adapters/cli_voice.py` plus wrapper commands (`codex-voice`, `claude-voice`, `opencode-voice`, `droid-voice`, `pi-voice`) for fast CLI-targeted tryout.
- Added `scripts/install_cli_adapters.sh` to install adapter commands into `~/.local/bin`.
- Added `chat` mode in `adapters/cli_voice.py` and `jv-chat` wrapper for push-to-talk loop usage.
- Added cross-platform `scripts/install_cli_adapters.py` installer for Linux/macOS/Windows command setup.

### Changed
- Set initiative baseline around one-line integration goals (after `connect`) for VAD/STT/TTS with English defaults and multilingual support.
- Added `jack-voice-bridge` to workspace members in `Cargo.toml`.
- Refactored bridge runtime loop to avoid nested Tokio runtime panics during `tts.synthesize`.
- Fixed `kokoro_tts::needs_direct_pipeline` routing so English voices use the built-in path and multilingual voices use the direct pipeline.
- Updated Python SDK docs/examples to use `engine="kokoro"` with voice `35` and include a no-install `PYTHONPATH` run path.
- Updated README and SDK docs to document platform-agnostic auto-discovery (`JACK_VOICE_BRIDGE_CMD` -> local targets -> `PATH`) and WSL Pulse auto-config behavior.
- Updated `README.md` and new `adapters/README.md` with one-command setup and per-CLI wrapper usage.
- Expanded adapter `doctor` checks to probe VAD/STT/TTS readiness in a single command (no mic required).

## [0.1.0] - 2026-02-12

### Added
- Baseline release snapshot created from the current repository state (no prior commit history available to reconstruct earlier dated entries).
- Workspace containing `jack-voice` and `supertonic` library crates with core voice-pipeline and TTS components.
