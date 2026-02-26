# Claude Code Instructions for jack-voice

## Project Overview

`jack-voice` is a local-first voice stack for developer tools: VAD, STT, TTS, turn detection, and model management in Rust, with an NDJSON bridge, WebSocket server, and SDKs for coding CLI integration.

The goal: any CLI should be able to add voice in one or two lines after `connect()`.

## Workspace Layout

```
jack-voice/                     # Workspace root
├── Cargo.toml                  # Workspace manifest (members: jack-voice, jack-voice-bridge, jack-voice-ws)
├── SPEC.md                     # V1 protocol specification
├── ROADMAP.md                  # Calendar-dated milestones to GA
├── PLAN.md                     # Sub-agent delivery plan
├── CHANGELOG.md                # Keep-a-Changelog format
├── docs/                       # Design memos and documentation
│   ├── MEMO.md
│   └── MEMO_GLM5.md
│
├── jack-voice/                 # Core voice library crate
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs
│   │   ├── vad.rs, stt.rs, tts.rs, etc.
│   │   └── fixtures/          # Test audio files
│   └── tests/                   # Integration tests
│
├── jack-voice-bridge/          # NDJSON subprocess bridge binary
│   ├── Cargo.toml
│   ├── src/
│   └── tests/
│
├── jack-voice-ws/             # OpenAI-compatible WebSocket server
│   ├── Cargo.toml
│   ├── src/
│   └── migrations/
│
├── adapters/                   # CLI tool integrations
│   ├── bin/                    # CLI adapter scripts (codex-voice, voice-stt, etc.)
│   ├── cli/                    # Installation scripts
│   ├── cli_voice.py
│   └── README.md
│
├── sdk/                        # SDK wrappers
│   ├── ts/jack-voice-sdk/
│   └── python/jack-voice-sdk/
│
└── examples/                   # Cross-crate examples
```

## Build Commands

```bash
# Full workspace
cargo check --workspace
cargo build --workspace
cargo test --workspace
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings

# Individual crates
cargo build -p jack-voice
cargo build -p jack-voice-bridge
cargo build -p jack-voice-ws
cargo test -p jack-voice-ws

# Feature builds
cargo build -p jack-voice --features cuda
cargo build -p jack-voice --features directml
cargo build -p jack-voice-ws --features livekit

# Release build
cargo build --workspace --release
```

## Architecture

### Crates
- `jack-voice`: Core voice library (VAD/STT/TTS/models)
- `jack-voice-bridge`: NDJSON stdio bridge
- `jack-voice-ws`: WebSocket server (OpenAI-compatible)

### Dependencies
- `supertonic`: Pulled in as dependency of jack-voice (not a workspace member)

## Testing

All crates use `tests/` directory for integration tests:
```
jack-voice/tests/
jack-voice-bridge/tests/
```

Unit tests use inline `#[cfg(test)]` modules in source files.

## CLI Adapters

Located in `adapters/bin/` with `<tool>-voice` naming:
- codex-voice
- claude-voice
- voice-stt
- voice-tts
- etc.
