# Claude Code Instructions for jack-voice

## Project Overview

`jack-voice` is a local-first voice stack for developer tools: VAD, STT, TTS, turn detection, and model management in Rust, with an NDJSON bridge + SDKs for coding CLI integration.

The goal: any CLI should be able to add voice in one or two lines after `connect()`.

## Workspace Layout

```
jack-voice/                     # Workspace root
├── Cargo.toml                  # Workspace manifest (members: jack-voice, jack-voice-bridge, supertonic)
├── SPEC.md                     # V1 protocol specification (the source of truth)
├── ROADMAP.md                  # Calendar-dated milestones to GA (2026-05-22)
├── PLAN.md                     # Sub-agent delivery plan and quality gates
├── CHANGELOG.md                # Keep-a-Changelog format
│
├── jack-voice/                 # Core voice library crate
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs              # Public API re-exports
│       ├── vad.rs              # Voice Activity Detection (Silero ONNX)
│       ├── stt.rs              # Speech-to-Text (Whisper/Moonshine/Parakeet)
│       ├── parakeet_stt.rs     # Parakeet TDT/EOU backends
│       ├── tts.rs              # Text-to-Speech (Supertonic/Kokoro)
│       ├── kokoro_tts.rs       # Kokoro multilingual TTS
│       ├── models.rs           # Model download, caching, path management
│       ├── pipeline.rs         # VoicePipeline, VoiceEvent
│       ├── audio.rs            # Audio capture/playback (cpal/rodio)
│       ├── audio_quality.rs    # Audio quality gates
│       ├── calibration.rs      # Voice calibration
│       ├── speaker.rs          # Speaker profiles, SttMode, TurnMetrics
│       ├── turn_detector.rs    # Smart turn detection
│       └── watchdog.rs         # Timeout tracking
│
├── jack-voice-bridge/          # NDJSON subprocess bridge binary
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs             # Bridge runtime loop (stdin/stdout NDJSON)
│       ├── protocol.rs         # Request/Response envelopes, RPC methods, error codes
│       └── audio.rs            # Base64 audio encoding/decoding
│
├── supertonic/                 # Standalone Supertonic TTS engine
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── phonemizer.rs
│       └── voice_style.rs
│
├── adapters/                   # CLI adapter wrappers (codex-voice, claude-voice, etc.)
│   ├── cli_voice.py
│   └── README.md
│
├── sdk/                        # Prototype SDK wrappers
│   ├── ts/jack-voice-sdk-ts/
│   └── python/jack_voice_sdk/
│
└── scripts/                    # Build/install/verify scripts
    ├── verify_local.sh
    └── install_cli_adapters.py
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
cargo build -p supertonic
cargo test -p jack-voice
cargo test -p jack-voice-bridge

# Feature builds
cargo build -p jack-voice --features cuda       # CUDA acceleration
cargo build -p jack-voice --features directml   # DirectML (Windows)

# Release build
cargo build --workspace --release
```

## Bridge Smoke Test

```bash
cargo build -p jack-voice-bridge
export LD_LIBRARY_PATH=$PWD/target/debug:${LD_LIBRARY_PATH}
echo '{"type":"request","id":"1","method":"runtime.hello","params":{}}' | ./target/debug/jack-voice-bridge
```

## Architecture

### Data Flow
```
Host CLI (codex, claude code, etc.)
    │
    ├── spawns jack-voice-bridge as subprocess
    │
    ├── stdin  → NDJSON requests  → bridge
    ├── stdout ← NDJSON responses ← bridge
    └── stderr ← logs only        ← bridge
```

### NDJSON Protocol (V1)

Request envelope:
```json
{"type":"request","id":"req_123","method":"stt.transcribe","params":{...}}
```

Response envelope:
```json
{"type":"response","id":"req_123","ok":true,"result":{...}}
```

Error response:
```json
{"type":"response","id":"req_123","ok":false,"error":{"code":"MODEL_MISSING","message":"...","retryable":false}}
```

### RPC Methods (V1)
| Method              | Purpose                                    |
|---------------------|--------------------------------------------|
| `runtime.hello`     | Handshake, protocol version, model status  |
| `models.status`     | Check which models are downloaded          |
| `models.ensure`     | Download missing models                    |
| `vad.detect`        | Voice activity detection on audio frame    |
| `stt.transcribe`    | Speech-to-text on audio buffer             |
| `tts.synthesize`    | Text-to-speech generation                  |
| `runtime.shutdown`  | Graceful shutdown                          |

### Error Codes
`PARSE_ERROR`, `INVALID_REQUEST`, `INVALID_PARAMS`, `METHOD_NOT_FOUND`, `PAYLOAD_TOO_LARGE`, `UNSUPPORTED_AUDIO_FORMAT`, `AUDIO_DECODE_FAILED`, `MODEL_MISSING`, `OPERATION_TIMEOUT`, `INTERNAL_ERROR`

### Audio Contract
- Input: PCM16 LE mono 16kHz or float32 LE mono 16kHz, base64-encoded in `audio_b64`
- TTS output: float32 LE, sample rate varies by engine (24000 for Kokoro, etc.)

## TTS Engines

Currently two engines, with a third (Pocket TTS) planned:

| Engine     | Type                  | Languages | Status      |
|------------|-----------------------|-----------|-------------|
| Supertonic | Diffusion (ONNX)      | English   | Working     |
| Kokoro     | Neural (kokoro-tiny)  | Multi     | Working     |
| Pocket TTS | Transformer (Candle)  | English   | **Planned** |

### Pocket TTS Integration (Next Priority)

[Pocket TTS](https://github.com/babybirdprd/pocket-tts) is a 100M parameter CPU-optimized TTS by Kyutai Labs, ported to pure Rust via Candle. It should replace Supertonic as the default English TTS.

**Why:**
- ~200ms first chunk latency, faster-than-realtime generation
- Pure Rust (Candle), no ONNX runtime dependency
- Streaming support (FlowLM + Mimi codec)
- Voice cloning from any WAV file
- int8 quantization for smaller memory footprint
- Metal acceleration on macOS (`--features metal`)
- WASM support for browser deployment

**Crate:** `pocket-tts = "0.6.2"` on crates.io

**Integration plan:**
1. Add `TtsEngine::Pocket` variant to `jack-voice/src/tts.rs`
2. Add `pocket-tts` dependency to `jack-voice/Cargo.toml`
3. Implement `new_pocket()` and wire into `synthesize()` / `synthesize_streaming()`
4. Add `"pocket"` to `parse_tts_engine()` in `jack-voice-bridge/src/main.rs`
5. Add `CachedTtsEngine::Pocket` to the bridge state
6. Update auto-selection fallback order: Pocket → Kokoro → Supertonic
7. Model weights auto-download from HuggingFace (gated — requires `HF_TOKEN`)

**Key API:**
```rust
use pocket_tts::TTSModel;

let model = TTSModel::load("b6369a24")?;
let voice_state = model.get_voice_state("voice.wav")?;  // or use preset: "alba"
let audio = model.generate("Hello world", &voice_state)?;

// Streaming:
for chunk in model.generate_stream("Long text...", &voice_state) {
    let audio_chunk = chunk?;
    // play or send chunk
}
```

**Voices:** `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`

## STT Backends

| Backend         | Mode      | Languages | Notes                    |
|-----------------|-----------|-----------|--------------------------|
| Whisper Base    | Batch     | English   | Default low-latency      |
| Moonshine       | Streaming | English   | Real-time partial results |
| Whisper Turbo   | Batch     | Multi     | Multilingual fallback    |
| Parakeet TDT    | Batch     | Multi     | NVIDIA Parakeet          |
| Parakeet EOU    | Batch     | Multi     | End-of-utterance variant |

## Key Dependencies

| Crate            | Purpose                              |
|------------------|--------------------------------------|
| `sherpa-rs`      | Silero VAD + Whisper/Moonshine STT   |
| `kokoro-tiny`    | Kokoro multilingual TTS              |
| `parakeet-rs`    | Parakeet STT backends                |
| `ort`            | ONNX Runtime (Smart Turn, Supertonic)|
| `cpal`           | Audio capture                        |
| `rodio`          | Audio playback                       |
| `pocket-tts`     | Pocket TTS (planned)                 |

## Coding Conventions

- **Formatting:** `cargo fmt --all` (default rustfmt, 4-space indent)
- **Naming:** `snake_case` functions/modules, `CamelCase` types, `SCREAMING_SNAKE_CASE` constants
- **Errors:** `thiserror` for error types, return `Result` with context
- **Tests:** inline `#[cfg(test)]` modules, name tests `test_<behavior>`
- **Commits:** imperative subjects with optional scope: `tts: add pocket-tts engine support`
- **No stdout in bridge:** Bridge stdout is the protocol channel. Use `eprintln!` for logs.

## Common Patterns

### Adding a new TTS engine

1. **`jack-voice/src/tts.rs`:**
   - Add variant to `TtsEngine` enum
   - Add variant to `TtsImpl` enum
   - Add `new_<engine>()` constructor
   - Wire into `with_engine()`, `synthesize()`, `synthesize_streaming()`
   - Add `set_speaker()` handling
   - Add `available_<engine>_voices()` if applicable

2. **`jack-voice-bridge/src/main.rs`:**
   - Add `CachedTtsEngine::<Engine>` variant with `as_str()` and `as_jack_voice_engine()`
   - Add `RequestedTtsEngine::<Engine>` variant
   - Update `parse_tts_engine()` to accept the new name
   - Update `ensure_tts_instance()` auto-selection fallback chain

3. **`jack-voice/src/models.rs`:**
   - Add model readiness check function
   - Add model download/ensure function
   - Update `build_models_status()` in bridge

### Adding a new RPC method

1. Add variant to `RpcMethod` enum in `protocol.rs`
2. Add string mapping in `FromStr` impl and `supported()` list
3. Add params struct in `main.rs`
4. Add handler function `handle_<method>()`
5. Wire into `dispatch_request()` match
6. Add test in `protocol.rs` tests

## Environment Variables

| Variable                 | Purpose                                    |
|--------------------------|--------------------------------------------|
| `JACK_VOICE_MODELS_DIR`  | Override model cache directory             |
| `HF_TOKEN`               | HuggingFace token (for gated model downloads) |
| `LD_LIBRARY_PATH`        | Must include ONNX runtime libs on Linux    |
| `JACK_VOICE_BRIDGE_CMD`  | Override bridge binary path (SDK use)      |

## Performance Targets (from SPEC.md)

- Bridge startup to `runtime.hello`: p95 ≤ 300ms
- VAD call (20-40ms frame): p95 ≤ 50ms
- STT English (2-4s utterance): p95 ≤ 1500ms
- STT Multilingual (2-4s utterance): p95 ≤ 2200ms
- TTS first audio byte (≤120 chars): p95 ≤ 900ms
- Crash-free reliability: ≥ 99.5% across 10k sessions

## Current State & Next Steps

### Done
- Core voice library with VAD/STT/TTS
- NDJSON bridge with all V1 methods
- Prototype TS and Python SDKs
- CLI adapters for codex-voice, claude-voice, etc.

### Next Priority: Pocket TTS Integration
Add `pocket-tts` as a third TTS engine. It should become the default for English because it's faster, more reliable, and has better voice quality than Supertonic, while being pure Rust with no ONNX dependency.

### After That (per ROADMAP.md)
- Phase 1 (by 2026-03-20): Bridge MVP with streaming sessions
- Phase 2 (by 2026-04-10): SDK beta with one-line DX
- Phase 3 (by 2026-05-01): Multilingual hardening
- Phase 4 (by 2026-05-22): GA release

## Important Files to Read First

If you're new to this codebase, read in this order:
1. This file (`CLAUDE.md`)
2. `SPEC.md` — the protocol contract
3. `jack-voice/src/tts.rs` — TTS abstraction layer
4. `jack-voice-bridge/src/main.rs` — bridge runtime
5. `jack-voice-bridge/src/protocol.rs` — RPC types
6. `jack-voice/src/lib.rs` — public API surface
