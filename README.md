# jack-voice

Stop typing. Start talking.

`jack-voice` is a local-first voice stack for developer tools: VAD, STT, TTS, turn detection, and model management in Rust, with an NDJSON bridge + SDKs for coding CLI integration.

## Why This Exists

Developer tools are moving from keyboard-only to conversational workflows. Most teams still need to stitch together multiple voice components, providers, and APIs just to get basic speech I/O.

`jack-voice` gives you one inference core for:
- Speech start detection (VAD)
- Speech-to-text (STT)
- Text-to-speech (TTS)
- Turn completion logic
- Model download and cache management

The goal is simple: any CLI should be able to add voice in one or two lines.

## Project Status

- Available now:
  - Rust workspace with `jack-voice` and `supertonic` crates
  - `jack-voice-bridge` subprocess runtime (`stdin`/`stdout` NDJSON)
  - Prototype SDK wrappers:
    - `sdk/ts/jack-voice-sdk-ts`
    - `sdk/python/jack_voice_sdk`
- Spec and delivery plan: `SPEC.md`, `ROADMAP.md`, `PLAN.md`.

## What You Can Use Today (Rust)

`jack-voice` is still library-first at the core, with bridge/SDK layers now available as implementation prototypes.

### Workspace
- `jack-voice/`: core voice pipeline primitives (VAD/STT/TTS/models/turn detection)
- `supertonic/`: standalone Supertonic TTS engine

### Build and Test
```bash
cargo check --workspace
cargo build --workspace
cargo test --workspace
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

### Bridge Smoke Test
```bash
cargo build -p jack-voice-bridge
export LD_LIBRARY_PATH=$PWD/target/debug:${LD_LIBRARY_PATH}
echo '{"type":"request","id":"1","method":"runtime.hello","params":{}}' | ./target/debug/jack-voice-bridge
```

### End-to-End Smoke Test
```bash
cargo build -p jack-voice-bridge
python3 jack-voice-bridge/scripts/smoke_test.py --bridge ./target/debug/jack-voice-bridge
```

### Full Local Verification
```bash
bash scripts/verify_local.sh
```

### Minimal Rust Usage
```rust
use jack_voice::{models, NoopProgress, VoiceActivityDetector, SpeechToText, TextToSpeech, TtsEngine, SttMode};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1) Ensure required models are present
    models::ensure_models(&NoopProgress).await?;

    // 2) Create engines
    let mut vad = VoiceActivityDetector::new()?;
    let mut stt = SpeechToText::new(SttMode::Batch)?;
    let mut tts = TextToSpeech::with_engine(TtsEngine::Kokoro)?;

    // 3) Use your own captured 16k mono samples
    let samples: Vec<f32> = vec![];
    let _ = vad.process(&samples)?;
    let text = stt.transcribe(&samples)?;
    let audio = tts.synthesize(&text.text)?;

    println!("Transcribed: {}", text.text);
    println!("Synthesized {} samples @ {} Hz", audio.samples.len(), audio.sample_rate);
    Ok(())
}
```

## Developer Experience

The integration model for coding CLIs is a subprocess bridge plus thin SDK wrappers.

### TypeScript (prototype)
```ts
const jv = await JackVoice.connect();
const speech = await jv.vad(frame16k);
const text = await jv.stt(utterance16k, { language: "auto" });
const wav = await jv.tts("Build finished.", { voice: "F1" }); // Supertonic voice
```

### Python (prototype)
```python
jv = JackVoice.connect()
speech = jv.vad(frame_16k)
text = jv.stt(utterance_16k, language="auto")
audio = jv.tts("Compilazione completata.", engine="kokoro", voice="35")
```

### Host CLI Integration Pattern

`codex`, `claude code`, `opencode`, `droid`, and `pi` can all integrate the same way:

1. Spawn `jack-voice-bridge` as a subprocess.
2. Keep one long-lived process per CLI session.
3. Send NDJSON requests (`vad.detect`, `stt.transcribe`, `tts.synthesize`) over stdin and read responses from stdout.

If your host runtime is Node, use `sdk/ts/jack-voice-sdk-ts`.  
If your host runtime is Python, use `sdk/python/jack_voice_sdk`.  
If your host runtime is something else, call the NDJSON protocol directly.

Both SDKs now auto-bootstrap runtime configuration across Linux/macOS/Windows:
1. Resolve bridge command from `JACK_VOICE_BRIDGE_CMD`, then local `target/{debug,release}`, then `PATH`.
2. Auto-configure dynamic library search paths.
3. Auto-set WSL Pulse server (`/mnt/wslg/PulseServer`) when available.

### Ready-Made CLI Adapters

For fastest adoption, use the wrappers in `adapters/`:

```bash
cargo build -p jack-voice-bridge
python3 scripts/install_cli_adapters.py
jv-doctor
```

Windows equivalent installer command: `py -3 scripts/install_cli_adapters.py`.

`jv-doctor` now verifies VAD/STT/TTS availability without needing microphone input.

Then try:

```bash
codex-voice
claude-voice
opencode-voice
droid-voice
pi-voice
jv-chat --target codex
```

Use `--print-only` to validate mic + STT flow when a target CLI binary is unavailable.

Details: `adapters/README.md`.

Implementation details, acceptance criteria, and rollout gates live in `SPEC.md`.

## Architecture (Current + Planned)

- Inference core (`jack-voice`):
  - VAD via Silero
  - STT via Moonshine/Whisper/Parakeet
  - TTS via Supertonic/Kokoro
- TTS engine crate (`supertonic`): diffusion-based local synthesis
- Runtime: `jack-voice-bridge` over `stdin/stdout` NDJSON
- SDKs: TypeScript + Python wrappers with parity tests (prototype stage)

## Multilingual Strategy

- English defaults for low-latency developer workflows.
- Multilingual STT/TTS via Whisper Turbo / Parakeet / Kokoro paths.
- Explicit language and voice controls as first-class options in the bridge SDKs.

## Related Docs

- `SPEC.md`: V1 technical specification and protocol contract
- `ROADMAP.md`: dated milestones to GA
- `PLAN.md`: sub-agent execution plan and quality gates
- `CHANGELOG.md`: project change history
- `AGENTS.md`: contributor guidance for this repository

## Contributing

Focus on small, test-backed changes and keep APIs explicit.

Recommended loop:
1. `cargo fmt --all`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo test --workspace`

If you are building toward the bridge/SDK vision, align proposals with `SPEC.md` before implementing.
