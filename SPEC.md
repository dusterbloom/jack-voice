# jack-voice CLI Integration Spec (V1)

## 1. Problem
Coding CLI tools (for example `codex`, `claude code`, and similar agent CLIs) need voice capabilities that are:
- Local-first (no required cloud service)
- Fast enough for interactive use
- Simple to integrate (ideally 1-2 lines per VAD/STT/TTS action)
- Reliable across English and multilingual use cases

Today, `jack-voice` has strong Rust components (VAD/STT/TTS/model management), but there is no stable cross-language runtime interface optimized for CLI tool integration.

## 2. Goals
- Provide a stable subprocess + stdio JSON runtime protocol (NDJSON framing).
- Make TypeScript and Python integration require:
  - One connect line
  - One line per VAD/STT/TTS call
- Support English defaults and multilingual operation with explicit language/voice controls.
- Keep `jack-voice` as the inference core; add a thin bridge/runtime layer.
- Support both one-shot RPC usage and streaming session usage.
- Be practical to ship in incremental milestones.

## 3. Non-Goals
- Building a network daemon/service as the primary integration path.
- Replacing existing Rust APIs in `jack-voice`.
- Building a GUI or desktop app.
- Solving telephony/media server scale-out in V1.
- Perfect language identification across all languages in V1.

## 4. Target Users
- Maintainers of coding CLIs that want local voice I/O.
- Plugin/extension developers integrating voice into terminal workflows.
- Teams needing privacy-preserving, on-device transcription/synthesis.

## 5. Proposed Architecture
### 5.1 Components
- `jack-voice` (existing Rust library): VAD/STT/TTS/pipeline/model management.
- `jack-voice-bridge` (new Rust binary crate): subprocess entrypoint with stdio JSON protocol.
- `jack-voice-sdk-ts` (new lightweight Node wrapper): spawns bridge, request/response helpers.
- `jack-voice-sdk-py` (new lightweight Python wrapper): same behavior for Python CLIs.

### 5.2 Data Flow
1. Host CLI spawns `jack-voice-bridge`.
2. CLI sends JSON commands on `stdin`.
3. Bridge sends JSON responses/events on `stdout` (one JSON object per line).
4. Optional logs go to `stderr` only (never mixed with protocol stream).

### 5.3 Mapping to Existing Code
- VAD: `VoiceActivityDetector` (`jack-voice/src/vad.rs`)
- STT: `SpeechToText` + Parakeet/Whisper/Moonshine backends (`jack-voice/src/stt.rs`, `jack-voice/src/parakeet_stt.rs`)
- TTS: `TextToSpeech` with Supertonic/Kokoro (`jack-voice/src/tts.rs`, `jack-voice/src/kokoro_tts.rs`)
- Event model: `VoiceEvent` (`jack-voice/src/pipeline.rs`) mapped to JSON events
- Models: `models::*` APIs for presence/download paths (`jack-voice/src/models.rs`)

## 6. SDK Surface
### 6.1 Common Surface (TS and Python)
- `connect(options?) -> client`
- `client.vad(samples, options?) -> VadResult`
- `client.stt(samples, options?) -> SttResult`
- `client.tts(text, options?) -> TtsResult`
- `client.start_session(options?) -> session`
- `session.push_audio(samples)`
- `session.end_audio()`
- `session.events()` async iterator / callback stream
- `client.close()`

### 6.2 One-line Operations (after connect)
- VAD: one call with an audio frame/buffer
- STT: one call with utterance samples
- TTS: one call with text and voice/language options

## 7. Runtime Protocol (Subprocess/Stdio JSON)
### 7.1 Framing
- Transport: NDJSON (UTF-8 JSON objects delimited by `\n`)
- `stdin`: requests/commands
- `stdout`: responses + async events
- `stderr`: logs only

### 7.2 Message Envelope
Request:
```json
{"type":"request","id":"req_123","method":"stt.transcribe","params":{...}}
```

Response:
```json
{"type":"response","id":"req_123","ok":true,"result":{...}}
```

Error response:
```json
{"type":"response","id":"req_123","ok":false,"error":{"code":"MODEL_MISSING","message":"...","retryable":false}}
```

Async event:
```json
{"type":"event","session_id":"sess_1","event":"stt.partial","ts_ms":1730000000000,"data":{...}}
```

### 7.3 Core Methods (V1)
- `runtime.hello`
- `models.status`
- `models.ensure`
- `vad.detect`
- `stt.transcribe`
- `tts.synthesize`
- `session.start`
- `session.audio.push`
- `session.audio.end`
- `session.cancel`
- `runtime.shutdown`

### 7.4 Audio Payload Contract
- Default input format: PCM16 little-endian, mono, 16kHz
- Optional input format: float32 mono, 16kHz
- JSON field:
  - `audio_b64`: base64 encoded bytes
  - `format`: `"pcm_s16le"` or `"f32le"`
  - `sample_rate_hz`: `16000`
  - `channels`: `1`

### 7.5 TTS Output Contract
- Response includes:
  - `audio_b64`
  - `format` (`"pcm_f32le"` in V1)
  - `sample_rate_hz` (engine-dependent, e.g. 24000 for Kokoro)
  - `duration_ms`

### 7.6 Session Events (Streaming Mode)
- `session.ready`
- `vad.speech_start`
- `stt.partial`
- `stt.final`
- `turn.complete`
- `tts.start`
- `tts.chunk` (optional in V1; single chunk allowed)
- `tts.end`
- `warning`
- `error`

### 7.7 Cancellation + Backpressure
- Every request may include `timeout_ms`.
- Long-running operations support `session.cancel`.
- Bridge emits `warning` with `code: "INPUT_BACKPRESSURE"` when internal queue crosses threshold.

### 7.8 Versioning
- `runtime.hello` returns protocol version and supported methods.
- Breaking protocol changes require major version bump.

## 8. Model Strategy
### 8.1 Defaults
- VAD default: Silero.
- STT default policy:
  - English low-latency mode: Moonshine/Parakeet path selected by configured mode.
  - Multilingual/fallback mode: Whisper Turbo or Parakeet TDT.
- TTS default policy:
  - English default: Supertonic or Kokoro English voice.
  - Multilingual default: Kokoro voice/language map; Supertonic where supported.

### 8.2 Selection Controls
- Per-request overrides:
  - `stt.mode`: `streaming | batch | auto`
  - `stt.language`: explicit code or `auto`
  - `tts.engine`: `supertonic | kokoro | auto`
  - `tts.voice`: explicit voice ID

### 8.3 Downloading and Caching
- `models.ensure` is explicit; runtime does not auto-download unless configured.
- Model cache directory:
  - default from `jack-voice::models::get_models_dir()`
  - override via env (`JACK_VOICE_MODELS_DIR`) and/or `runtime.hello` options
- Optional checksum manifest in V1.1 (recommended gate before GA).

### 8.4 Offline Behavior
- If model missing and offline mode enabled: deterministic `MODEL_MISSING` error.
- No network calls during inference path.

## 9. Performance / SLO Targets
Targets are for reference CPU laptop class hardware with models pre-downloaded.

- Bridge startup to `runtime.hello` ready:
  - p95 <= 300 ms (process-level ready)
- First successful VAD call (model loaded):
  - p95 <= 50 ms for 20-40 ms frame
- STT English (2-4 second utterance):
  - p95 <= 1500 ms to final text
- STT Multilingual (2-4 second utterance):
  - p95 <= 2200 ms to final text
- TTS first audio byte for <= 120 char input:
  - p95 <= 900 ms
- Crash-free session reliability:
  - >= 99.5% across 10k session starts

## 10. Security & Privacy
- Default local-only inference; no telemetry/audio upload by default.
- `models.ensure` is the only network-capable path and must be explicit.
- No shell command execution from protocol methods.
- Protocol parser must enforce:
  - max message size
  - max base64 payload size
  - schema validation
- Optional runtime controls:
  - `allow_network_downloads` flag (default false in enterprise mode)
  - `redact_text_in_logs` flag (default true)
- Audio/text data retention:
  - in-memory only unless caller explicitly asks to persist outputs.

## 11. Observability
- Structured logs to `stderr` with `request_id`, `session_id`, `method`, `latency_ms`.
- Metrics emitted as events (or optional stats endpoint in future):
  - `request_count`, `error_count`
  - `request_latency_ms`
  - `stt_latency_ms`, `tts_latency_ms`
  - `audio_drop_count`, `backpressure_count`
  - `model_load_ms` per engine
- Health method:
  - `runtime.hello` includes model availability and capability flags.

## 12. Concrete Integration Examples
### 12.1 TypeScript (Node CLI)
```ts
const jv = await JackVoice.connect();                       // spawns subprocess bridge
const speech = await jv.vad(frame16k);                      // 1 line VAD
const text = await jv.stt(utterance16k, { language: "auto" }); // 1 line STT
const wav = await jv.tts("Build finished.", { voice: "if_sara" }); // 1 line TTS (multilingual voice)
```

### 12.2 Python (CLI tool)
```python
jv = JackVoice.connect()                                    # spawns subprocess bridge
speech = jv.vad(frame_16k)                                  # 1 line VAD
text = jv.stt(utterance_16k, language="auto")               # 1 line STT
audio = jv.tts("Compilazione completata.", voice="35")      # 1 line TTS (Italian Kokoro voice)
```

## 13. Acceptance Criteria
- Protocol:
  - NDJSON stdio protocol implemented with request/response correlation and async events.
  - Documented method and event schemas with stable error codes.
- SDKs:
  - TS and Python wrappers shipped with identical high-level methods (`vad`, `stt`, `tts`).
  - Each operation works in one line after connect.
- Functionality:
  - English and multilingual STT/TTS verified with automated smoke tests.
  - Session streaming mode emits `stt.partial` and `stt.final` when supported by backend.
- Reliability:
  - Bridge handles malformed JSON without process crash.
  - Cancellation and timeout behavior validated.
- Privacy:
  - No outbound network during inference-only tests.

## 14. Rollout Gates
### Gate 0: Spec + Protocol Freeze
- This spec approved.
- JSON schema files created for methods/events/error codes.

### Gate 1: Internal Alpha (Rust bridge only)
- `jack-voice-bridge` supports `runtime.hello`, `models.status`, `vad.detect`, `stt.transcribe`, `tts.synthesize`.
- Manual tests with sample audio in repo fixtures.

### Gate 2: SDK Beta (TypeScript + Python)
- Thin SDKs released with subprocess management and one-line APIs.
- Integration smoke tests against at least two host CLIs.
- Basic observability fields verified.

### Gate 3: Reliability + Security Hardening
- Fuzz/negative tests for protocol parser.
- Payload-size limits and timeout enforcement complete.
- Optional checksum verification for model files enabled.

### Gate 4: GA
- SLO targets met in CI benchmark environment.
- Backward compatibility policy documented.
- Versioned protocol and migration notes published.

## 15. Implementation Notes (Practical First Slice)
- Keep V1 scope narrow:
  - one-shot VAD/STT/TTS RPC first
  - session streaming second
- Reuse existing `VoiceEvent` semantics for event names where possible.
- Avoid protocol churn: generate TS/Python types from JSON schema once method set is stable.
