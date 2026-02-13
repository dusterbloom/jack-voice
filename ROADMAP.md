# jack-voice Cross-CLI Roadmap

## Scope and Assumptions
- Goal: make VAD, STT, and TTS available to coding CLIs in one or two lines after connect.
- Integration model: `jack-voice-bridge` subprocess (`stdin`/`stdout` NDJSON) plus TS/Python SDK wrappers.
- Team: 4 engineers (2 Rust runtime, 1 SDK/integration, 1 QA/release).
- Start: **2026-02-16** (first Monday after 2026-02-12).

## Phase 0: Contract Freeze
- Dates: **2026-02-16 to 2026-02-27**
- Goal: lock protocol and success metrics before implementation.

Milestones
- **2026-02-18**: V1 method list frozen (`runtime.hello`, `models.status`, `vad.detect`, `stt.transcribe`, `tts.synthesize`).
- **2026-02-23**: JSON request/response/error schemas approved.
- **2026-02-27**: SLO baseline finalized using fixture audio.

Exit Criteria
- 100% V1 methods have schema and error-code definitions.
- Baseline benchmark runs reproducibly on Linux and macOS.

## Phase 1: Bridge MVP
- Dates: **2026-03-02 to 2026-03-20**
- Goal: ship a stable Rust bridge binary using existing `jack-voice` primitives.

Milestones
- **2026-03-06**: NDJSON parser, request correlation, and timeout handling merged.
- **2026-03-13**: End-to-end VAD/STT/TTS RPCs working locally.
- **2026-03-20**: `session.start` + streaming event flow (`stt.partial`, `stt.final`) implemented.

Dependencies
- `jack-voice/src/vad.rs`, `jack-voice/src/stt.rs`, `jack-voice/src/tts.rs`, `jack-voice/src/models.rs`

Exit Criteria
- All V1 methods pass protocol tests.
- No crashes in 1,000-session soak test.

## Phase 2: SDKs and One-Line Developer Experience
- Dates: **2026-03-23 to 2026-04-10**
- Goal: deliver TS and Python SDKs with identical high-level APIs.

Milestones
- **2026-03-27**: `connect`, `vad`, `stt`, `tts`, `close` APIs complete in both SDKs.
- **2026-04-03**: One-line samples validated inside two host CLIs.
- **2026-04-10**: Beta release with integration guides and minimal examples.

Dependencies
- Stable bridge behavior from Phase 1.
- Packaging/release setup for npm + PyPI.

Exit Criteria
- After connect, `vad`/`stt`/`tts` each run in a single line in TS and Python examples.
- SDK parity tests pass across Linux/macOS/Windows runners.

## Phase 3: Multilingual and Reliability Hardening
- Dates: **2026-04-13 to 2026-05-01**
- Goal: harden multilingual paths and production reliability.

Milestones
- **2026-04-17**: Language/voice override paths validated (`language=auto` plus explicit codes).
- **2026-04-24**: Backpressure, cancellation, and malformed-input defenses complete.
- **2026-05-01**: RC1 tagged with reliability and latency dashboards.

Exit Criteria
- English and multilingual smoke tests pass for STT and TTS.
- Crash-free rate >= 99.5% in soak runs.
- p95 latency stays within targets defined in `SPEC.md`.

## Phase 4: Launch and Adoption
- Dates: **2026-05-04 to 2026-05-22**
- Goal: GA launch and early-adopter onboarding.

Milestones
- **2026-05-08**: Docs pack complete (quickstart, troubleshooting, migration notes).
- **2026-05-15**: Release dry run + rollback drill complete.
- **2026-05-22**: **v1.0.0 GA**.

Exit Criteria
- Fresh install to first successful VAD/STT/TTS call <= 15 minutes.
- No open P0/P1 issues at GA cut.

## Key Risks and Mitigations
- Protocol churn risk.
  Mitigation: freeze schema by 2026-02-27 and version strictly.
- Model/runtime variability across hardware.
  Mitigation: keep deterministic smoke suite and explicit fallback policy.
- SDK drift between TS and Python.
  Mitigation: shared contract tests run in CI for both SDKs.

## Interview Narrative
The plan is intentionally staged: freeze interfaces first, then build the bridge, then package a one-line developer experience in TS/Python, and only then optimize multilingual reliability for GA. This keeps complexity bounded, proves value early, and gives clear, measurable gates at every phase.
