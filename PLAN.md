# Sub-Agent Delivery Plan

## Objective
- Deliver a production-ready integration stack where coding CLIs can call VAD/STT/TTS in one line after `connect()`, with English defaults and multilingual support.

## Sub-Agent Topology
- `Agent A (Protocol + Bridge Owner)`
  Owns `jack-voice-bridge` protocol implementation (NDJSON, request correlation, error taxonomy, timeouts, cancellation).
- `Agent B (Core Voice Adapter Owner)`
  Maps bridge methods to existing `jack-voice` modules (`vad`, `stt`, `tts`, `models`) and owns model-readiness behavior.
- `Agent C (SDK Owner)`
  Builds and tests TypeScript + Python SDK wrappers with identical APIs.
- `Agent D (Quality + Release Owner)`
  Owns contract tests, soak tests, latency reporting, docs examples, changelog/release checklist.

## Workstreams
- `WS1 Contract`: finalize schemas for methods/events/errors and version policy.
- `WS2 Runtime`: implement bridge process and session/event lifecycle.
- `WS3 SDKs`: expose `connect`, `vad`, `stt`, `tts`, `close` with one-line examples.
- `WS4 Quality`: add cross-language parity tests, fuzz/negative protocol tests, soak and latency checks.
- `WS5 Release`: package artifacts, migration notes, and GA checklist.

## Dated Checkpoints
- `C0 (2026-02-16)`: kickoff and ownership lock.
- `C1 (2026-02-27)`: schema freeze (`WS1` done).
- `C2 (2026-03-20)`: bridge MVP accepted (`WS2` done).
- `C3 (2026-04-10)`: TS/Python beta SDKs accepted (`WS3` done).
- `C4 (2026-05-01)`: reliability gates pass (`WS4` done).
- `C5 (2026-05-22)`: GA release readiness sign-off (`WS5` done).

## Handoff Artifacts
- `H1`: Protocol schema package (JSON schemas + error code table).
- `H2`: Bridge test report (unit, integration, malformed-input behavior).
- `H3`: SDK parity report (TS vs Python behavior matrix).
- `H4`: Performance report (p50/p95 for VAD/STT/TTS).
- `H5`: Launch bundle (docs, release notes, rollback checklist).

## Quality Gates
- Protocol compatibility tests pass on Linux/macOS/Windows.
- One-line examples run successfully in both SDKs.
- English + multilingual smoke tests pass for STT and TTS.
- Crash-free soak run >= 99.5% across 1,000 sessions.

## Definition of Done
- `SPEC.md` acceptance criteria are satisfied.
- `ROADMAP.md` phase exit criteria are met through Phase 4.
- GA docs allow a fresh user to run successful `connect + vad/stt/tts` in <= 15 minutes.
- Remaining gaps are explicitly documented with owner and follow-up date.
