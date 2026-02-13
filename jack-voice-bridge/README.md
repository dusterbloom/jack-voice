# jack-voice-bridge

`jack-voice-bridge` is a local subprocess runtime for `jack-voice` using NDJSON over stdio.

## Protocol

- `stdin`: request messages
- `stdout`: response messages
- `stderr`: logs/progress only

Request envelope:

```json
{"type":"request","id":"req_1","method":"runtime.hello","params":{}}
```

Response envelope:

```json
{"type":"response","id":"req_1","ok":true,"result":{"protocol_version":"1.0.0"}}
```

## Implemented Methods

- `runtime.hello`
- `models.status`
- `models.ensure`
- `vad.detect`
- `stt.transcribe`
- `tts.synthesize`
- `runtime.shutdown`

## Local Smoke Test

From repo root:

```bash
cargo build -p jack-voice-bridge
export LD_LIBRARY_PATH=$PWD/target/debug:${LD_LIBRARY_PATH}
python3 jack-voice-bridge/scripts/smoke_test.py --bridge ./target/debug/jack-voice-bridge
```

## Audio Payload

Input audio fields for `vad.detect` and `stt.transcribe`:

- `audio_b64` (base64 bytes)
- `format`: `pcm_s16le` or `f32le`
- `sample_rate_hz`: currently `16000`
- `channels`: currently `1`

TTS output returns `audio_b64` in `f32le` format plus `sample_rate_hz`.
