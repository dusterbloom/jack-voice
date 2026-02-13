# jack-voice-sdk-ts

Minimal Node TypeScript wrapper around `jack-voice-bridge` using NDJSON over stdio.

## Install (local workspace)

```bash
npm install ./sdk/ts/jack-voice-sdk-ts
```

## Usage

```ts
import { JackVoice } from "jack-voice-sdk-ts";

const jv = await JackVoice.connect();

const frame16k = Buffer.alloc(3200); // 100ms PCM16 mono 16kHz
const vad = await jv.vad(frame16k);

const utterance16k = Buffer.alloc(32000); // example PCM16 utterance
const stt = await jv.stt(utterance16k, { language: "auto" });

const tts = await jv.tts("Build finished.", { voice: "F1" }); // Supertonic voice
const wavOrPcm = Buffer.from(String(tts.audio_b64), "base64");

await jv.close();
```

The SDK auto-discovers bridge binaries and configures loader paths across Linux/macOS/Windows.

## API

- `JackVoice.connect(options?)`: spawn bridge and send `runtime.hello`.
- `client.vad(audio, options?)`: sends `vad.detect`.
- `client.stt(audio, options?)`: sends `stt.transcribe`.
- `client.tts(text, options?)`: sends `tts.synthesize`.
- `client.close()`: sends `runtime.shutdown` and terminates process.

## Notes

- Audio input accepts `Buffer`, `ArrayBuffer`, or typed arrays.
- SDK correlates responses by request `id` and resolves each call independently.
- Uses built-in Node modules only (`child_process`, `readline`, `events`).
- Discovery order: `JACK_VOICE_BRIDGE_CMD` -> local workspace `target/{debug,release}` -> `PATH`.
- Set `autoConfigure: false` in `connect()` to disable automatic env/path setup.
