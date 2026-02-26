# jack-voice-sdk (Python)

Minimal standard-library SDK for the `jack-voice-bridge` subprocess protocol.

## What it does

- Starts the bridge process (`JackVoice.connect()`).
- Sends NDJSON requests on stdin.
- Correlates responses by request `id`.
- Exposes one-line methods: `vad`, `stt`, `tts`.
- Shuts down the bridge with `close()`.
- Auto-discovers local `jack-voice-bridge` binaries and configures loader paths across Linux/macOS/Windows.

## Install (editable)

```bash
cd sdk/python/jack_voice_sdk
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

## Example

```python
from jack_voice_sdk import JackVoice
client = JackVoice.connect()
try:
    frame_16k_pcm = b"\x00\x00" * 320
    utterance_16k_pcm = b"\x00\x00" * 3200

    vad_result = client.vad(frame_16k_pcm)
    stt_result = client.stt(utterance_16k_pcm, language="auto")
    tts_result = client.tts("Build finished.", engine="kokoro", voice="35")

    print(vad_result)
    print(stt_result)
    print(tts_result.keys())
finally:
    client.close()
```

You can also set `JACK_VOICE_BRIDGE_CMD` instead of passing `command=...`.

## Runtime Discovery Order

1. `JACK_VOICE_BRIDGE_CMD` (if set)
2. Local workspace builds (`target/debug`, `target/release`)
3. `jack-voice-bridge` from `PATH`

Set `auto_configure=False` in `connect()` to disable automatic env/path setup.

## Run Example Without Installing

```bash
PYTHONPATH=sdk/python/jack_voice_sdk python3 sdk/python/jack_voice_sdk/examples/basic.py
```
