# CLI Adapters

Thin adapters for `codex`, `claude`, `opencode`, `droid`, and `pi` using local `jack-voice`.

## 60-Second Setup

```bash
cargo build -p jack-voice-bridge
python3 scripts/install_cli_adapters.py
jv-doctor
```

On Windows, use `py -3 scripts/install_cli_adapters.py`.

If `jv-doctor` reports `bridge_ready: True`, you are ready.

## Try It Now

Transcribe microphone input:

```bash
jv-stt --seconds 5
```

Synthesize speech to raw f32le output:

```bash
jv-tts --text "Build finished." --engine kokoro --voice 35 --out /tmp/jv.f32
```

Talk loop (press Enter to record, `q` to quit):

```bash
jv-chat --target codex
```

If the target CLI binary is not on `PATH`, test only STT first:

```bash
jv-chat --target codex --print-only
```

## Per-CLI Wrappers

Each wrapper captures mic audio, runs STT, and launches the target CLI with the transcript appended as the final argument.

```bash
codex-voice
claude-voice
opencode-voice
droid-voice
pi-voice
```

Pass target CLI flags after `--`:

```bash
codex-voice -- --model gpt-5
claude-voice -- --dangerously-skip-permissions
```

## Advanced

Linux/macOS users can install via shell too:

```bash
bash scripts/install_cli_adapters.sh
```

Run adapter directly:

```bash
python3 adapters/cli_voice.py ask --target codex --seconds 6 --print-only
```

Useful commands:
- `doctor`: probes VAD, STT, and TTS readiness plus capture backend.
- `chat`: interactive record-dispatch loop for any target CLI.
- `stt-mic`: records mic and prints transcript.
- `tts`: synthesizes text and writes decoded audio bytes.
- `ask`: records mic, transcribes, and launches a target CLI.
