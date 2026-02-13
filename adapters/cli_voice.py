#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SDK_ROOT = REPO_ROOT / "sdk" / "python" / "jack_voice_sdk"
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))

from jack_voice_sdk import BridgeError, JackVoice  # noqa: E402

DEFAULT_SECONDS = 5.0
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_CHANNELS = 1
DEFAULT_TTS_ENGINE = "kokoro"
DEFAULT_TTS_VOICE = "35"

TARGET_EXECUTABLES: Dict[str, Sequence[str]] = {
    "codex": ("codex",),
    "claude": ("claude", "claude-code"),
    "opencode": ("opencode",),
    "droid": ("droid",),
    "pi": ("pi",),
}


class CaptureError(RuntimeError):
    pass


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "doctor":
            return cmd_doctor(json_output=args.json)
        if args.command == "stt-mic":
            return cmd_stt_mic(args)
        if args.command == "tts":
            return cmd_tts(args)
        if args.command == "ask":
            return cmd_ask(args)
        if args.command == "chat":
            return cmd_chat(args)
        parser.print_help()
        return 2
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except CaptureError as exc:
        print(f"[jack-voice-adapter] capture error: {exc}", file=sys.stderr)
        return 3
    except BridgeError as exc:
        print(f"[jack-voice-adapter] bridge error ({exc.code}): {exc}", file=sys.stderr)
        return 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli_voice.py",
        description="Cross-CLI jack-voice adapter (codex/claude/opencode/droid/pi).",
    )
    sub = parser.add_subparsers(dest="command")

    doctor = sub.add_parser("doctor", help="Check bridge and capture backend status.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    stt = sub.add_parser("stt-mic", help="Capture microphone audio and print transcript.")
    add_common_stt_args(stt)
    stt.add_argument("--json", action="store_true", help="Print full JSON output.")

    tts = sub.add_parser("tts", help="Synthesize text and write raw f32le audio.")
    tts.add_argument("--text", required=True, help="Text to synthesize.")
    tts.add_argument("--engine", default=DEFAULT_TTS_ENGINE, help="TTS engine.")
    tts.add_argument("--voice", default=DEFAULT_TTS_VOICE, help="TTS voice.")
    tts.add_argument(
        "--out",
        default="-",
        help="Output path for decoded f32le bytes. Use '-' to skip file output.",
    )
    tts.add_argument("--json", action="store_true", help="Print full JSON output.")

    ask = sub.add_parser(
        "ask",
        help="Capture mic, transcribe, and launch target CLI with transcript appended.",
    )
    ask.add_argument(
        "--target",
        choices=sorted(TARGET_EXECUTABLES.keys()),
        required=True,
        help="CLI target to launch.",
    )
    add_common_stt_args(ask)
    ask.add_argument(
        "--print-only",
        action="store_true",
        help="Only print transcript, do not launch target CLI.",
    )
    ask.add_argument(
        "--json",
        action="store_true",
        help="Print JSON metadata before launching target CLI.",
    )
    ask.add_argument(
        "--exec",
        dest="override_exec",
        default=None,
        help="Override executable used for launch.",
    )
    ask.add_argument(
        "target_args",
        nargs=argparse.REMAINDER,
        help="Arguments for target CLI. Prefix with `--` to stop adapter parsing.",
    )

    chat = sub.add_parser(
        "chat",
        help="Interactive talk loop: press Enter to capture, then dispatch transcript.",
    )
    chat.add_argument(
        "--target",
        choices=sorted(TARGET_EXECUTABLES.keys()),
        default="codex",
        help="CLI target to launch (default: codex).",
    )
    add_common_stt_args(chat)
    chat.add_argument(
        "--print-only",
        action="store_true",
        help="Only print transcripts, do not launch target CLI.",
    )
    chat.add_argument(
        "--json",
        action="store_true",
        help="Print JSON metadata for each capture turn.",
    )
    chat.add_argument(
        "--exec",
        dest="override_exec",
        default=None,
        help="Override executable used for launch.",
    )
    chat.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Stop after N capture turns (0 means unlimited).",
    )
    chat.add_argument(
        "target_args",
        nargs=argparse.REMAINDER,
        help="Arguments for target CLI. Prefix with `--` to stop adapter parsing.",
    )

    return parser


def add_common_stt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help=f"Capture duration in seconds (default: {DEFAULT_SECONDS}).",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="STT language hint (`auto` by default).",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional capture source/device name (Pulse source, ALSA hw, etc).",
    )


def cmd_doctor(*, json_output: bool) -> int:
    env = auto_audio_env(os.environ.copy())
    capture_backend = detect_capture_backend(env)
    status = {
        "bridge_ready": False,
        "vad_ready": False,
        "stt_ready": False,
        "tts_ready": False,
        "capture_backend": capture_backend,
        "pulse_server": env.get("PULSE_SERVER"),
        "wsl": is_wsl(),
        "targets": {target: resolve_executable_name(target) for target in TARGET_EXECUTABLES},
    }

    client: Optional[JackVoice] = None
    try:
        client = JackVoice.connect(env=env)
        vad_probe = client.vad(
            b"\x00\x00" * 320,
            sample_rate_hz=DEFAULT_SAMPLE_RATE,
            channels=DEFAULT_CHANNELS,
            audio_format="pcm_s16le",
        )
        status["vad_ready"] = True
        status["vad_is_speech"] = vad_probe.get("is_speech")

        probe = client.stt(
            b"\x00\x00" * 3200,
            sample_rate_hz=DEFAULT_SAMPLE_RATE,
            channels=DEFAULT_CHANNELS,
            audio_format="pcm_s16le",
            language="auto",
        )
        status["stt_ready"] = True
        status["stt_backend"] = probe.get("backend")

        tts_probe = client.tts(
            "jack voice online",
            engine=DEFAULT_TTS_ENGINE,
            voice=DEFAULT_TTS_VOICE,
        )
        status["tts_ready"] = True
        status["tts_engine"] = tts_probe.get("engine")
        status["tts_voice"] = tts_probe.get("voice")
        status["tts_audio_b64_len"] = len(str(tts_probe.get("audio_b64") or ""))

        status["bridge_ready"] = bool(status["vad_ready"] and status["stt_ready"] and status["tts_ready"])
    finally:
        if client is not None:
            client.close()

    if json_output:
        print(json.dumps(status, indent=2))
    else:
        print("bridge_ready:", status["bridge_ready"])
        print("vad_ready:", status["vad_ready"])
        print("stt_ready:", status["stt_ready"])
        print("tts_ready:", status["tts_ready"])
        print("capture_backend:", status["capture_backend"])
        print("pulse_server:", status["pulse_server"])
        print("vad_is_speech:", status.get("vad_is_speech"))
        print("stt_backend:", status.get("stt_backend"))
        print("tts_engine:", status.get("tts_engine"))
        print("tts_voice:", status.get("tts_voice"))
        print("tts_audio_b64_len:", status.get("tts_audio_b64_len"))
        print("targets:", status["targets"])
    return 0


def cmd_stt_mic(args: argparse.Namespace) -> int:
    env = auto_audio_env(os.environ.copy())
    audio = capture_audio_pcm16(
        seconds=args.seconds,
        env=env,
        source=args.source,
    )
    response = stt_bytes(audio, language=args.language, env=env)
    text = str(response.get("text") or "").strip()

    if args.json:
        payload = {
            "seconds": args.seconds,
            "bytes": len(audio),
            "backend": response.get("backend"),
            "text": text,
            "is_final": response.get("is_final"),
        }
        print(json.dumps(payload, indent=2))
    else:
        print(text)

    return 0


def cmd_tts(args: argparse.Namespace) -> int:
    env = auto_audio_env(os.environ.copy())
    client = JackVoice.connect(env=env)
    try:
        result = client.tts(args.text, engine=args.engine, voice=args.voice)
    finally:
        client.close()

    audio_b64 = str(result.get("audio_b64") or "")
    decoded = base64.b64decode(audio_b64) if audio_b64 else b""

    if args.out != "-":
        out_path = pathlib.Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(decoded)

    summary = {
        "engine": result.get("engine"),
        "voice": result.get("voice"),
        "sample_rate_hz": result.get("sample_rate_hz"),
        "duration_ms": result.get("duration_ms"),
        "sample_count": result.get("sample_count"),
        "bytes": len(decoded),
        "out": args.out,
    }
    if args.json:
        payload = dict(result)
        payload["audio_b64_len"] = len(audio_b64)
        payload["bytes"] = len(decoded)
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(summary))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    env = auto_audio_env(os.environ.copy())
    text, meta = transcribe_from_mic(
        seconds=args.seconds,
        language=args.language,
        source=args.source,
        env=env,
    )
    meta["target"] = args.target

    if args.json:
        print(json.dumps(meta, indent=2))

    if not text:
        print("[jack-voice-adapter] empty transcript; nothing to send.", file=sys.stderr)
        return 1

    if args.print_only:
        print(text)
        return 0

    return dispatch_transcript(
        target=args.target,
        transcript=text,
        override_exec=args.override_exec,
        target_args=args.target_args,
    )


def cmd_chat(args: argparse.Namespace) -> int:
    env = auto_audio_env(os.environ.copy())
    max_turns = int(args.max_turns)
    turns = 0

    print(
        "[jack-voice-adapter] chat mode: press Enter to record, 'q' to quit.",
        file=sys.stderr,
    )

    while True:
        if max_turns > 0 and turns >= max_turns:
            break

        try:
            line = input("jv-chat> ").strip().lower()
        except EOFError:
            break

        if line in {"q", "quit", "exit"}:
            break
        if line and line not in {"r", "rec", "record"}:
            print("Use Enter to record or 'q' to quit.", file=sys.stderr)
            continue

        text, meta = transcribe_from_mic(
            seconds=args.seconds,
            language=args.language,
            source=args.source,
            env=env,
        )
        meta["target"] = args.target

        if args.json:
            print(json.dumps(meta, indent=2))

        if not text:
            print("[jack-voice-adapter] empty transcript; try again.", file=sys.stderr)
            continue

        print(text)
        turns += 1

        if args.print_only:
            continue

        rc = dispatch_transcript(
            target=args.target,
            transcript=text,
            override_exec=args.override_exec,
            target_args=args.target_args,
        )
        if rc != 0:
            print(f"[jack-voice-adapter] target exited with code {rc}", file=sys.stderr)

    return 0


def transcribe_from_mic(
    *,
    seconds: float,
    language: str,
    source: Optional[str],
    env: Mapping[str, str],
) -> Tuple[str, Dict[str, object]]:
    audio = capture_audio_pcm16(
        seconds=seconds,
        env=env,
        source=source,
    )
    response = stt_bytes(audio, language=language, env=env)
    text = str(response.get("text") or "").strip()
    meta: Dict[str, object] = {
        "bytes": len(audio),
        "backend": response.get("backend"),
        "text": text,
        "is_final": response.get("is_final"),
    }
    return text, meta


def dispatch_transcript(
    *,
    target: str,
    transcript: str,
    override_exec: Optional[str],
    target_args: Sequence[str],
) -> int:
    executable = override_exec or resolve_executable_name(target)
    if not executable:
        print(
            f"[jack-voice-adapter] target CLI '{target}' not found in PATH.",
            file=sys.stderr,
        )
        print(transcript)
        return 2

    passthrough = normalize_target_args(target_args)

    cmd = [executable, *passthrough, transcript]
    print(f"[jack-voice-adapter] launching: {' '.join(cmd[:-1])} '<transcript>'", file=sys.stderr)
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def normalize_target_args(target_args: Sequence[str]) -> List[str]:
    passthrough = list(target_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return passthrough


def stt_bytes(audio_bytes: bytes, *, language: str, env: Mapping[str, str]) -> Dict[str, object]:
    client = JackVoice.connect(env=env)
    try:
        return client.stt(
            audio_bytes,
            sample_rate_hz=DEFAULT_SAMPLE_RATE,
            channels=DEFAULT_CHANNELS,
            audio_format="pcm_s16le",
            language=language,
        )
    finally:
        client.close()


def capture_audio_pcm16(
    *,
    seconds: float,
    env: Mapping[str, str],
    source: Optional[str],
) -> bytes:
    if seconds <= 0:
        raise CaptureError("seconds must be > 0")

    backends = available_capture_backends(env)
    if not backends:
        raise CaptureError(
            "No supported capture backend found. Install one of: parec (Pulse), arecord (ALSA), ffmpeg."
        )

    errors: List[str] = []
    for backend in backends:
        try:
            if backend == "pulse":
                return capture_pulse_pcm16(seconds=seconds, env=env, source=source)
            if backend == "alsa":
                return capture_alsa_pcm16(seconds=seconds, env=env, source=source)
            if backend == "ffmpeg-avfoundation":
                return capture_ffmpeg_avfoundation_pcm16(seconds=seconds, source=source)
            if backend == "ffmpeg-dshow":
                return capture_ffmpeg_dshow_pcm16(seconds=seconds, source=source)
        except CaptureError as exc:
            errors.append(f"{backend}: {exc}")
            continue

    raise CaptureError("; ".join(errors))


def capture_pulse_pcm16(
    *,
    seconds: float,
    env: Mapping[str, str],
    source: Optional[str],
) -> bytes:
    cmd = [
        "parec",
        "--record",
        "--rate",
        str(DEFAULT_SAMPLE_RATE),
        "--channels",
        str(DEFAULT_CHANNELS),
        "--format",
        "s16le",
        "--raw",
    ]
    if source:
        cmd.extend(["--device", source])
    else:
        cmd.extend(["--device", "@DEFAULT_SOURCE@"])

    return timed_capture_stdout(cmd, seconds=seconds, env=env)


def capture_alsa_pcm16(
    *,
    seconds: float,
    env: Mapping[str, str],
    source: Optional[str],
) -> bytes:
    duration = max(1, int(math.ceil(seconds)))
    cmd = [
        "arecord",
        "-q",
        "-f",
        "S16_LE",
        "-c",
        str(DEFAULT_CHANNELS),
        "-r",
        str(DEFAULT_SAMPLE_RATE),
        "-d",
        str(duration),
        "-t",
        "raw",
    ]
    if source:
        cmd.extend(["-D", source])

    completed = subprocess.run(cmd, capture_output=True, env=dict(env), check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore").strip()
        raise CaptureError(f"arecord failed: {stderr or f'code {completed.returncode}'}")
    if not completed.stdout:
        raise CaptureError("arecord returned no audio bytes")
    return completed.stdout


def capture_ffmpeg_avfoundation_pcm16(*, seconds: float, source: Optional[str]) -> bytes:
    input_selector = source or ":0"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "avfoundation",
        "-i",
        input_selector,
        "-t",
        str(seconds),
        "-ac",
        str(DEFAULT_CHANNELS),
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    return capture_with_run(cmd)


def capture_ffmpeg_dshow_pcm16(*, seconds: float, source: Optional[str]) -> bytes:
    input_selector = source or "audio=default"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "dshow",
        "-i",
        input_selector,
        "-t",
        str(seconds),
        "-ac",
        str(DEFAULT_CHANNELS),
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "-f",
        "s16le",
        "-",
    ]
    return capture_with_run(cmd)


def capture_with_run(cmd: Sequence[str]) -> bytes:
    completed = subprocess.run(cmd, capture_output=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="ignore").strip()
        raise CaptureError(f"{cmd[0]} failed: {stderr or f'code {completed.returncode}'}")
    if not completed.stdout:
        raise CaptureError(f"{cmd[0]} returned no audio bytes")
    return completed.stdout


def timed_capture_stdout(
    cmd: Sequence[str],
    *,
    seconds: float,
    env: Mapping[str, str],
) -> bytes:
    proc = subprocess.Popen(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(env),
    )
    try:
        time.sleep(seconds)
        if proc.poll() is None:
            proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate(timeout=1)
    finally:
        if proc.poll() is None:
            proc.kill()

    if not stdout:
        err = stderr.decode("utf-8", errors="ignore").strip()
        raise CaptureError(f"{cmd[0]} returned no audio bytes ({err or 'unknown error'})")
    return stdout


def detect_capture_backend(env: Mapping[str, str]) -> Optional[str]:
    backends = available_capture_backends(env)
    if not backends:
        return None
    return backends[0]


def available_capture_backends(env: Mapping[str, str]) -> List[str]:
    backends: List[str] = []
    if shutil.which("parec") and (env.get("PULSE_SERVER") or pathlib.Path("/mnt/wslg/PulseServer").exists()):
        backends.append("pulse")
    if shutil.which("arecord"):
        backends.append("alsa")
    if shutil.which("ffmpeg"):
        if sys.platform == "darwin":
            backends.append("ffmpeg-avfoundation")
        if os.name == "nt":
            backends.append("ffmpeg-dshow")
    return backends


def auto_audio_env(env: Mapping[str, str]) -> Dict[str, str]:
    merged = {str(k): str(v) for k, v in env.items()}
    pulse_server = merged.get("PULSE_SERVER", "")
    wslg_pulse = pathlib.Path("/mnt/wslg/PulseServer")
    if wslg_pulse.exists() and (not pulse_server or pulse_server.startswith("tcp:")):
        merged["PULSE_SERVER"] = f"unix:{wslg_pulse}"
    return merged


def resolve_executable_name(target: str) -> Optional[str]:
    for candidate in TARGET_EXECUTABLES[target]:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def is_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        text = pathlib.Path("/proc/sys/kernel/osrelease").read_text().lower()
        return "microsoft" in text
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
