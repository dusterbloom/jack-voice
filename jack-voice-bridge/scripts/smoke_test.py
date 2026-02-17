#!/usr/bin/env python3
"""Local smoke test for jack-voice-bridge NDJSON RPC.

Runs runtime/model checks plus VAD/STT/TTS round-trips against a local bridge process.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict


def rpc(stdin, stdout, req_id: int, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    message = {
        "type": "request",
        "id": str(req_id),
        "method": method,
        "params": params,
    }
    stdin.write(json.dumps(message) + "\n")
    stdin.flush()

    line = stdout.readline()
    if not line:
        raise RuntimeError(f"Bridge closed while waiting for {method}")

    response = json.loads(line)
    if response.get("type") != "response":
        raise RuntimeError(f"Unexpected message type for {method}: {response}")
    if response.get("id") != str(req_id):
        raise RuntimeError(f"Mismatched id for {method}: {response}")
    if not response.get("ok", False):
        raise RuntimeError(f"{method} failed: {response.get('error')}")

    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"{method} returned non-object result: {result}")
    return result


def rpc_tts_stream(stdin, stdout, req_id: int, params: Dict[str, Any]) -> tuple[Dict[str, Any], list[Dict[str, Any]]]:
    message = {
        "type": "request",
        "id": str(req_id),
        "method": "tts.stream",
        "params": params,
    }
    stdin.write(json.dumps(message) + "\n")
    stdin.flush()

    events: list[Dict[str, Any]] = []
    while True:
        line = stdout.readline()
        if not line:
            raise RuntimeError("Bridge closed while waiting for tts.stream")

        message = json.loads(line)
        message_type = message.get("type")

        if message_type == "event":
            if message.get("id") != str(req_id):
                raise RuntimeError(f"Mismatched event id for tts.stream: {message}")
            events.append(message)
            continue

        if message_type != "response":
            raise RuntimeError(f"Unexpected message type for tts.stream: {message}")
        if message.get("id") != str(req_id):
            raise RuntimeError(f"Mismatched response id for tts.stream: {message}")
        if not message.get("ok", False):
            raise RuntimeError(f"tts.stream failed: {message.get('error')}")

        result = message.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"tts.stream returned non-object result: {result}")
        return result, events


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bridge",
        default="./target/debug/jack-voice-bridge",
        help="Path to jack-voice-bridge binary",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    bridge_dir = os.path.abspath(os.path.join(os.path.dirname(args.bridge), "."))
    if os.name == "nt":
        loader_key = "PATH"
    elif platform.system() == "Darwin":
        loader_key = "DYLD_LIBRARY_PATH"
    else:
        loader_key = "LD_LIBRARY_PATH"
    current = env.get(loader_key, "")
    env[loader_key] = (
        f"{bridge_dir}{os.pathsep}{current}" if current else bridge_dir
    )

    proc = subprocess.Popen(
        [args.bridge],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
        env=env,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None

    try:
        req_id = 1
        hello = rpc(proc.stdin, proc.stdout, req_id, "runtime.hello", {})
        req_id += 1

        status = rpc(proc.stdin, proc.stdout, req_id, "models.status", {})
        req_id += 1

        silence_b64 = base64.b64encode(b"\x00\x00" * 320).decode("ascii")
        vad = rpc(
            proc.stdin,
            proc.stdout,
            req_id,
            "vad.detect",
            {
                "audio_b64": silence_b64,
                "format": "pcm_s16le",
                "sample_rate_hz": 16000,
                "channels": 1,
            },
        )
        req_id += 1

        stt = rpc(
            proc.stdin,
            proc.stdout,
            req_id,
            "stt.transcribe",
            {
                "audio_b64": base64.b64encode(b"\x00\x00" * 1600).decode("ascii"),
                "format": "pcm_s16le",
                "sample_rate_hz": 16000,
                "channels": 1,
                "language": "auto",
            },
        )
        req_id += 1

        readiness = status.get("readiness", {})
        if isinstance(readiness, dict) and readiness.get("pocket"):
            tts_params = {"text": "Build finished.", "engine": "pocket", "voice": "alba"}
        elif isinstance(readiness, dict) and readiness.get("kokoro"):
            tts_params = {"text": "Build finished.", "engine": "kokoro", "voice": "35"}
        else:
            tts_params = {"text": "Build finished.", "engine": "supertonic", "voice": "F1"}

        tts = rpc(proc.stdin, proc.stdout, req_id, "tts.synthesize", tts_params)
        req_id += 1

        stream_params = dict(tts_params)
        stream_params["text"] = "Build finished with streaming."
        tts_stream, tts_stream_events = rpc_tts_stream(proc.stdin, proc.stdout, req_id, stream_params)
        req_id += 1

        _ = rpc(proc.stdin, proc.stdout, req_id, "runtime.shutdown", {})

        print("protocol_version:", hello.get("protocol_version"))
        print("methods:", len(hello.get("methods", [])))
        print("models_all_ready:", status.get("all_ready"))
        print("vad:", {k: vad.get(k) for k in ("is_speech", "speech_detected")})
        print("stt:", {k: stt.get(k) for k in ("text", "is_final", "backend")})
        print(
            "tts:",
            {k: tts.get(k) for k in ("engine", "voice", "sample_rate_hz", "sample_count", "duration_ms")},
        )
        print("tts_audio_b64_len:", len(tts.get("audio_b64", "")))
        event_names = [evt.get("event") for evt in tts_stream_events]
        print("tts_stream_events:", event_names)
        print(
            "tts_stream:",
            {
                k: tts_stream.get(k)
                for k in (
                    "engine",
                    "voice",
                    "native_streaming",
                    "sample_rate_hz",
                    "sample_count",
                    "duration_ms",
                    "chunk_count",
                )
            },
        )
        return 0
    finally:
        if proc.poll() is None:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2)

        # Print stderr tail for debugging only when bridge exits non-zero.
        if proc.returncode not in (0, None):
            tail = proc.stderr.read()[-4000:]
            if tail:
                print("bridge_stderr_tail:\n" + tail, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
