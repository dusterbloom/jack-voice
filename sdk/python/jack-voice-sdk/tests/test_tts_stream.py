"""Tests for tts_stream event dispatch using a mock bridge process."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import pathlib
import textwrap
import threading

SDK_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(SDK_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_DIR))

from jack_voice.client import JackVoice, TtsChunkEvent, BridgeClosedError


MOCK_BRIDGE_SCRIPT = textwrap.dedent("""\
    import json
    import sys

    # Read and respond to each request from stdin
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        msg = json.loads(line)
        req_id = msg["id"]
        method = msg["method"]

        if method == "runtime.hello":
            resp = {"type": "response", "id": req_id, "ok": True, "result": {"version": "mock"}}
            sys.stdout.write(json.dumps(resp) + "\\n")
            sys.stdout.flush()

        elif method == "tts.stream":
            text = msg["params"].get("text", "")
            sample_rate = 24000

            # tts.start event
            start_evt = {
                "type": "event", "id": req_id, "event": "tts.start",
                "data": {"engine": "mock", "voice": "test", "format": "f32le", "channels": 1}
            }
            sys.stdout.write(json.dumps(start_evt) + "\\n")
            sys.stdout.flush()

            # Two tts.chunk events with fake audio
            import base64
            for i in range(2):
                fake_audio = b"\\x00" * 16
                chunk_evt = {
                    "type": "event", "id": req_id, "event": "tts.chunk",
                    "data": {
                        "index": i,
                        "audio_b64": base64.b64encode(fake_audio).decode(),
                        "format": "f32le",
                        "sample_rate_hz": sample_rate,
                        "channels": 1,
                        "sample_count": 4,
                    }
                }
                sys.stdout.write(json.dumps(chunk_evt) + "\\n")
                sys.stdout.flush()

            # tts.end event
            end_evt = {
                "type": "event", "id": req_id, "event": "tts.end",
                "data": {
                    "engine": "mock", "voice": "test", "format": "f32le",
                    "sample_rate_hz": sample_rate, "channels": 1,
                    "sample_count": 8, "duration_ms": 0.33, "chunk_count": 2
                }
            }
            sys.stdout.write(json.dumps(end_evt) + "\\n")
            sys.stdout.flush()

            # Final response
            resp = {
                "type": "response", "id": req_id, "ok": True,
                "result": {
                    "streamed": True, "engine": "mock", "voice": "test",
                    "sample_rate_hz": sample_rate, "chunk_count": 2
                }
            }
            sys.stdout.write(json.dumps(resp) + "\\n")
            sys.stdout.flush()

        elif method == "runtime.shutdown":
            resp = {"type": "response", "id": req_id, "ok": True, "result": {}}
            sys.stdout.write(json.dumps(resp) + "\\n")
            sys.stdout.flush()
            break
""")


def _make_mock_client() -> JackVoice:
    """Start a mock bridge process and connect."""
    proc = subprocess.Popen(
        [sys.executable, "-c", MOCK_BRIDGE_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    client = JackVoice(proc, default_timeout=5.0)
    # Send hello
    client._request("runtime.hello", {})
    return client


class TestTtsStream:
    def test_yields_start_chunks_end(self):
        client = _make_mock_client()
        try:
            events = list(client.tts_stream("Hello world"))
            event_names = [e.event for e in events]
            assert event_names == ["tts.start", "tts.chunk", "tts.chunk", "tts.end"]
        finally:
            client.close()

    def test_chunk_audio_bytes(self):
        client = _make_mock_client()
        try:
            chunks = [e for e in client.tts_stream("test") if e.event == "tts.chunk"]
            assert len(chunks) == 2
            for c in chunks:
                assert len(c.audio_bytes) == 16
                assert c.sample_rate_hz == 24000
        finally:
            client.close()

    def test_start_has_engine_voice(self):
        client = _make_mock_client()
        try:
            events = list(client.tts_stream("test"))
            start = events[0]
            assert start.event == "tts.start"
            assert start.data["engine"] == "mock"
            assert start.data["voice"] == "test"
        finally:
            client.close()

    def test_end_has_metadata(self):
        client = _make_mock_client()
        try:
            events = list(client.tts_stream("test"))
            end = events[-1]
            assert end.event == "tts.end"
            assert end.data["chunk_count"] == 2
            assert end.data["sample_count"] == 8
        finally:
            client.close()


class TestTtsChunkEvent:
    def test_audio_bytes_decodes_b64(self):
        raw = b"\x01\x02\x03\x04"
        evt = TtsChunkEvent(event="tts.chunk", data={"audio_b64": base64.b64encode(raw).decode()})
        assert evt.audio_bytes == raw

    def test_audio_bytes_empty_for_non_chunk(self):
        evt = TtsChunkEvent(event="tts.start", data={"engine": "mock"})
        assert evt.audio_bytes == b""

    def test_sample_rate_default(self):
        evt = TtsChunkEvent(event="tts.chunk", data={})
        assert evt.sample_rate_hz == 24000

    def test_sample_rate_from_data(self):
        evt = TtsChunkEvent(event="tts.chunk", data={"sample_rate_hz": 16000})
        assert evt.sample_rate_hz == 16000
