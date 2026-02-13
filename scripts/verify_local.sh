#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export LD_LIBRARY_PATH="${ROOT_DIR}/target/debug:${LD_LIBRARY_PATH:-}"

echo "[1/6] Running jack-voice library tests"
cargo test -p jack-voice --lib

echo "[2/6] Running bridge tests"
cargo test -p jack-voice-bridge

echo "[3/6] Running bridge protocol smoke test"
cargo build -p jack-voice-bridge
python3 jack-voice-bridge/scripts/smoke_test.py --bridge ./target/debug/jack-voice-bridge

echo "[4/6] Running Python SDK smoke test"
PYTHONPATH="${ROOT_DIR}/sdk/python/jack_voice_sdk:${PYTHONPATH:-}" python3 - <<'PY'
from jack_voice_sdk import JackVoice

client = JackVoice.connect()
try:
    frame = b"\x00\x00" * 320
    utterance = b"\x00\x00" * 3200

    vad = client.vad(frame)
    stt = client.stt(utterance, language="auto")
    tts = client.tts("Build finished.", engine="kokoro", voice="35")

    print("vad:", vad)
    print("stt:", stt)
    print(
        "tts:",
        {
            "engine": tts.get("engine"),
            "voice": tts.get("voice"),
            "sample_rate_hz": tts.get("sample_rate_hz"),
            "duration_ms": tts.get("duration_ms"),
            "audio_b64_len": len(tts.get("audio_b64", "")),
        },
    )
finally:
    client.close()
PY

echo "[5/6] Running Node bridge smoke test"
node <<'JS'
const { spawn } = require("node:child_process");
const { createInterface } = require("node:readline");

const env = {
  ...process.env,
  LD_LIBRARY_PATH: `${process.cwd()}/target/debug:${process.env.LD_LIBRARY_PATH ?? ""}`,
};

const child = spawn("./target/debug/jack-voice-bridge", { stdio: ["pipe", "pipe", "pipe"], env });
const rl = createInterface({ input: child.stdout, crlfDelay: Infinity });

let seq = 0;
const pending = new Map();

function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = `req_${++seq}`;
    pending.set(id, { resolve, reject, method });
    child.stdin.write(JSON.stringify({ type: "request", id, method, params }) + "\n");
  });
}

rl.on("line", (line) => {
  let msg;
  try {
    msg = JSON.parse(line);
  } catch {
    return;
  }
  if (msg.type !== "response") return;
  const p = pending.get(msg.id);
  if (!p) return;
  pending.delete(msg.id);
  if (msg.ok) p.resolve(msg.result || {});
  else p.reject(new Error((msg.error && msg.error.message) || `request failed: ${p.method}`));
});

child.stderr.on("data", () => {});

(async () => {
  try {
    const hello = await send("runtime.hello", {});
    const tts = await send("tts.synthesize", { text: "Build finished.", engine: "kokoro", voice: "35" });
    await send("runtime.shutdown", {});
    console.log("node hello:", { protocol_version: hello.protocol_version, methods: hello.methods?.length || 0 });
    console.log("node tts:", {
      engine: tts.engine,
      voice: tts.voice,
      sample_rate_hz: tts.sample_rate_hz,
      duration_ms: tts.duration_ms,
      audio_b64_len: (tts.audio_b64 || "").length,
    });
  } finally {
    child.stdin.end();
  }
})().catch((err) => {
  console.error("node smoke failed:", err);
  if (!child.killed) {
    child.kill("SIGTERM");
  }
  process.exitCode = 1;
});
JS

echo "[6/6] Running adapter doctor check"
python3 adapters/cli_voice.py doctor --json >/tmp/jv-adapter-doctor.json
cat /tmp/jv-adapter-doctor.json

echo "All local checks passed."
