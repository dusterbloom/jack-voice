Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RootDir

function Invoke-Python {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Args
    )

    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 @Args
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python @Args
    }
    else {
        throw "Python is not installed or not on PATH."
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($Args -join ' ')"
    }
}

$BridgeDir = Join-Path $RootDir "target\debug"
$env:PATH = "$BridgeDir;$env:PATH"

Write-Host "[1/6] Running jack-voice library tests"
cargo test -p jack-voice --lib
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/6] Running bridge tests"
cargo test -p jack-voice-bridge
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/6] Running bridge protocol smoke test"
cargo build -p jack-voice-bridge
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$BridgePath = Join-Path $BridgeDir "jack-voice-bridge.exe"
if (-not (Test-Path $BridgePath)) {
    $BridgePath = Join-Path $BridgeDir "jack-voice-bridge"
}
Invoke-Python "jack-voice-bridge/scripts/smoke_test.py" "--bridge" $BridgePath

Write-Host "[4/6] Running Python SDK smoke test"
$pythonSmoke = @'
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
'@
$env:PYTHONPATH = "$RootDir\sdk\python\jack_voice_sdk;$($env:PYTHONPATH)"
Invoke-Python "-c" $pythonSmoke

Write-Host "[5/6] Running Node bridge smoke test"
$nodeSmoke = @'
const { spawn } = require("node:child_process");
const { createInterface } = require("node:readline");

const loaderEnv =
  process.platform === "win32"
    ? {
        PATH: `${process.cwd()}\\\\target\\\\debug;${process.env.PATH ?? ""}`,
      }
    : process.platform === "darwin"
    ? {
        DYLD_LIBRARY_PATH: `${process.cwd()}/target/debug:${process.env.DYLD_LIBRARY_PATH ?? ""}`,
      }
    : {
        LD_LIBRARY_PATH: `${process.cwd()}/target/debug:${process.env.LD_LIBRARY_PATH ?? ""}`,
      };

const env = {
  ...process.env,
  ...loaderEnv,
};

const bridgePath =
  process.platform === "win32"
    ? ".\\\\target\\\\debug\\\\jack-voice-bridge.exe"
    : "./target/debug/jack-voice-bridge";

const child = spawn(bridgePath, { stdio: ["pipe", "pipe", "pipe"], env });
const rl = createInterface({ input: child.stdout, crlfDelay: Infinity });

let seq = 0;
const pending = new Map();

function send(method, params = {}) {
  return new Promise((resolve, reject) => {
    const id = `req_${++seq}`;
    pending.set(id, { resolve, reject, method });
    child.stdin.write(JSON.stringify({ type: "request", id, method, params }) + "\\n");
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
    const tts = await send("tts.synthesize", {
      text: "Build finished.",
      engine: "kokoro",
      voice: "35",
    });
    await send("runtime.shutdown", {});
    console.log("node hello:", {
      protocol_version: hello.protocol_version,
      methods: hello.methods?.length || 0,
    });
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
'@
node -e $nodeSmoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[6/6] Running adapter doctor check"
$doctorPath = Join-Path $env:TEMP "jv-adapter-doctor.json"
Invoke-Python "adapters/cli_voice.py" "doctor" "--json" | Set-Content -Path $doctorPath -Encoding utf8
Get-Content $doctorPath

Write-Host "All local checks passed."
