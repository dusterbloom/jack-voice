from __future__ import annotations

import base64
import itertools
import json
import os
import pathlib
import queue
import shlex
import shutil
import subprocess
import sys
import threading
from typing import Any, Dict, Mapping, Optional, Sequence, Union

Command = Union[str, Sequence[str], None]
BytesLike = Union[bytes, bytearray, memoryview]


class BridgeError(RuntimeError):
    """Bridge returned an error response."""

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        retryable: Optional[bool] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = details or {}


class BridgeClosedError(BridgeError):
    """Bridge process closed before request completion."""


class JackVoice:
    def __init__(self, process: subprocess.Popen[str], *, default_timeout: float = 30.0) -> None:
        if process.stdin is None or process.stdout is None:
            raise ValueError("Bridge process must be started with stdin/stdout pipes.")

        self._process = process
        self._default_timeout = float(default_timeout)
        self._id_counter = itertools.count(1)
        self._closed = threading.Event()

        self._pending_lock = threading.Lock()
        self._pending: Dict[str, "queue.Queue[object]"] = {}
        self._write_lock = threading.Lock()

        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="jack-voice-sdk-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop,
            name="jack-voice-sdk-stderr",
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread.start()

    @classmethod
    def connect(
        cls,
        command: Command = None,
        *,
        default_timeout: float = 30.0,
        startup_timeout: Optional[float] = None,
        hello_options: Optional[Dict[str, Any]] = None,
        env: Optional[Mapping[str, str]] = None,
        cwd: Optional[str] = None,
        auto_configure: bool = True,
    ) -> "JackVoice":
        cmd = _resolve_command(command, env=env, cwd=cwd)
        proc_env = _build_process_env(
            cmd,
            env=env,
            cwd=cwd,
            auto_configure=auto_configure,
        )

        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=proc_env,
                cwd=cwd,
            )
        except FileNotFoundError as exc:
            attempted = " ".join(cmd)
            raise BridgeError(
                "jack-voice-bridge executable not found. "
                "Build it with `cargo build -p jack-voice-bridge` or set JACK_VOICE_BRIDGE_CMD.",
                code="BRIDGE_NOT_FOUND",
                details={"attempted_command": attempted},
            ) from exc

        client = cls(process, default_timeout=default_timeout)
        try:
            client._request("runtime.hello", hello_options or {}, timeout=startup_timeout)
        except Exception:
            client.close()
            raise
        return client

    def vad(
        self,
        audio: BytesLike,
        *,
        sample_rate_hz: int = 16000,
        channels: int = 1,
        audio_format: str = "pcm_s16le",
        timeout: Optional[float] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = self._audio_params(audio, sample_rate_hz=sample_rate_hz, channels=channels, audio_format=audio_format)
        if options:
            params.update(options)
        return self._request("vad.detect", params, timeout=timeout)

    def stt(
        self,
        audio: BytesLike,
        *,
        sample_rate_hz: int = 16000,
        channels: int = 1,
        audio_format: str = "pcm_s16le",
        language: Optional[str] = None,
        mode: Optional[str] = None,
        timeout: Optional[float] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = self._audio_params(audio, sample_rate_hz=sample_rate_hz, channels=channels, audio_format=audio_format)
        if language is not None:
            params["language"] = language
        if mode is not None:
            params["mode"] = mode
        if options:
            params.update(options)
        return self._request("stt.transcribe", params, timeout=timeout)

    def tts(
        self,
        text: str,
        *,
        voice: Optional[str] = None,
        language: Optional[str] = None,
        engine: Optional[str] = None,
        timeout: Optional[float] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"text": text}
        if voice is not None:
            params["voice"] = voice
        if language is not None:
            params["language"] = language
        if engine is not None:
            params["engine"] = engine
        if options:
            params.update(options)
        return self._request("tts.synthesize", params, timeout=timeout)

    def close(self, *, timeout: float = 2.0) -> None:
        if self._process.poll() is None and not self._closed.is_set():
            try:
                self._request("runtime.shutdown", {}, timeout=timeout)
            except Exception:
                pass

        self._closed.set()

        if self._process.stdin is not None and not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except Exception:
                pass

        if self._process.poll() is None:
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                try:
                    self._process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

        if self._process.stdout is not None and not self._process.stdout.closed:
            try:
                self._process.stdout.close()
            except Exception:
                pass

        if self._process.stderr is not None and not self._process.stderr.closed:
            try:
                self._process.stderr.close()
            except Exception:
                pass

        self._fail_pending(BridgeClosedError("Bridge connection closed.", code="PROCESS_EXITED"))

    def __enter__(self) -> "JackVoice":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        if self._closed.is_set():
            raise BridgeClosedError("Bridge connection is closed.", code="PROCESS_EXITED")

        request_id = f"req_{next(self._id_counter)}"
        reply_queue: "queue.Queue[object]" = queue.Queue(maxsize=1)

        with self._pending_lock:
            self._pending[request_id] = reply_queue

        payload: Dict[str, Any] = {
            "type": "request",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        wait_timeout = self._resolve_timeout(timeout)
        payload["timeout_ms"] = max(1, int(wait_timeout * 1000))

        try:
            line = json.dumps(payload, separators=(",", ":"))
            with self._write_lock:
                if self._process.stdin is None or self._process.stdin.closed:
                    raise BridgeClosedError("Bridge stdin is closed.", code="PROCESS_EXITED")
                self._process.stdin.write(line + "\n")
                self._process.stdin.flush()
        except Exception as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            if isinstance(exc, BridgeError):
                raise
            raise BridgeClosedError("Failed to write request to bridge.", code="WRITE_FAILED") from exc

        try:
            response = reply_queue.get(timeout=wait_timeout)
        except queue.Empty as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            raise TimeoutError(f"Timed out waiting for bridge response to {method}.") from exc

        if isinstance(response, Exception):
            raise response
        if not isinstance(response, dict):
            raise BridgeError("Bridge returned an invalid response.")
        if not response.get("ok", False):
            error = response.get("error") or {}
            raise BridgeError(
                error.get("message", f"Bridge request failed: {method}"),
                code=error.get("code"),
                retryable=error.get("retryable"),
                details=error,
            )

        result = response.get("result")
        if not isinstance(result, dict):
            return {}
        return result

    def _reader_loop(self) -> None:
        stdout = self._process.stdout
        if stdout is None:
            self._closed.set()
            self._fail_pending(BridgeClosedError("Bridge stdout is unavailable.", code="PROCESS_EXITED"))
            return

        try:
            for raw_line in stdout:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(message, dict):
                    continue
                if message.get("type") != "response":
                    continue

                response_id = message.get("id")
                if not isinstance(response_id, str):
                    continue
                with self._pending_lock:
                    reply_queue = self._pending.pop(response_id, None)
                if reply_queue is not None:
                    reply_queue.put(message)
        finally:
            self._closed.set()
            return_code = self._process.poll()
            self._fail_pending(
                BridgeClosedError(
                    f"Bridge process exited with code {return_code}.",
                    code="PROCESS_EXITED",
                )
            )

    def _stderr_loop(self) -> None:
        stderr = self._process.stderr
        if stderr is None:
            return
        for _ in stderr:
            if self._closed.is_set() and self._process.poll() is not None:
                break

    def _fail_pending(self, error: Exception) -> None:
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for reply_queue in pending:
            try:
                reply_queue.put_nowait(error)
            except queue.Full:
                pass

    def _resolve_timeout(self, timeout: Optional[float]) -> float:
        resolved = self._default_timeout if timeout is None else float(timeout)
        if resolved <= 0:
            raise ValueError("timeout must be > 0")
        return resolved

    @staticmethod
    def _audio_params(
        audio: BytesLike,
        *,
        sample_rate_hz: int,
        channels: int,
        audio_format: str,
    ) -> Dict[str, Any]:
        if audio_format not in {"pcm_s16le", "f32le"}:
            raise ValueError("audio_format must be 'pcm_s16le' or 'f32le'")
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if channels <= 0:
            raise ValueError("channels must be > 0")

        audio_bytes = _to_bytes(audio)
        return {
            "audio_b64": base64.b64encode(audio_bytes).decode("ascii"),
            "format": audio_format,
            "sample_rate_hz": sample_rate_hz,
            "channels": channels,
        }


def connect(
    command: Command = None,
    *,
    default_timeout: float = 30.0,
    startup_timeout: Optional[float] = None,
    hello_options: Optional[Dict[str, Any]] = None,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[str] = None,
    auto_configure: bool = True,
) -> JackVoice:
    return JackVoice.connect(
        command=command,
        default_timeout=default_timeout,
        startup_timeout=startup_timeout,
        hello_options=hello_options,
        env=env,
        cwd=cwd,
        auto_configure=auto_configure,
    )


def _resolve_command(
    command: Command,
    *,
    env: Optional[Mapping[str, str]] = None,
    cwd: Optional[str] = None,
) -> Sequence[str]:
    if command is None:
        env_command = _get_env_value("JACK_VOICE_BRIDGE_CMD", env).strip()
        if env_command:
            parsed = shlex.split(env_command)
            if parsed:
                return parsed
        return (_discover_bridge_binary(cwd=cwd),)

    if isinstance(command, str):
        parsed = shlex.split(command)
        if not parsed:
            raise ValueError("command string is empty")
        return parsed

    parsed = [str(part) for part in command]
    if not parsed:
        raise ValueError("command sequence is empty")
    return parsed


def _build_process_env(
    command: Sequence[str],
    *,
    env: Optional[Mapping[str, str]],
    cwd: Optional[str],
    auto_configure: bool,
) -> Dict[str, str]:
    proc_env = {str(k): str(v) for k, v in os.environ.items()}
    if env:
        for key, value in env.items():
            proc_env[str(key)] = str(value)

    if not auto_configure:
        return proc_env

    runtime_dirs = _discover_runtime_dirs(command, cwd=cwd, env=proc_env)
    _prepend_env_path(proc_env, _loader_search_var(), runtime_dirs)

    if _is_wsl() and not proc_env.get("PULSE_SERVER"):
        wslg_pulse = pathlib.Path("/mnt/wslg/PulseServer")
        if wslg_pulse.exists():
            proc_env["PULSE_SERVER"] = f"unix:{wslg_pulse}"

    return proc_env


def _discover_bridge_binary(*, cwd: Optional[str]) -> str:
    binary_name = _bridge_binary_name()

    for candidate in _candidate_bridge_binaries(cwd=cwd):
        if candidate.is_file():
            return str(candidate)

    resolved = shutil.which(binary_name)
    if resolved:
        return resolved

    # Keep subprocess lookup behavior as a final fallback.
    return binary_name


def _discover_runtime_dirs(
    command: Sequence[str],
    *,
    cwd: Optional[str],
    env: Mapping[str, str],
) -> Sequence[str]:
    dirs: list[str] = []

    executable_path = _resolve_executable_path(command[0], cwd=cwd, env=env)
    if executable_path is not None:
        dirs.append(str(executable_path.parent))

    for runtime_dir in _candidate_runtime_dirs(cwd=cwd):
        dirs.append(str(runtime_dir))

    return _dedupe_preserve_order(dirs)


def _resolve_executable_path(
    executable: str,
    *,
    cwd: Optional[str],
    env: Mapping[str, str],
) -> Optional[pathlib.Path]:
    candidate = pathlib.Path(executable)
    if candidate.is_absolute() or candidate.parent != pathlib.Path("."):
        base = pathlib.Path(cwd).resolve() if cwd else pathlib.Path.cwd()
        absolute = candidate if candidate.is_absolute() else (base / candidate)
        if absolute.exists():
            return absolute.resolve()
        return None

    resolved = shutil.which(executable, path=env.get("PATH"))
    if not resolved:
        return None
    return pathlib.Path(resolved).resolve()


def _bridge_binary_name() -> str:
    return "jack-voice-bridge.exe" if os.name == "nt" else "jack-voice-bridge"


def _candidate_bridge_binaries(*, cwd: Optional[str]) -> Sequence[pathlib.Path]:
    binary_name = _bridge_binary_name()
    candidates: list[pathlib.Path] = []

    for root in _candidate_roots(cwd=cwd):
        for rel in (
            ("target", "debug"),
            ("target", "release"),
            ("jack-voice-bridge", "target", "debug"),
            ("jack-voice-bridge", "target", "release"),
        ):
            candidates.append(root.joinpath(*rel, binary_name))

    return _dedupe_paths(candidates)


def _candidate_runtime_dirs(*, cwd: Optional[str]) -> Sequence[pathlib.Path]:
    dirs: list[pathlib.Path] = []

    for root in _candidate_roots(cwd=cwd):
        for rel in (
            ("target", "debug"),
            ("target", "release"),
            ("jack-voice-bridge", "target", "debug"),
            ("jack-voice-bridge", "target", "release"),
        ):
            candidate = root.joinpath(*rel)
            if candidate.is_dir():
                dirs.append(candidate.resolve())

    return _dedupe_paths(dirs)


def _candidate_roots(*, cwd: Optional[str]) -> Sequence[pathlib.Path]:
    roots: list[pathlib.Path] = []

    if cwd:
        roots.extend(_ancestors(pathlib.Path(cwd).resolve(), max_depth=6))

    roots.extend(_ancestors(pathlib.Path.cwd().resolve(), max_depth=6))
    roots.extend(_ancestors(pathlib.Path(__file__).resolve().parent, max_depth=8))

    return _dedupe_paths(roots)


def _ancestors(path: pathlib.Path, *, max_depth: int) -> Sequence[pathlib.Path]:
    values = [path]
    for idx, parent in enumerate(path.parents):
        if idx >= max_depth:
            break
        values.append(parent)
    return values


def _loader_search_var() -> str:
    if os.name == "nt":
        return "PATH"
    if sys.platform == "darwin":
        return "DYLD_LIBRARY_PATH"
    return "LD_LIBRARY_PATH"


def _prepend_env_path(env: Dict[str, str], key: str, entries: Sequence[str]) -> None:
    existing = [item for item in env.get(key, "").split(os.pathsep) if item]
    merged = _dedupe_preserve_order([*entries, *existing])
    if merged:
        env[key] = os.pathsep.join(merged)


def _get_env_value(name: str, env: Optional[Mapping[str, str]]) -> str:
    if env and name in env:
        value = env[name]
        return str(value) if value is not None else ""
    return os.environ.get(name, "")


def _is_wsl() -> bool:
    if sys.platform != "linux":
        return False
    try:
        if "microsoft" in pathlib.Path("/proc/sys/kernel/osrelease").read_text().lower():
            return True
    except OSError:
        pass
    release = getattr(os.uname(), "release", "").lower()
    return "microsoft" in release


def _dedupe_paths(values: Sequence[pathlib.Path]) -> Sequence[pathlib.Path]:
    deduped: list[pathlib.Path] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _dedupe_preserve_order(values: Sequence[str]) -> Sequence[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _to_bytes(audio: BytesLike) -> bytes:
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)
    raise TypeError("audio must be bytes-like (bytes, bytearray, or memoryview)")
