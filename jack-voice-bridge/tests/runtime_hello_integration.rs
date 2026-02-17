use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

struct BridgeHarness {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl BridgeHarness {
    fn spawn() -> Self {
        let bridge_path = resolve_bridge_path();

        let mut child = Command::new(bridge_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn bridge process");

        let stdin = child.stdin.take().expect("missing child stdin");
        let stdout = child.stdout.take().expect("missing child stdout");

        Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
        }
    }

    fn request(&mut self, payload: Value) -> Value {
        let encoded = serde_json::to_string(&payload).expect("request should serialize");
        writeln!(self.stdin, "{encoded}").expect("failed to write request");
        self.stdin.flush().expect("failed to flush request");

        let mut line = String::new();
        let bytes_read = self
            .stdout
            .read_line(&mut line)
            .expect("failed to read response");
        assert!(bytes_read > 0, "bridge closed stdout unexpectedly");

        serde_json::from_str(line.trim()).expect("response should be valid json")
    }

    fn rpc_ok(&mut self, id: &str, method: &str, params: Value) -> Value {
        let response = self.request(json!({
            "type": "request",
            "id": id,
            "method": method,
            "params": params
        }));

        assert_eq!(
            response.get("type").and_then(Value::as_str),
            Some("response"),
            "unexpected response type: {response}"
        );
        assert_eq!(
            response.get("id").and_then(Value::as_str),
            Some(id),
            "unexpected response id: {response}"
        );
        assert_eq!(
            response.get("ok").and_then(Value::as_bool),
            Some(true),
            "expected successful response: {response}"
        );

        response
    }

    fn rpc_err(&mut self, id: &str, method: &str, params: Value) -> Value {
        let response = self.request(json!({
            "type": "request",
            "id": id,
            "method": method,
            "params": params
        }));

        assert_eq!(
            response.get("type").and_then(Value::as_str),
            Some("response"),
            "unexpected response type: {response}"
        );
        assert_eq!(
            response.get("id").and_then(Value::as_str),
            Some(id),
            "unexpected response id: {response}"
        );
        assert_eq!(
            response.get("ok").and_then(Value::as_bool),
            Some(false),
            "expected error response: {response}"
        );

        response
    }

    fn shutdown(mut self) {
        let _ = self.request(json!({
            "type": "request",
            "id": "shutdown",
            "method": "runtime.shutdown",
            "params": {}
        }));

        let status = self.child.wait().expect("failed to wait for bridge");
        assert!(status.success(), "bridge exited with status: {status}");
    }
}

fn resolve_bridge_path() -> String {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_jack-voice-bridge") {
        return path;
    }
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_jack_voice_bridge") {
        return path;
    }

    let test_bin = std::env::current_exe().expect("failed to resolve current test executable");
    let target_debug_dir = test_bin
        .parent()
        .and_then(|p| p.parent())
        .expect("failed to resolve target/debug directory");

    let bridge_path = target_debug_dir.join("jack-voice-bridge");
    if bridge_path.exists() {
        return bridge_path.to_string_lossy().to_string();
    }

    let bridge_path_exe = target_debug_dir.join("jack-voice-bridge.exe");
    if bridge_path_exe.exists() {
        return bridge_path_exe.to_string_lossy().to_string();
    }

    panic!(
        "could not locate bridge binary; checked env vars and {}",
        bridge_path.display()
    );
}

#[test]
fn runtime_hello_reports_pocket_readiness() {
    let mut bridge = BridgeHarness::spawn();

    let response = bridge.rpc_ok("1", "runtime.hello", json!({}));
    let readiness = response["result"]["models"]["readiness"]
        .as_object()
        .expect("runtime.hello should include models.readiness object");

    assert!(
        readiness.contains_key("pocket"),
        "models.readiness is missing pocket key: {readiness:?}"
    );
    assert!(
        readiness["pocket"].is_boolean(),
        "models.readiness.pocket should be a boolean"
    );
    let methods = response["result"]["methods"]
        .as_array()
        .expect("runtime.hello should include methods array");
    assert!(
        methods.iter().any(|v| v.as_str() == Some("tts.stream")),
        "runtime.hello methods should advertise tts.stream: {methods:?}"
    );

    bridge.shutdown();
}

#[test]
fn models_status_reports_pocket_readiness() {
    let mut bridge = BridgeHarness::spawn();

    let response = bridge.rpc_ok("1", "models.status", json!({}));
    let readiness = response["result"]["readiness"]
        .as_object()
        .expect("models.status should include readiness object");
    let missing = response["result"]["missing"]
        .as_array()
        .expect("models.status should include missing array");

    assert!(
        readiness.contains_key("pocket"),
        "models.status readiness is missing pocket key: {readiness:?}"
    );
    assert!(
        readiness["pocket"].is_boolean(),
        "models.status readiness.pocket should be a boolean"
    );
    assert!(
        missing.iter().all(Value::is_string),
        "models.status missing entries should all be strings: {missing:?}"
    );

    bridge.shutdown();
}

#[test]
fn tts_stream_rejects_unsupported_format() {
    let mut bridge = BridgeHarness::spawn();

    let response = bridge.rpc_err(
        "1",
        "tts.stream",
        json!({
            "text": "hello",
            "format": "wav"
        }),
    );

    let error = response["error"]
        .as_object()
        .expect("tts.stream error response should include error object");
    assert_eq!(
        error.get("code").and_then(Value::as_str),
        Some("UNSUPPORTED_AUDIO_FORMAT"),
        "unexpected tts.stream error: {response}"
    );

    bridge.shutdown();
}
