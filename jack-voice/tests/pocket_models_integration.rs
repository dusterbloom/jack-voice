use jack_voice::models;
use jack_voice::NoopProgress;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct EnvVarGuard {
    key: &'static str,
    original: Option<String>,
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.original {
            Some(value) => std::env::set_var(self.key, value),
            None => std::env::remove_var(self.key),
        }
    }
}

fn set_env_var(key: &'static str, value: &str) -> EnvVarGuard {
    let original = std::env::var(key).ok();
    std::env::set_var(key, value);
    EnvVarGuard { key, original }
}

fn clear_env_var(key: &'static str) -> EnvVarGuard {
    let original = std::env::var(key).ok();
    std::env::remove_var(key);
    EnvVarGuard { key, original }
}

fn unique_temp_dir(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    let dir = std::env::temp_dir()
        .join("jack-voice-tests")
        .join(format!("{name}-{nanos}"));
    fs::create_dir_all(&dir).expect("failed to create temporary test directory");
    dir
}

fn create_hf_snapshot_file(hf_home: &Path, repo: &str, revision: &str, relative_path: &str) {
    let file_path = hf_home
        .join("hub")
        .join(repo)
        .join("snapshots")
        .join(revision)
        .join(relative_path);
    let parent = file_path
        .parent()
        .expect("snapshot file should always have a parent");
    fs::create_dir_all(parent).expect("failed to create snapshot parent directory");
    fs::write(&file_path, b"test-data").expect("failed to write snapshot file");
}

fn set_isolated_env(root: &Path) -> Vec<EnvVarGuard> {
    let root_str = root.to_string_lossy().to_string();
    vec![
        set_env_var("HF_HOME", &root_str),
        set_env_var("HOME", &root_str),
        set_env_var("USERPROFILE", &root_str),
        set_env_var("LOCALAPPDATA", &root_str),
    ]
}

#[test]
fn pocket_model_ready_returns_false_when_assets_are_missing() {
    let _lock = env_lock().lock().expect("env lock poisoned");
    let root = unique_temp_dir("pocket-ready-missing");
    let _env = set_isolated_env(&root);

    assert!(
        !models::pocket_model_ready(),
        "expected pocket model readiness to be false without snapshots"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn pocket_model_ready_returns_true_when_required_snapshots_exist() {
    let _lock = env_lock().lock().expect("env lock poisoned");
    let root = unique_temp_dir("pocket-ready-present");
    let _env = set_isolated_env(&root);

    create_hf_snapshot_file(
        &root,
        "models--kyutai--pocket-tts",
        "rev-1",
        "tts_b6369a24.safetensors",
    );
    create_hf_snapshot_file(
        &root,
        "models--kyutai--pocket-tts-without-voice-cloning",
        "rev-2",
        "tokenizer.model",
    );
    create_hf_snapshot_file(
        &root,
        "models--kyutai--pocket-tts-without-voice-cloning",
        "rev-2",
        "embeddings/alba.safetensors",
    );

    assert!(
        models::pocket_model_ready(),
        "expected pocket model readiness to be true with required snapshots"
    );

    let _ = fs::remove_dir_all(root);
}

#[test]
fn ensure_pocket_model_is_noop_without_hf_token() {
    let _lock = env_lock().lock().expect("env lock poisoned");
    let root = unique_temp_dir("pocket-ensure-no-token");
    let _env = set_isolated_env(&root);
    let _token = clear_env_var("HF_TOKEN");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");

    let result = runtime.block_on(models::ensure_pocket_model(&NoopProgress));
    assert!(
        result.is_ok(),
        "expected ensure_pocket_model to skip cleanly when HF_TOKEN is missing: {:?}",
        result
    );
    assert!(
        !models::pocket_model_ready(),
        "expected readiness to remain false without downloaded assets"
    );

    let _ = fs::remove_dir_all(root);
}
