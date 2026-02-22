//! Integration tests for Qwen3-TTS integration
//!
//! These tests require model downloads and GPU support.
//! Run with: cargo test -p jack-voice qwen_integration
//!
//! For tests requiring model download:
//!   cargo test -p jack-voice qwen_integration -- --ignored

use jack_voice::{models, qwen_tts::QwenModelSize, NoopProgress, TextToSpeech, TtsEngine};
use std::path::PathBuf;

const REFERENCE_AUDIO_PATH: &str = "test_fixtures/carriera_fantozzi.wav";

fn ensure_qwen_lite_model() {
    if !models::qwen_model_ready(QwenModelSize::Lite) {
        println!("[test] Downloading Qwen Lite model...");
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(models::ensure_qwen_model(
            QwenModelSize::Lite,
            &NoopProgress,
        ))
        .expect("Model download failed");
        println!("[test] Qwen Lite model ready");
    }
}

fn ensure_qwen_large_model() {
    if !models::qwen_model_ready(QwenModelSize::Large) {
        println!("[test] Downloading Qwen Large model...");
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(models::ensure_qwen_model(
            QwenModelSize::Large,
            &NoopProgress,
        ))
        .expect("Model download failed");
        println!("[test] Qwen Large model ready");
    }
}

fn reference_audio_exists() -> bool {
    PathBuf::from(REFERENCE_AUDIO_PATH).exists()
}

#[test]
fn qwen_lite_model_dir_exists_after_download() {
    ensure_qwen_lite_model();

    let model_dir = models::qwen_model_dir(QwenModelSize::Lite);
    assert!(
        model_dir.exists(),
        "Model dir should exist: {:?}",
        model_dir
    );
}

#[test]
fn qwen_lite_model_files_exist_after_download() {
    ensure_qwen_lite_model();

    let model_dir = models::qwen_model_dir(QwenModelSize::Lite);
    assert!(
        model_dir.join("model.safetensors").exists(),
        "model.safetensors should exist"
    );
    assert!(
        model_dir.join("config.json").exists(),
        "config.json should exist"
    );
    assert!(
        model_dir
            .join("speech_tokenizer/model.safetensors")
            .exists(),
        "speech_tokenizer/model.safetensors should exist"
    );
}

#[test]
fn qwen_model_ready_returns_true_after_download() {
    ensure_qwen_lite_model();

    assert!(
        models::qwen_model_ready(QwenModelSize::Lite),
        "qwen_model_ready should return true after download"
    );
}

#[test]
fn qwen_lite_engine_creates_successfully() {
    ensure_qwen_lite_model();

    let tts = TextToSpeech::with_engine(TtsEngine::Qwen);
    assert!(tts.is_ok(), "Failed to create Qwen TTS: {:?}", tts.err());
}

#[test]
fn qwen_lite_synthesizes_non_empty_audio() {
    ensure_qwen_lite_model();

    let mut tts = TextToSpeech::with_engine(TtsEngine::Qwen).expect("Failed to create Qwen TTS");

    let audio = tts.synthesize("Hello, world!").expect("Synthesis failed");

    assert!(
        !audio.samples.is_empty(),
        "Audio samples should not be empty"
    );
    assert_eq!(audio.sample_rate, 24000, "Sample rate should be 24000");

    let duration_secs = audio.samples.len() as f32 / audio.sample_rate as f32;
    assert!(
        duration_secs > 0.3,
        "Audio too short for 'Hello, world!': {:.2}s",
        duration_secs
    );
}

#[test]
fn qwen_lite_streaming_produces_audio() {
    ensure_qwen_lite_model();

    let mut tts = TextToSpeech::with_engine(TtsEngine::Qwen).expect("Failed to create Qwen TTS");

    let mut chunks = 0;
    let mut total_samples = 0;

    tts.synthesize_streaming("Testing streaming synthesis", |samples, _sr| {
        chunks += 1;
        total_samples += samples.len();
        true
    })
    .expect("Streaming failed");

    assert!(chunks > 0, "Should have received at least one chunk");
    assert!(total_samples > 0, "Should have audio samples");

    let duration_secs = total_samples as f32 / 24000.0;
    println!(
        "[test] Streaming: {} chunks, {:.2}s audio",
        chunks, duration_secs
    );
}

#[test]
fn qwen_lite_voice_selection_works() {
    ensure_qwen_lite_model();

    let mut tts = TextToSpeech::with_engine(TtsEngine::Qwen).expect("Failed to create Qwen TTS");

    tts.set_speaker("serena").expect("Should accept serena");
    assert_eq!(tts.current_speaker(), "serena");

    tts.set_speaker("aiden").expect("Should accept aiden");
    assert_eq!(tts.current_speaker(), "aiden");

    let audio = tts.synthesize("Voice test").expect("Synthesis failed");
    assert!(!audio.samples.is_empty());
}

#[test]
fn qwen_lite_rejects_invalid_voice() {
    ensure_qwen_lite_model();

    let mut tts = TextToSpeech::with_engine(TtsEngine::Qwen).expect("Failed to create Qwen TTS");

    let result = tts.set_speaker("not_a_real_voice");
    assert!(result.is_err(), "Should reject invalid voice");
}

#[test]
fn qwen_lite_engine_type_is_correct() {
    ensure_qwen_lite_model();

    let tts = TextToSpeech::with_engine(TtsEngine::Qwen).expect("Failed to create Qwen TTS");

    assert_eq!(tts.engine_type(), "qwen");
}

#[test]
fn qwen_lite_sample_rate_is_24khz() {
    ensure_qwen_lite_model();

    let tts = TextToSpeech::with_engine(TtsEngine::Qwen).expect("Failed to create Qwen TTS");

    assert_eq!(tts.sample_rate(), 24000);
}

#[test]
fn qwen_lite_supports_voice_cloning_returns_false() {
    ensure_qwen_lite_model();

    let tts = TextToSpeech::with_engine(TtsEngine::Qwen).expect("Failed to create Qwen TTS");

    assert!(
        !tts.supports_voice_cloning(),
        "Lite model should not support voice cloning"
    );
}

#[test]
fn qwen_lite_synthesizes_italian_text() {
    ensure_qwen_lite_model();

    let mut tts = TextToSpeech::with_engine(TtsEngine::Qwen).expect("Failed to create Qwen TTS");

    let audio = tts
        .synthesize("La corazzata potioschi e' una cagata pazzesca")
        .expect("Italian synthesis failed");

    assert!(!audio.samples.is_empty());

    let duration_secs = audio.samples.len() as f32 / audio.sample_rate as f32;
    assert!(
        duration_secs > 1.0,
        "Italian text should produce >1s audio, got {:.2}s",
        duration_secs
    );
}

#[test]
#[ignore = "Requires Qwen Large model download (~4GB)"]
fn qwen_large_model_downloads_and_loads() {
    ensure_qwen_large_model();

    let model_dir = models::qwen_model_dir(QwenModelSize::Large);
    assert!(model_dir.exists(), "Large model dir should exist");
    assert!(
        model_dir.join("model.safetensors").exists(),
        "model.safetensors should exist"
    );

    let tts = TextToSpeech::with_engine(TtsEngine::QwenLarge);
    assert!(
        tts.is_ok(),
        "Failed to create QwenLarge TTS: {:?}",
        tts.err()
    );
}

#[test]
#[ignore = "Requires Qwen Large model download (~4GB)"]
fn qwen_large_supports_voice_cloning() {
    ensure_qwen_large_model();

    let tts =
        TextToSpeech::with_engine(TtsEngine::QwenLarge).expect("Failed to create QwenLarge TTS");

    assert!(
        tts.supports_voice_cloning(),
        "Large model should support voice cloning"
    );
}

#[test]
#[ignore = "Requires Qwen Large model and reference audio"]
fn qwen_large_voice_cloning_with_reference() {
    if !reference_audio_exists() {
        println!(
            "[test] Skipping: reference audio not found at {}",
            REFERENCE_AUDIO_PATH
        );
        return;
    }

    ensure_qwen_large_model();

    let mut tts =
        TextToSpeech::with_engine(TtsEngine::QwenLarge).expect("Failed to create QwenLarge TTS");

    let ref_path = PathBuf::from(REFERENCE_AUDIO_PATH);
    tts.set_voice_clone_reference(ref_path, None)
        .expect("Failed to set voice clone reference");

    let audio = tts
        .synthesize("La corazzata potioschi e' una cagata pazzesca")
        .expect("Voice clone synthesis failed");

    assert!(
        !audio.samples.is_empty(),
        "Cloned audio should not be empty"
    );

    let duration_secs = audio.samples.len() as f32 / audio.sample_rate as f32;
    println!("[test] Voice clone: {:.2}s audio", duration_secs);
}

#[test]
#[ignore = "Requires Qwen Large model and reference audio"]
fn qwen_large_engine_type_is_correct() {
    ensure_qwen_large_model();

    let tts =
        TextToSpeech::with_engine(TtsEngine::QwenLarge).expect("Failed to create QwenLarge TTS");

    assert_eq!(tts.engine_type(), "qwen-large");
}

#[test]
fn can_run_qwen_checks_gpu() {
    let result = TextToSpeech::can_run_qwen();
    println!("[test] can_run_qwen() = {}", result);
}

#[test]
fn available_qwen_voices_is_not_empty() {
    let voices = TextToSpeech::available_qwen_voices();
    assert!(!voices.is_empty(), "Should have available voices");
    assert_eq!(voices.len(), 9, "Should have 9 preset voices");
}

#[test]
fn qwen_engine_serialization_roundtrip() {
    let engines = vec![TtsEngine::Qwen, TtsEngine::QwenLarge];

    for engine in engines {
        let json = serde_json::to_string(&engine).expect("Should serialize");
        let parsed: TtsEngine = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(engine, parsed, "Roundtrip failed for {:?}", engine);
    }
}
