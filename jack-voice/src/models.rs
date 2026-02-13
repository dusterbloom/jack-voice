// Jack Voice - Model Management
// Download and manage ML models for voice pipeline
// Refactored from jack-desktop to remove Tauri dependency

use std::fs;
use std::io::{Read, Write as IoWrite};
use std::path::PathBuf;
use std::sync::RwLock;

/// Callback trait for model download progress reporting
pub trait ModelProgressCallback: Send + Sync {
    fn on_download_start(&self, model: &str, size_mb: u64);
    fn on_download_progress(&self, model: &str, progress_percent: u32, downloaded_mb: u64);
    fn on_download_complete(&self, model: &str);
    fn on_extracting(&self, model: &str);
}

/// No-op progress callback (silent downloads)
pub struct NoopProgress;
impl ModelProgressCallback for NoopProgress {
    fn on_download_start(&self, _model: &str, _size_mb: u64) {}
    fn on_download_progress(&self, _model: &str, _progress: u32, _downloaded_mb: u64) {}
    fn on_download_complete(&self, _model: &str) {}
    fn on_extracting(&self, _model: &str) {}
}

/// Logging progress callback (prints to log)
pub struct LogProgress;
impl ModelProgressCallback for LogProgress {
    fn on_download_start(&self, model: &str, size_mb: u64) {
        log::info!("[MODELS] Downloading: {} ({}MB)", model, size_mb);
    }
    fn on_download_progress(&self, model: &str, progress: u32, downloaded_mb: u64) {
        log::info!("[MODELS] {}: {}% ({}MB)", model, progress, downloaded_mb);
    }
    fn on_download_complete(&self, model: &str) {
        log::info!("[MODELS] Complete: {}", model);
    }
    fn on_extracting(&self, model: &str) {
        log::info!("[MODELS] Extracting: {}", model);
    }
}

/// Configurable models directory
static MODELS_DIR_OVERRIDE: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Set a custom models directory (call before any model operations)
pub fn set_models_dir(dir: PathBuf) {
    *MODELS_DIR_OVERRIDE.write().unwrap() = Some(dir);
}

/// Model bundle definitions with download URLs
#[derive(Clone)]
pub struct ModelBundle {
    pub name: &'static str,
    pub url: &'static str,
    pub extract_dir: &'static str,
    pub size_mb: u64,
}

/// Available model bundles from sherpa-onnx releases
pub const MODEL_BUNDLES: &[ModelBundle] = &[
    ModelBundle {
        name: "silero_vad.onnx",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
        extract_dir: "",
        size_mb: 2,
    },
    ModelBundle {
        name: "sherpa-onnx-whisper-base.en",
        url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-base.en.tar.bz2",
        extract_dir: "sherpa-onnx-whisper-base.en",
        size_mb: 152,
    },
    ModelBundle {
        name: "smart-turn-v3.2-cpu.onnx",
        url: "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx",
        extract_dir: "",
        size_mb: 9,
    },
];

pub const PARAFORMER_MODEL: ModelBundle = ModelBundle {
    name: "sherpa-onnx-paraformer-en-2024-07-17",
    url: "https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-en-2024-07-17/resolve/main/paraformer-en-2024-07-17.tar.bz2",
    extract_dir: "sherpa-onnx-paraformer-en-2024-07-17",
    size_mb: 250,
};

pub const MOONSHINE_MODEL: ModelBundle = ModelBundle {
    name: "sherpa-onnx-moonshine-base-en-int8",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-base-en-int8.tar.bz2",
    extract_dir: "sherpa-onnx-moonshine-base-en-int8",
    size_mb: 274,
};

pub const WHISPER_TURBO_MODEL: ModelBundle = ModelBundle {
    name: "sherpa-onnx-whisper-turbo",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2",
    extract_dir: "sherpa-onnx-whisper-turbo",
    size_mb: 1500,
};

pub const PARAKEET_EOU_MODEL: ModelBundle = ModelBundle {
    name: "parakeet-eou",
    url:
        "https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1/resolve/main/onnx/encoder.onnx",
    extract_dir: "parakeet-eou",
    size_mb: 120,
};

pub const PARAKEET_TDT_MODEL: ModelBundle = ModelBundle {
    name: "parakeet-tdt",
    url: "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/resolve/main/onnx/model.onnx",
    extract_dir: "parakeet-tdt",
    size_mb: 600,
};

/// Supertonic-2: multilingual TTS (en, ko, es, pt, fr)
pub const SUPERTONIC_MODELS: &[ModelBundle] = &[
    ModelBundle {
        name: "supertonic-text_encoder.onnx",
        url: "https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/text_encoder.onnx",
        extract_dir: "supertonic",
        size_mb: 27,
    },
    ModelBundle {
        name: "supertonic-duration_predictor.onnx",
        url: "https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/duration_predictor.onnx",
        extract_dir: "supertonic",
        size_mb: 2,
    },
    ModelBundle {
        name: "supertonic-vector_estimator.onnx",
        url: "https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/vector_estimator.onnx",
        extract_dir: "supertonic",
        size_mb: 132,
    },
    ModelBundle {
        name: "supertonic-vocoder.onnx",
        url: "https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/vocoder.onnx",
        extract_dir: "supertonic",
        size_mb: 101,
    },
];

pub const SUPERTONIC_VOICES: &[(&str, &str)] = &[
    (
        "F1",
        "https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/F1.json",
    ),
    (
        "F2",
        "https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/F2.json",
    ),
    (
        "F3",
        "https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/F3.json",
    ),
    (
        "F4",
        "https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/F4.json",
    ),
    (
        "F5",
        "https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/F5.json",
    ),
    (
        "M1",
        "https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/M1.json",
    ),
    (
        "M2",
        "https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/M2.json",
    ),
    (
        "M3",
        "https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/M3.json",
    ),
    (
        "M4",
        "https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/M4.json",
    ),
    (
        "M5",
        "https://huggingface.co/Supertone/supertonic-2/resolve/main/voice_styles/M5.json",
    ),
];

pub const SUPERTONIC_CONFIG_URL: &str =
    "https://huggingface.co/Supertone/supertonic-2/resolve/main/onnx/unicode_indexer.json";

pub const KOKORO_MODEL: ModelBundle = ModelBundle {
    name: "kokoro-multi-lang-v1_0",
    url: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2",
    extract_dir: "kokoro-multi-lang-v1_0",
    size_mb: 310,
};

/// Get the models directory
pub fn get_models_dir() -> Result<PathBuf, ModelError> {
    // Check for override first
    if let Some(dir) = MODELS_DIR_OVERRIDE.read().unwrap().as_ref() {
        if !dir.exists() {
            fs::create_dir_all(dir).map_err(|e| ModelError::IoError(e.to_string()))?;
        }
        return Ok(dir.clone());
    }

    let data_dir = dirs::data_dir()
        .ok_or_else(|| ModelError::PathError("Could not find data directory".to_string()))?;

    let models_dir = data_dir.join("jack-voice").join("models");

    if !models_dir.exists() {
        fs::create_dir_all(&models_dir).map_err(|e| ModelError::IoError(e.to_string()))?;
    }

    Ok(models_dir)
}

/// Get path to a specific model file
pub fn get_model_path(model_name: &str) -> Result<PathBuf, ModelError> {
    let models_dir = get_models_dir()?;
    Ok(models_dir.join(model_name))
}

/// Check if a model or directory exists
pub fn model_exists(name: &str) -> bool {
    if let Ok(path) = get_model_path(name) {
        path.exists()
    } else {
        false
    }
}

/// Check if all required models are downloaded
pub fn all_models_ready() -> bool {
    MODEL_BUNDLES.iter().all(|bundle| {
        if bundle.extract_dir.is_empty() {
            model_exists(bundle.name)
        } else {
            model_exists(bundle.extract_dir)
        }
    })
}

pub fn moonshine_model_ready() -> bool {
    model_exists(MOONSHINE_MODEL.extract_dir)
}

pub fn whisper_turbo_model_ready() -> bool {
    model_exists(WHISPER_TURBO_MODEL.extract_dir)
}

pub fn kokoro_model_ready() -> bool {
    model_exists(KOKORO_MODEL.extract_dir)
}

pub fn paraformer_model_ready() -> bool {
    model_exists(PARAFORMER_MODEL.name)
}

pub fn parakeet_eou_ready() -> bool {
    model_exists(PARAKEET_EOU_MODEL.extract_dir)
}

pub fn parakeet_tdt_ready() -> bool {
    model_exists(PARAKEET_TDT_MODEL.extract_dir)
}

// ============================================
// Model path structures
// ============================================

pub struct ParaformerPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joiner: PathBuf,
    pub tokens: PathBuf,
}

pub fn get_paraformer_paths() -> Result<ParaformerPaths, ModelError> {
    let base = get_model_path("sherpa-onnx-paraformer-en-2024-07-17")?;
    if !base.exists() {
        return Err(ModelError::ModelNotFound(
            "Paraformer model not downloaded".to_string(),
        ));
    }

    let encoder = base.join("paraformer-en-encoder.int8.onnx");
    let decoder = base.join("paraformer-en-decoder.int8.onnx");
    let joiner = base.join("paraformer-en-joiner.int8.onnx");

    if encoder.exists() && decoder.exists() && joiner.exists() {
        log::info!("Using Paraformer streaming model (int8 quantized)");
        Ok(ParaformerPaths {
            encoder,
            decoder,
            joiner,
            tokens: base.join("paraformer-en-tokens.txt"),
        })
    } else {
        log::info!("Using Paraformer streaming model (fp32)");
        Ok(ParaformerPaths {
            encoder: base.join("paraformer-en-encoder.onnx"),
            decoder: base.join("paraformer-en-decoder.onnx"),
            joiner: base.join("paraformer-en-joiner.onnx"),
            tokens: base.join("paraformer-en-tokens.txt"),
        })
    }
}

pub struct ParakeetEouPaths {
    pub model_dir: PathBuf,
}

pub struct ParakeetTdtPaths {
    pub model_dir: PathBuf,
}

pub fn get_parakeet_eou_paths() -> Result<ParakeetEouPaths, ModelError> {
    let dir = get_model_path(PARAKEET_EOU_MODEL.extract_dir)?;
    if !dir.exists() {
        return Err(ModelError::ModelNotFound(
            "Parakeet EOU model not downloaded".to_string(),
        ));
    }
    Ok(ParakeetEouPaths { model_dir: dir })
}

pub fn get_parakeet_tdt_paths() -> Result<ParakeetTdtPaths, ModelError> {
    let dir = get_model_path(PARAKEET_TDT_MODEL.extract_dir)?;
    if !dir.exists() {
        return Err(ModelError::ModelNotFound(
            "Parakeet TDT model not downloaded".to_string(),
        ));
    }
    Ok(ParakeetTdtPaths { model_dir: dir })
}

pub struct WhisperPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub tokens: PathBuf,
}

pub fn get_whisper_paths() -> Result<WhisperPaths, ModelError> {
    let base = get_model_path("sherpa-onnx-whisper-base.en")?;
    if !base.exists() {
        return Err(ModelError::ModelNotFound(
            "Whisper model not downloaded".to_string(),
        ));
    }

    let encoder = base.join("base.en-encoder.int8.onnx");
    let decoder = base.join("base.en-decoder.int8.onnx");

    if encoder.exists() && decoder.exists() {
        log::info!("Using Whisper Small model (English, int8 quantized)");
        Ok(WhisperPaths {
            encoder,
            decoder,
            tokens: base.join("base.en-tokens.txt"),
        })
    } else {
        log::info!("Using Whisper Small model (English, fp32)");
        Ok(WhisperPaths {
            encoder: base.join("base.en-encoder.onnx"),
            decoder: base.join("base.en-decoder.onnx"),
            tokens: base.join("base.en-tokens.txt"),
        })
    }
}

pub fn get_whisper_turbo_paths() -> Result<WhisperPaths, ModelError> {
    let base = get_model_path("sherpa-onnx-whisper-turbo")?;
    if !base.exists() {
        return Err(ModelError::ModelNotFound(
            "Whisper Turbo model not downloaded".to_string(),
        ));
    }

    let encoder = base.join("turbo-encoder.int8.onnx");
    let decoder = base.join("turbo-decoder.int8.onnx");

    if encoder.exists() && decoder.exists() {
        log::info!("Using Whisper Turbo model (multilingual, int8 quantized)");
        Ok(WhisperPaths {
            encoder,
            decoder,
            tokens: base.join("turbo-tokens.txt"),
        })
    } else {
        log::info!("Using Whisper Turbo model (multilingual, fp32)");
        Ok(WhisperPaths {
            encoder: base.join("turbo-encoder.onnx"),
            decoder: base.join("turbo-decoder.onnx"),
            tokens: base.join("turbo-tokens.txt"),
        })
    }
}

pub struct MoonshinePaths {
    pub preprocessor: PathBuf,
    pub encoder: PathBuf,
    pub cached_decoder: PathBuf,
    pub uncached_decoder: PathBuf,
    pub tokens: PathBuf,
}

pub fn get_moonshine_paths() -> Result<MoonshinePaths, ModelError> {
    let base = get_model_path(MOONSHINE_MODEL.extract_dir)?;
    if !base.exists() {
        return Err(ModelError::ModelNotFound(
            "Moonshine model not downloaded".to_string(),
        ));
    }

    let preprocessor = base.join("preprocess.onnx");
    let encoder = base.join("encode.int8.onnx");
    let cached_decoder = base.join("cached_decode.int8.onnx");
    let uncached_decoder = base.join("uncached_decode.int8.onnx");
    let tokens = base.join("tokens.txt");

    if preprocessor.exists() && encoder.exists() {
        log::info!("Found Moonshine model files with .int8.onnx naming");
        return Ok(MoonshinePaths {
            preprocessor,
            encoder,
            cached_decoder,
            uncached_decoder,
            tokens,
        });
    }

    let fp32_preprocessor = base.join("preprocess.onnx");
    let fp32_encoder = base.join("encode.onnx");
    let fp32_cached = base.join("cached_decode.onnx");
    let fp32_uncached = base.join("uncached_decode.onnx");

    if fp32_preprocessor.exists() && fp32_encoder.exists() {
        log::info!("Found Moonshine model files with fp32 naming");
        return Ok(MoonshinePaths {
            preprocessor: fp32_preprocessor,
            encoder: fp32_encoder,
            cached_decoder: fp32_cached,
            uncached_decoder: fp32_uncached,
            tokens,
        });
    }

    log::error!("Moonshine model files not found in {:?}", base);
    if let Ok(entries) = std::fs::read_dir(base) {
        let files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .collect();
        log::error!("Available files: {:?}", files);
    }

    Err(ModelError::ModelNotFound(
        "Moonshine model files not found".to_string(),
    ))
}

pub struct SupertonicPaths {
    pub model_dir: PathBuf,
    pub encoder: PathBuf,
    pub duration_predictor: PathBuf,
    pub decoder: PathBuf,
    pub vocoder: PathBuf,
    pub voices_dir: PathBuf,
    pub unicode_indexer: PathBuf,
}

impl SupertonicPaths {
    pub fn all_exist(&self) -> bool {
        self.encoder.exists()
            && self.duration_predictor.exists()
            && self.decoder.exists()
            && self.vocoder.exists()
            && self.unicode_indexer.exists()
    }

    pub fn voice_path(&self, voice_id: &str) -> PathBuf {
        self.voices_dir.join(format!("{}.json", voice_id))
    }
}

pub fn get_supertonic_paths() -> Result<SupertonicPaths, ModelError> {
    let base = get_model_path("supertonic")?;
    if !base.exists() {
        return Err(ModelError::ModelNotFound(
            "Supertonic TTS model not downloaded".to_string(),
        ));
    }
    Ok(SupertonicPaths {
        model_dir: base.clone(),
        encoder: base.join("text_encoder.onnx"),
        duration_predictor: base.join("duration_predictor.onnx"),
        decoder: base.join("vector_estimator.onnx"),
        vocoder: base.join("vocoder.onnx"),
        voices_dir: base.join("voices"),
        unicode_indexer: base.join("unicode_indexer.json"),
    })
}

pub struct KokoroModelPaths {
    pub model: PathBuf,
    pub voices: PathBuf,
    pub tokens: PathBuf,
    pub data_dir: PathBuf,
    pub dict_dir: PathBuf,
    pub lexicon: PathBuf,
}

pub fn get_kokoro_paths_for_language(language: &str) -> Result<KokoroModelPaths, ModelError> {
    let base = get_model_path(KOKORO_MODEL.extract_dir)?;
    if !base.exists() {
        return Err(ModelError::ModelNotFound(
            "Kokoro TTS model not downloaded".to_string(),
        ));
    }

    let lexicon_file = match language {
        "en-us" => "lexicon-us-en.txt",
        "en-gb" => "lexicon-gb-en.txt",
        "zh" => "lexicon-zh.txt",
        _ => "",
    };

    Ok(KokoroModelPaths {
        model: base.join("model.onnx"),
        voices: base.join("voices.bin"),
        tokens: base.join("tokens.txt"),
        data_dir: base.join("espeak-ng-data"),
        dict_dir: base.join("dict"),
        lexicon: if lexicon_file.is_empty() {
            PathBuf::new()
        } else {
            base.join(lexicon_file)
        },
    })
}

pub fn get_kokoro_paths() -> Result<KokoroModelPaths, ModelError> {
    get_kokoro_paths_for_language("en-us")
}

// ============================================
// Model download functions (Tauri-free)
// ============================================

/// Ensure all required models are downloaded
pub async fn ensure_models(progress: &dyn ModelProgressCallback) -> Result<(), ModelError> {
    println!(
        "[MODELS] ensure_models called, checking {} bundles",
        MODEL_BUNDLES.len()
    );

    for bundle in MODEL_BUNDLES {
        let target_name = if bundle.extract_dir.is_empty() {
            bundle.name
        } else {
            bundle.extract_dir
        };

        if !model_exists(target_name) {
            log::info!("Downloading model: {}", bundle.name);
            progress.on_download_start(bundle.name, bundle.size_mb);
            download_model(bundle, progress).await?;
            progress.on_download_complete(bundle.name);
        }
    }

    ensure_supertonic_models(progress).await?;
    ensure_kokoro_model(progress).await?;
    ensure_moonshine_model(progress).await?;
    ensure_parakeet_models(progress).await?;

    println!("[MODELS] All models ready!");
    Ok(())
}

pub async fn ensure_supertonic_models(
    progress: &dyn ModelProgressCallback,
) -> Result<(), ModelError> {
    let models_dir = get_models_dir()?;
    let supertonic_dir = models_dir.join("supertonic");
    let voices_dir = supertonic_dir.join("voices");

    if !supertonic_dir.exists() {
        fs::create_dir_all(&supertonic_dir).map_err(|e| ModelError::IoError(e.to_string()))?;
    }
    if !voices_dir.exists() {
        fs::create_dir_all(&voices_dir).map_err(|e| ModelError::IoError(e.to_string()))?;
    }

    for bundle in SUPERTONIC_MODELS {
        let target_name = bundle
            .name
            .strip_prefix("supertonic-")
            .unwrap_or(bundle.name);
        let target_path = supertonic_dir.join(target_name);

        if !target_path.exists() {
            progress.on_download_start(bundle.name, bundle.size_mb);
            download_file(bundle.url, &target_path, progress).await?;
            progress.on_download_complete(bundle.name);
        }
    }

    for (voice_id, url) in SUPERTONIC_VOICES {
        let voice_path = voices_dir.join(format!("{}.json", voice_id));
        if !voice_path.exists() {
            download_file(url, &voice_path, progress).await?;
        }
    }

    let indexer_path = supertonic_dir.join("unicode_indexer.json");
    if !indexer_path.exists() {
        download_file(SUPERTONIC_CONFIG_URL, &indexer_path, progress).await?;
    }

    Ok(())
}

pub async fn ensure_kokoro_model(progress: &dyn ModelProgressCallback) -> Result<(), ModelError> {
    if model_exists(KOKORO_MODEL.extract_dir) {
        return Ok(());
    }
    progress.on_download_start(KOKORO_MODEL.name, KOKORO_MODEL.size_mb);
    download_model(&KOKORO_MODEL, progress).await?;
    progress.on_download_complete(KOKORO_MODEL.name);
    Ok(())
}

pub async fn ensure_moonshine_model(
    progress: &dyn ModelProgressCallback,
) -> Result<(), ModelError> {
    if model_exists(MOONSHINE_MODEL.extract_dir) {
        return Ok(());
    }
    progress.on_download_start(MOONSHINE_MODEL.name, MOONSHINE_MODEL.size_mb);
    download_model(&MOONSHINE_MODEL, progress).await?;
    progress.on_download_complete(MOONSHINE_MODEL.name);
    Ok(())
}

pub async fn ensure_whisper_turbo_model(
    progress: &dyn ModelProgressCallback,
) -> Result<(), ModelError> {
    if model_exists(WHISPER_TURBO_MODEL.extract_dir) {
        return Ok(());
    }
    progress.on_download_start(WHISPER_TURBO_MODEL.name, WHISPER_TURBO_MODEL.size_mb);
    download_model(&WHISPER_TURBO_MODEL, progress).await?;
    progress.on_download_complete(WHISPER_TURBO_MODEL.name);
    Ok(())
}

pub async fn ensure_parakeet_models(
    progress: &dyn ModelProgressCallback,
) -> Result<(), ModelError> {
    let models_dir = get_models_dir()?;

    let eou_dir = models_dir.join(PARAKEET_EOU_MODEL.extract_dir);
    if !eou_dir.exists() {
        fs::create_dir_all(&eou_dir).map_err(|e| ModelError::IoError(e.to_string()))?;
    }

    let eou_files = [
        ("encoder.onnx", "https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/encoder.onnx"),
        ("decoder_joint.onnx", "https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/decoder_joint.onnx"),
        ("tokenizer.json", "https://huggingface.co/altunenes/parakeet-rs/resolve/main/realtime_eou_120m-v1-onnx/tokenizer.json"),
    ];

    for (filename, url) in eou_files {
        let target = eou_dir.join(filename);
        if !target.exists() {
            progress.on_download_start(&format!("parakeet-eou/{}", filename), 0);
            download_file(url, &target, progress).await?;
        }
    }

    let tdt_dir = models_dir.join(PARAKEET_TDT_MODEL.extract_dir);
    if !tdt_dir.exists() {
        fs::create_dir_all(&tdt_dir).map_err(|e| ModelError::IoError(e.to_string()))?;
    }

    let tdt_files = [
        ("encoder-model.onnx", "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx"),
        ("encoder-model.onnx.data", "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.onnx.data"),
        ("decoder_joint-model.onnx", "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.onnx"),
        ("vocab.txt", "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt"),
    ];

    for (filename, url) in tdt_files {
        let target = tdt_dir.join(filename);
        if !target.exists() {
            progress.on_download_start(&format!("parakeet-tdt/{}", filename), 0);
            download_file(url, &target, progress).await?;
        }
    }

    Ok(())
}

async fn download_file(
    url: &str,
    target: &PathBuf,
    _progress: &dyn ModelProgressCallback,
) -> Result<(), ModelError> {
    let url = url.to_string();
    let target = target.clone();
    let _model_label = match get_models_dir() {
        Ok(models_dir) => target
            .strip_prefix(&models_dir)
            .ok()
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|| target.to_string_lossy().to_string()),
        Err(_) => target.to_string_lossy().to_string(),
    };

    tokio::task::spawn_blocking(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .map_err(|e| ModelError::DownloadError(e.to_string()))?;

        let mut response = client
            .get(&url)
            .send()
            .map_err(|e| ModelError::DownloadError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(ModelError::DownloadError(format!(
                "HTTP {}: {}",
                response.status(),
                url
            )));
        }

        let tmp_target = target.with_file_name(format!(
            "{}.downloading",
            target.file_name().unwrap_or_default().to_string_lossy()
        ));

        let mut file =
            fs::File::create(&tmp_target).map_err(|e| ModelError::IoError(e.to_string()))?;

        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read = response
                .read(&mut buffer)
                .map_err(|e| ModelError::IoError(e.to_string()))?;
            if bytes_read == 0 {
                break;
            }
            file.write_all(&buffer[..bytes_read])
                .map_err(|e| ModelError::IoError(e.to_string()))?;
        }

        drop(file);
        fs::rename(&tmp_target, &target).map_err(|e| ModelError::IoError(e.to_string()))?;

        Ok(())
    })
    .await
    .map_err(|e| ModelError::DownloadError(e.to_string()))?
}

pub async fn download_model(
    bundle: &ModelBundle,
    _progress: &dyn ModelProgressCallback,
) -> Result<(), ModelError> {
    let models_dir = get_models_dir()?;
    let url = bundle.url.to_string();
    let name = bundle.name.to_string();
    let extract_dir = bundle.extract_dir.to_string();

    tokio::task::spawn_blocking(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(1800))
            .build()
            .map_err(|e| ModelError::DownloadError(e.to_string()))?;

        let mut response = client
            .get(&url)
            .send()
            .map_err(|e| ModelError::DownloadError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(ModelError::DownloadError(format!(
                "HTTP {}: {}",
                response.status(),
                name
            )));
        }

        let is_archive = url.ends_with(".tar.bz2");

        if is_archive {
            let temp_path = models_dir.join(format!("{}.downloading", name));
            let mut file =
                fs::File::create(&temp_path).map_err(|e| ModelError::IoError(e.to_string()))?;

            let mut buffer = [0u8; 8192];

            loop {
                let bytes_read = response
                    .read(&mut buffer)
                    .map_err(|e| ModelError::IoError(e.to_string()))?;
                if bytes_read == 0 {
                    break;
                }
                file.write_all(&buffer[..bytes_read])
                    .map_err(|e| ModelError::IoError(e.to_string()))?;
            }

            drop(file);

            log::info!("Extracting {}", name);

            let tar_file =
                fs::File::open(&temp_path).map_err(|e| ModelError::IoError(e.to_string()))?;
            let decoder = bzip2::read::BzDecoder::new(tar_file);
            let mut archive = tar::Archive::new(decoder);
            archive
                .unpack(&models_dir)
                .map_err(|e| ModelError::IoError(format!("Failed to extract: {}", e)))?;

            let _ = fs::remove_file(&temp_path);
            log::info!("Extracted {} to {:?}", name, models_dir.join(&extract_dir));
        } else {
            let target_path = models_dir.join(&name);
            let mut file =
                fs::File::create(&target_path).map_err(|e| ModelError::IoError(e.to_string()))?;

            let mut downloaded: u64 = 0;
            let mut buffer = [0u8; 8192];

            loop {
                let bytes_read = response
                    .read(&mut buffer)
                    .map_err(|e| ModelError::IoError(e.to_string()))?;
                if bytes_read == 0 {
                    break;
                }
                file.write_all(&buffer[..bytes_read])
                    .map_err(|e| ModelError::IoError(e.to_string()))?;
                downloaded += bytes_read as u64;
            }

            log::info!("Downloaded {} ({} bytes)", name, downloaded);
        }

        Ok(())
    })
    .await
    .map_err(|e| ModelError::DownloadError(e.to_string()))?
}

pub fn total_models_size_mb() -> u64 {
    MODEL_BUNDLES.iter().map(|m| m.size_mb).sum()
}

pub fn get_missing_models() -> Vec<&'static ModelBundle> {
    MODEL_BUNDLES
        .iter()
        .filter(|m| {
            let target = if m.extract_dir.is_empty() {
                m.name
            } else {
                m.extract_dir
            };
            !model_exists(target)
        })
        .collect()
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Path error: {0}")]
    PathError(String),
    #[error("IO error: {0}")]
    IoError(String),
    #[error("Download error: {0}")]
    DownloadError(String),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
}
