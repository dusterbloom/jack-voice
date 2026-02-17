// Jack Desktop - Text-to-Speech
// Supports multiple TTS engines:
// - Pocket (fast English, pure Rust Candle)
// - Supertonic (fast English)
// - Kokoro (local multilingual)

use pocket_tts::{ModelState as PocketModelState, TTSModel as PocketTtsModel};
use std::path::Path;
use supertonic::{TextToSpeech as SupertonicTts, VoiceStyleData};

use crate::kokoro_tts::KokoroTts;
use crate::models;

const POCKET_MODEL_VARIANT: &str = "b6369a24";
const POCKET_DEFAULT_VOICE: &str = "alba";
const POCKET_VOICES: &[&str] = &[
    "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
];

/// Audio output from TTS
#[derive(Clone, Debug)]
pub struct AudioOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// TTS Engine type
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TtsEngine {
    Pocket,
    Supertonic,
    Kokoro,
}

/// Internal TTS implementation
enum TtsImpl {
    Pocket(PocketTts),
    Supertonic(SupertonicTts),
    Kokoro(KokoroTts),
}

struct PocketTts {
    model: PocketTtsModel,
    voice_state: PocketModelState,
    voice_id: String,
}

#[derive(Debug)]
enum PocketVoiceInput<'a> {
    Preset(&'a str),
    VoiceCloneWav(&'a Path),
    PromptStateFile(&'a Path),
}

impl PocketTts {
    fn new_with_voice(voice_id: &str) -> Result<Self, TtsError> {
        let model = PocketTtsModel::load(POCKET_MODEL_VARIANT)
            .map_err(|e| TtsError::InitError(format!("Pocket init failed: {}", e)))?;
        let voice_state = load_pocket_voice_state(&model, voice_id)?;

        Ok(Self {
            model,
            voice_state,
            voice_id: voice_id.to_string(),
        })
    }

    fn set_voice(&mut self, voice_id: &str) -> Result<(), TtsError> {
        self.voice_state = load_pocket_voice_state(&self.model, voice_id)?;
        self.voice_id = voice_id.to_string();
        Ok(())
    }

    fn synthesize(&self, text: &str) -> Result<AudioOutput, TtsError> {
        let audio = self
            .model
            .generate(text, &self.voice_state)
            .map_err(|e| TtsError::SynthesisError(format!("Pocket synthesis failed: {}", e)))?;
        let channels = audio
            .to_vec2::<f32>()
            .map_err(|e| TtsError::SynthesisError(format!("Pocket output decode failed: {}", e)))?;
        let samples = channels.into_iter().next().unwrap_or_default();

        Ok(AudioOutput {
            samples,
            sample_rate: self.sample_rate(),
        })
    }

    fn synthesize_streaming<F>(&self, text: &str, on_chunk: &mut F) -> Result<u32, TtsError>
    where
        F: FnMut(&[f32], u32) -> bool,
    {
        for chunk in self.model.generate_stream(text, &self.voice_state) {
            let chunk = chunk.map_err(|e| {
                TtsError::SynthesisError(format!("Pocket streaming synthesis failed: {}", e))
            })?;
            let chunk = chunk.squeeze(0).map_err(|e| {
                TtsError::SynthesisError(format!("Pocket streaming chunk decode failed: {}", e))
            })?;
            let channels = chunk.to_vec2::<f32>().map_err(|e| {
                TtsError::SynthesisError(format!("Pocket streaming chunk decode failed: {}", e))
            })?;
            let samples = channels.into_iter().next().unwrap_or_default();

            if !on_chunk(&samples, self.sample_rate()) {
                break;
            }
        }

        Ok(self.sample_rate())
    }

    fn sample_rate(&self) -> u32 {
        self.model.sample_rate as u32
    }
}

fn load_pocket_voice_state(
    model: &PocketTtsModel,
    voice_id: &str,
) -> Result<PocketModelState, TtsError> {
    match classify_pocket_voice_input(voice_id)? {
        PocketVoiceInput::Preset(preset_voice_id) => {
            let prompt_hf_path = format!(
                "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/{}.safetensors",
                preset_voice_id
            );
            let prompt_path =
                pocket_tts::weights::download_if_necessary(&prompt_hf_path).map_err(|e| {
                    TtsError::ModelNotFound(format!(
                        "Pocket voice '{}' download failed: {}",
                        preset_voice_id, e
                    ))
                })?;

            model
                .get_voice_state_from_prompt_file(&prompt_path)
                .map_err(|e| {
                    TtsError::InitError(format!(
                        "Pocket voice '{}' load failed: {}",
                        preset_voice_id, e
                    ))
                })
        }
        PocketVoiceInput::VoiceCloneWav(path) => model.get_voice_state(path).map_err(|e| {
            TtsError::InitError(format!(
                "Pocket voice cloning failed from '{}': {}",
                path.display(),
                e
            ))
        }),
        PocketVoiceInput::PromptStateFile(path) => {
            model.get_voice_state_from_prompt_file(path).map_err(|e| {
                TtsError::InitError(format!(
                    "Pocket prompt state load failed from '{}': {}",
                    path.display(),
                    e
                ))
            })
        }
    }
}

fn classify_pocket_voice_input<'a>(voice_id: &'a str) -> Result<PocketVoiceInput<'a>, TtsError> {
    let voice_id = voice_id.trim();
    if voice_id.is_empty() {
        return Err(TtsError::ModelNotFound(
            "Pocket voice cannot be empty".to_string(),
        ));
    }

    if POCKET_VOICES.contains(&voice_id) {
        return Ok(PocketVoiceInput::Preset(voice_id));
    }

    let path = Path::new(voice_id);
    if !path.exists() {
        return Err(TtsError::ModelNotFound(format!(
            "Unknown Pocket voice '{}'. Expected one of: {} or an existing .wav/.safetensors file path",
            voice_id,
            POCKET_VOICES.join(", ")
        )));
    }

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match extension.as_str() {
        "wav" => Ok(PocketVoiceInput::VoiceCloneWav(path)),
        "safetensors" => Ok(PocketVoiceInput::PromptStateFile(path)),
        _ => Err(TtsError::ModelNotFound(format!(
            "Unsupported Pocket voice file '{}'. Expected .wav for cloning or .safetensors for prompt state",
            path.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{classify_pocket_voice_input, PocketVoiceInput, TtsError};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_file(ext: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join("jack-voice-tests");
        fs::create_dir_all(&dir).expect("failed to create temp test dir");

        let path = dir.join(format!("pocket-voice-{nanos}.{ext}"));
        fs::write(&path, b"test").expect("failed to write temp test file");
        path
    }

    #[test]
    fn classify_pocket_voice_input_accepts_preset() {
        match classify_pocket_voice_input("alba").expect("preset should be accepted") {
            PocketVoiceInput::Preset("alba") => {}
            _ => panic!("expected preset voice classification"),
        }
    }

    #[test]
    fn classify_pocket_voice_input_accepts_wav_file() {
        let path = unique_temp_file("WAV");
        let voice = path.to_string_lossy().to_string();

        match classify_pocket_voice_input(&voice).expect("wav path should be accepted") {
            PocketVoiceInput::VoiceCloneWav(classified_path) => {
                assert_eq!(classified_path, path.as_path());
            }
            _ => panic!("expected wav voice clone path classification"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn classify_pocket_voice_input_accepts_prompt_state_file() {
        let path = unique_temp_file("safetensors");
        let voice = path.to_string_lossy().to_string();

        match classify_pocket_voice_input(&voice).expect("prompt path should be accepted") {
            PocketVoiceInput::PromptStateFile(classified_path) => {
                assert_eq!(classified_path, path.as_path());
            }
            _ => panic!("expected prompt state file classification"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn classify_pocket_voice_input_rejects_unknown_voice() {
        let err = classify_pocket_voice_input("not-a-real-voice")
            .expect_err("unknown non-path voice should fail");
        match err {
            TtsError::ModelNotFound(message) => {
                assert!(message.contains("Unknown Pocket voice"));
            }
            _ => panic!("expected model-not-found error"),
        }
    }

    #[test]
    fn classify_pocket_voice_input_rejects_unsupported_file_extension() {
        let path = unique_temp_file("txt");
        let voice = path.to_string_lossy().to_string();

        let err = classify_pocket_voice_input(&voice).expect_err("txt file should be rejected");
        match err {
            TtsError::ModelNotFound(message) => {
                assert!(message.contains("Unsupported Pocket voice file"));
            }
            _ => panic!("expected model-not-found error"),
        }

        let _ = fs::remove_file(path);
    }
}

pub struct TextToSpeech {
    engine: TtsImpl,
    speaker_id: String,
    speed: f32,
    sample_rate: u32,
}

impl TextToSpeech {
    /// Create a new TTS instance with default engine (Pocket)
    pub fn new() -> Result<Self, TtsError> {
        Self::with_engine(TtsEngine::Pocket)
    }

    /// Create TTS with specific engine
    pub fn with_engine(engine_type: TtsEngine) -> Result<Self, TtsError> {
        match engine_type {
            TtsEngine::Pocket => Self::new_pocket(),
            TtsEngine::Supertonic => Self::new_supertonic(),
            TtsEngine::Kokoro => Self::new_kokoro(),
        }
    }

    /// Create Pocket TTS instance
    fn new_pocket() -> Result<Self, TtsError> {
        Self::new_pocket_with_voice(POCKET_DEFAULT_VOICE)
    }

    /// Create Pocket TTS instance with specific preset voice
    pub fn new_pocket_with_voice(voice_id: &str) -> Result<Self, TtsError> {
        let pocket = PocketTts::new_with_voice(voice_id)?;
        let sample_rate = pocket.sample_rate();

        Ok(Self {
            engine: TtsImpl::Pocket(pocket),
            speaker_id: voice_id.to_string(),
            speed: 1.0,
            sample_rate,
        })
    }

    /// Create Supertonic TTS instance
    fn new_supertonic() -> Result<Self, TtsError> {
        let paths = models::get_supertonic_paths()?;
        Self::with_supertonic_paths(&paths)
    }

    /// Create Kokoro TTS instance
    fn new_kokoro() -> Result<Self, TtsError> {
        Self::new_kokoro_with_voice("0") // Default to voice 0 (Alloy - American English)
    }

    /// Create Kokoro TTS instance with specific voice
    /// Initializes with correct language for the voice
    pub fn new_kokoro_with_voice(voice_id: &str) -> Result<Self, TtsError> {
        // Parse voice ID to determine language
        let voice_num = voice_id.parse::<i32>().unwrap_or(0);
        let language = crate::kokoro_tts::voice_id_to_language(voice_num);

        log::info!(
            "[TTS] Initializing Kokoro with voice {} (language: {})",
            voice_id,
            language
        );

        let kokoro = KokoroTts::new_with_language(language)
            .map_err(|e| TtsError::InitError(format!("Kokoro init failed: {}", e)))?;

        Ok(Self {
            engine: TtsImpl::Kokoro(kokoro),
            speaker_id: voice_id.to_string(),
            speed: 1.0,
            sample_rate: 24000, // Kokoro fixed sample rate
        })
    }

    /// Create TTS with specific Supertonic model paths
    pub fn with_supertonic_paths(paths: &models::SupertonicPaths) -> Result<Self, TtsError> {
        // Verify required model files exist
        if !paths.all_exist() {
            return Err(TtsError::ModelNotFound(
                "Supertonic models not fully downloaded".to_string(),
            ));
        }

        // Create Supertonic TTS instance
        let mut tts =
            SupertonicTts::new(&paths.model_dir).map_err(|e| TtsError::InitError(e.to_string()))?;

        // Load default voice (F1)
        let default_voice = "F1";
        let voice_path = paths.voice_path(default_voice);

        if voice_path.exists() {
            let voice_data =
                VoiceStyleData::from_json_file(&voice_path, default_voice, "Default Voice")
                    .map_err(|e| TtsError::InitError(format!("Failed to load voice: {}", e)))?;
            tts.set_voice_style(&voice_data);
        } else {
            return Err(TtsError::ModelNotFound(format!(
                "Default voice file not found: {}",
                voice_path.display()
            )));
        }

        Ok(Self {
            engine: TtsImpl::Supertonic(tts),
            speaker_id: default_voice.to_string(),
            speed: 1.0,
            sample_rate: supertonic::SAMPLE_RATE,
        })
    }

    /// Check if model is ready
    pub fn is_ready(&self) -> bool {
        true // If we got here, the model loaded successfully
    }

    /// Set speaker voice.
    /// Pocket accepts preset names (`alba`, `marius`, etc.) or local `.wav`/`.safetensors` paths.
    /// Supertonic accepts voice IDs like `F1`/`M2`; Kokoro accepts numeric IDs as strings.
    pub fn set_speaker(&mut self, speaker_id: &str) -> Result<(), TtsError> {
        match &mut self.engine {
            TtsImpl::Pocket(pocket) => {
                pocket.set_voice(speaker_id)?;
                self.speaker_id = speaker_id.to_string();
                Ok(())
            }
            TtsImpl::Supertonic(tts) => {
                let paths = models::get_supertonic_paths()?;
                let voice_path = paths.voice_path(speaker_id);

                if !voice_path.exists() {
                    return Err(TtsError::ModelNotFound(format!(
                        "Voice file not found: {}",
                        voice_path.display()
                    )));
                }

                let voice_data =
                    VoiceStyleData::from_json_file(&voice_path, speaker_id, speaker_id)
                        .map_err(|e| TtsError::InitError(format!("Failed to load voice: {}", e)))?;
                tts.set_voice_style(&voice_data);

                self.speaker_id = speaker_id.to_string();
                Ok(())
            }
            TtsImpl::Kokoro(kokoro) => {
                // Kokoro uses numeric speaker IDs, validate and set language
                match speaker_id.parse::<i32>() {
                    Ok(voice_id) => {
                        // Update language based on voice ID
                        kokoro.set_language_for_voice(voice_id).map_err(|e| {
                            TtsError::InitError(format!(
                                "Failed to set language for voice {}: {}",
                                voice_id, e
                            ))
                        })?;

                        self.speaker_id = speaker_id.to_string();
                        log::info!("[TTS] Set Kokoro voice to {}", voice_id);
                        Ok(())
                    }
                    Err(_) => Err(TtsError::InitError(format!(
                        "Invalid Kokoro speaker ID: {}",
                        speaker_id
                    ))),
                }
            }
        }
    }

    /// Set the speaker voice by numeric ID (for backwards compatibility)
    pub fn set_speaker_id(&mut self, id: i32) {
        let voice = match &self.engine {
            TtsImpl::Pocket(_) => match id {
                0 => "alba",
                1 => "marius",
                2 => "javert",
                3 => "jean",
                4 => "fantine",
                5 => "cosette",
                6 => "eponine",
                7 => "azelma",
                _ => POCKET_DEFAULT_VOICE,
            },
            TtsImpl::Supertonic(_) => match id {
                0 => "F1",
                1 => "F2",
                2 => "M1",
                3 => "M2",
                _ => "F1",
            },
            TtsImpl::Kokoro(_) => match id {
                0..=10 => {
                    let voice_id = id.to_string();
                    if let Err(e) = self.set_speaker(&voice_id) {
                        log::warn!("Failed to set speaker {}: {}", voice_id, e);
                    }
                    return;
                }
                _ => "0",
            },
        };

        if let Err(e) = self.set_speaker(voice) {
            log::warn!("Failed to set speaker {}: {}", voice, e);
        }
    }

    /// Set the speech speed (0.5 = half speed, 2.0 = double speed)
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.clamp(0.25, 4.0);
        match &mut self.engine {
            TtsImpl::Pocket(_) => {
                // Pocket currently uses fixed decoding parameters.
            }
            TtsImpl::Supertonic(tts) => tts.set_speed(self.speed),
            TtsImpl::Kokoro(_) => {
                // Kokoro speed is set per-synthesis call, not as a persistent setting
                // We'll store it and use it in synthesize()
            }
        }
    }

    /// Synthesize text to audio samples
    pub fn synthesize(&mut self, text: &str) -> Result<AudioOutput, TtsError> {
        if text.is_empty() {
            return Ok(AudioOutput {
                samples: Vec::new(),
                sample_rate: self.sample_rate,
            });
        }

        match &mut self.engine {
            TtsImpl::Pocket(tts) => tts.synthesize(text),
            TtsImpl::Supertonic(tts) => {
                let audio = tts
                    .synthesize(text)
                    .map_err(|e| TtsError::SynthesisError(e.to_string()))?;

                Ok(AudioOutput {
                    samples: audio.samples,
                    sample_rate: audio.sample_rate,
                })
            }
            TtsImpl::Kokoro(tts) => {
                let speaker_id = self.speaker_id.parse::<i32>().unwrap_or(0);
                let audio = tts
                    .synthesize(text, speaker_id, self.speed)
                    .map_err(|e| TtsError::SynthesisError(e.to_string()))?;

                Ok(AudioOutput {
                    samples: audio.samples,
                    sample_rate: audio.sample_rate,
                })
            }
        }
    }

    /// Synthesize text and stream audio chunks to a callback.
    ///
    /// For engines that don't support true incremental audio generation, this
    /// calls `synthesize()` internally and invokes `on_chunk` once.
    pub fn synthesize_streaming<F>(&mut self, text: &str, mut on_chunk: F) -> Result<u32, TtsError>
    where
        F: FnMut(&[f32], u32) -> bool,
    {
        let text = text.trim();
        if text.is_empty() {
            return Ok(self.sample_rate);
        }

        match &mut self.engine {
            TtsImpl::Pocket(tts) => tts.synthesize_streaming(text, &mut on_chunk),
            _ => {
                let audio = self.synthesize(text)?;
                let _ = on_chunk(&audio.samples, audio.sample_rate);
                Ok(audio.sample_rate)
            }
        }
    }

    /// Get the output sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get current speaker ID
    pub fn current_speaker(&self) -> &str {
        &self.speaker_id
    }

    /// Get current engine type
    pub fn engine_type(&self) -> &str {
        match self.engine {
            TtsImpl::Pocket(_) => "pocket",
            TtsImpl::Supertonic(_) => "supertonic",
            TtsImpl::Kokoro(_) => "kokoro",
        }
    }

    /// Get available voices for Pocket
    pub fn available_pocket_voices() -> Vec<VoiceInfo> {
        vec![
            VoiceInfo {
                id: 0,
                name: "Alba".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 1,
                name: "Marius".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 2,
                name: "Javert".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 3,
                name: "Jean".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 4,
                name: "Fantine".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 5,
                name: "Cosette".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 6,
                name: "Eponine".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 7,
                name: "Azelma".to_string(),
                language: "en".to_string(),
            },
        ]
    }

    /// Get available voices for Supertonic
    pub fn available_supertonic_voices() -> Vec<VoiceInfo> {
        vec![
            VoiceInfo {
                id: 0,
                name: "Female 1 (F1)".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 1,
                name: "Female 2 (F2)".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 2,
                name: "Male 1 (M1)".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 3,
                name: "Male 2 (M2)".to_string(),
                language: "en".to_string(),
            },
        ]
    }

    /// Get available voices for Kokoro
    pub fn available_kokoro_voices() -> Vec<VoiceInfo> {
        // Kokoro-en-v0_19 has 11 speakers (0-10)
        (0..11)
            .map(|id| VoiceInfo {
                id,
                name: format!("Voice {}", id),
                language: "en".to_string(),
            })
            .collect()
    }

    /// Get available voices (legacy method, returns current engine's voices)
    pub fn available_voices() -> Vec<VoiceInfo> {
        Self::available_pocket_voices()
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct VoiceInfo {
    pub id: i32,
    pub name: String,
    pub language: String,
}

#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Initialization error: {0}")]
    InitError(String),
    #[error("Synthesis error: {0}")]
    SynthesisError(String),
}

impl From<crate::models::ModelError> for TtsError {
    fn from(e: crate::models::ModelError) -> Self {
        TtsError::ModelNotFound(e.to_string())
    }
}
