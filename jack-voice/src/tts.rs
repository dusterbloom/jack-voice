// Jack Desktop - Text-to-Speech
// Supports multiple TTS engines:
// - Supertonic (fast English)
// - Kokoro (local multilingual)

use supertonic::{TextToSpeech as SupertonicTts, VoiceStyleData};

use crate::kokoro_tts::KokoroTts;
use crate::models;

/// Audio output from TTS
#[derive(Clone, Debug)]
pub struct AudioOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// TTS Engine type
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TtsEngine {
    Supertonic,
    Kokoro,
}

/// Internal TTS implementation
enum TtsImpl {
    Supertonic(SupertonicTts),
    Kokoro(KokoroTts),
}

pub struct TextToSpeech {
    engine: TtsImpl,
    speaker_id: String,
    speed: f32,
    sample_rate: u32,
}

impl TextToSpeech {
    /// Create a new TTS instance with default engine (Supertonic)
    pub fn new() -> Result<Self, TtsError> {
        Self::with_engine(TtsEngine::Supertonic)
    }

    /// Create TTS with specific engine
    pub fn with_engine(engine_type: TtsEngine) -> Result<Self, TtsError> {
        match engine_type {
            TtsEngine::Supertonic => Self::new_supertonic(),
            TtsEngine::Kokoro => Self::new_kokoro(),
        }
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

    /// Set the speaker voice by ID (e.g., "F1", "F2", "M1", "M2" for Supertonic, "0"-"10" for Kokoro)
    pub fn set_speaker(&mut self, speaker_id: &str) -> Result<(), TtsError> {
        match &mut self.engine {
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
        // Map numeric IDs to voice IDs
        let voice_id = match id {
            0 => "F1",
            1 => "F2",
            2 => "M1",
            3 => "M2",
            _ => "F1",
        };

        if let Err(e) = self.set_speaker(voice_id) {
            log::warn!("Failed to set speaker {}: {}", voice_id, e);
        }
    }

    /// Set the speech speed (0.5 = half speed, 2.0 = double speed)
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.clamp(0.25, 4.0);
        match &mut self.engine {
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

        let audio = self.synthesize(text)?;
        let _ = on_chunk(&audio.samples, audio.sample_rate);
        Ok(audio.sample_rate)
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
            TtsImpl::Supertonic(_) => "supertonic",
            TtsImpl::Kokoro(_) => "kokoro",
        }
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
        // Default to Supertonic for backwards compatibility
        Self::available_supertonic_voices()
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
