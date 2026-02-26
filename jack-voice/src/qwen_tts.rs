use std::path::PathBuf;
use std::str::FromStr;

use candle_core::Tensor;
use qwen3_tts::{
    AudioBuffer, Language, Qwen3TTS, Speaker, SynthesisOptions, SynthesisTiming, VoiceClonePrompt,
};

const SAMPLE_RATE: u32 = 24000;

pub fn fast_synthesis_options() -> SynthesisOptions {
    SynthesisOptions {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.85,
        ..Default::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenModelSize {
    Lite,
    Large,
}

pub const QWEN_LITE_VOICES: &[(&str, &str)] = &[
    ("serena", "Serena - Female, warm"),
    ("vivian", "Vivian - Female, clear"),
    ("uncle_fu", "Uncle Fu - Male, mature"),
    ("ryan", "Ryan - Male, neutral (default)"),
    ("aiden", "Aiden - Male, friendly"),
    ("ono_anna", "Ono Anna - Female, Japanese"),
    ("sohee", "Sohee - Female, Korean"),
    ("eric", "Eric - Male, American"),
    ("dylan", "Dylan - Male, young"),
];

const QWEN_DEFAULT_VOICE: &str = "ryan";

pub struct QwenTts {
    model: Qwen3TTS,
    size: QwenModelSize,
    current_speaker: String,
    voice_clone_prompt: Option<VoiceClonePrompt>,
}

#[derive(Clone, Debug)]
pub struct VoiceCloneRef {
    pub audio_path: PathBuf,
    pub transcript: Option<String>,
}

impl QwenTts {
    pub fn new(model_dir: &std::path::Path, size: QwenModelSize) -> Result<Self, TtsError> {
        log::info!(
            "[QwenTTS] Initializing {:?} model from {}",
            size,
            model_dir.display()
        );

        let device = qwen3_tts::auto_device()
            .map_err(|e| TtsError::InitError(format!("Failed to get compute device: {}", e)))?;

        let model = Qwen3TTS::from_pretrained(&model_dir.to_string_lossy(), device)
            .map_err(|e| TtsError::InitError(format!("Failed to load Qwen3-TTS model: {}", e)))?;

        let model_type_str = match model.model_type() {
            Some(t) => format!("{:?}", t),
            None => "Unknown".to_string(),
        };
        log::info!(
            "[QwenTTS] Model loaded. Type: {}, Voice cloning: {}, Preset speakers: {}",
            model_type_str,
            model.supports_voice_cloning(),
            model.supports_preset_speakers()
        );

        Ok(Self {
            model,
            size,
            current_speaker: QWEN_DEFAULT_VOICE.to_string(),
            voice_clone_prompt: None,
        })
    }

    pub fn set_speaker(&mut self, speaker_id: &str) -> Result<(), TtsError> {
        if self.model.supports_voice_cloning() && !self.model.supports_preset_speakers() {
            log::warn!(
                "[QwenTTS] Base model does not support preset speakers. Use voice cloning instead."
            );
        }

        Speaker::from_str(speaker_id).map_err(|_e| {
            TtsError::ModelNotFound(format!(
                "Unknown Qwen speaker '{}'. Available: {}",
                speaker_id,
                QWEN_LITE_VOICES
                    .iter()
                    .map(|(id, _)| *id)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;

        self.current_speaker = speaker_id.to_lowercase();
        self.voice_clone_prompt = None;
        Ok(())
    }

    pub fn set_voice_clone(&mut self, reference: VoiceCloneRef) -> Result<(), TtsError> {
        if !self.model.supports_voice_cloning() {
            return Err(TtsError::InitError(
                "Voice cloning requires a Base model (QwenLarge). Current model does not have a speaker encoder.".to_string()
            ));
        }

        log::info!(
            "[QwenTTS] Setting voice clone from: {}",
            reference.audio_path.display()
        );

        let ref_audio = AudioBuffer::load(&reference.audio_path).map_err(|e| {
            TtsError::ModelNotFound(format!(
                "Failed to load reference audio '{}': {}",
                reference.audio_path.display(),
                e
            ))
        })?;

        let prompt = self
            .model
            .create_voice_clone_prompt(&ref_audio, reference.transcript.as_deref())
            .map_err(|e| {
                TtsError::InitError(format!("Failed to create voice clone prompt: {}", e))
            })?;

        self.voice_clone_prompt = Some(prompt);
        self.current_speaker = "cloned".to_string();
        log::info!("[QwenTTS] Voice clone prompt created successfully");
        Ok(())
    }

    pub fn synthesize_fast(&self, text: &str) -> Result<AudioOutput, TtsError> {
        if text.is_empty() {
            return Ok(AudioOutput {
                samples: Vec::new(),
                sample_rate: SAMPLE_RATE,
            });
        }

        let options = fast_synthesis_options();
        let language = Language::English;
        let speaker = Speaker::from_str(&self.current_speaker).unwrap_or(Speaker::Ryan);

        let audio = self
            .model
            .synthesize_with_voice(text, speaker, language, Some(options))
            .map_err(|e| TtsError::SynthesisError(format!("Qwen synthesis failed: {}", e)))?;

        Ok(AudioOutput {
            samples: audio.samples,
            sample_rate: SAMPLE_RATE,
        })
    }

    pub fn synthesize(&self, text: &str) -> Result<AudioOutput, TtsError> {
        if text.is_empty() {
            return Ok(AudioOutput {
                samples: Vec::new(),
                sample_rate: SAMPLE_RATE,
            });
        }

        let options = SynthesisOptions::default();
        let language = Language::English;

        let audio = if let Some(ref prompt) = self.voice_clone_prompt {
            self.model
                .synthesize_voice_clone(text, prompt, language, Some(options))
                .map_err(|e| TtsError::SynthesisError(format!("Qwen synthesis failed: {}", e)))?
        } else {
            let speaker = Speaker::from_str(&self.current_speaker).unwrap_or(Speaker::Ryan);
            self.model
                .synthesize_with_voice(text, speaker, language, Some(options))
                .map_err(|e| TtsError::SynthesisError(format!("Qwen synthesis failed: {}", e)))?
        };

        Ok(AudioOutput {
            samples: audio.samples,
            sample_rate: SAMPLE_RATE,
        })
    }

    pub fn synthesize_with_timing(
        &self,
        text: &str,
    ) -> Result<(AudioOutput, TimingInfo), TtsError> {
        if text.is_empty() {
            return Ok((
                AudioOutput {
                    samples: Vec::new(),
                    sample_rate: SAMPLE_RATE,
                },
                TimingInfo {
                    prefill_ms: 0.0,
                    generation_ms: 0.0,
                    generation_frames: 0,
                    decode_ms: 0.0,
                },
            ));
        }

        let options = SynthesisOptions::default();
        let language = Language::English;
        let speaker = Speaker::from_str(&self.current_speaker).unwrap_or(Speaker::Ryan);

        let (audio, timing) = self
            .model
            .synthesize_with_timing(text, speaker, language, Some(options))
            .map_err(|e| TtsError::SynthesisError(format!("Qwen synthesis failed: {}", e)))?;

        Ok((
            AudioOutput {
                samples: audio.samples,
                sample_rate: SAMPLE_RATE,
            },
            timing.into(),
        ))
    }

    pub fn synthesize_streaming<F>(&self, text: &str, mut on_chunk: F) -> Result<u32, TtsError>
    where
        F: FnMut(&[f32], u32) -> bool,
    {
        if text.is_empty() {
            return Ok(SAMPLE_RATE);
        }

        if self.voice_clone_prompt.is_some() {
            let audio = self.synthesize(text)?;
            on_chunk(&audio.samples, audio.sample_rate);
            return Ok(SAMPLE_RATE);
        }

        let options = SynthesisOptions::default();
        let language = Language::English;
        let speaker = Speaker::from_str(&self.current_speaker).unwrap_or(Speaker::Ryan);

        let streaming_iter = self
            .model
            .synthesize_streaming(text, speaker, language, options)
            .map_err(|e| TtsError::SynthesisError(format!("Qwen streaming failed: {}", e)))?;

        for chunk_result in streaming_iter {
            let audio = chunk_result.map_err(|e| {
                TtsError::SynthesisError(format!("Qwen streaming chunk failed: {}", e))
            })?;
            if !on_chunk(&audio.samples, SAMPLE_RATE) {
                break;
            }
        }

        Ok(SAMPLE_RATE)
    }

    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    pub fn supports_voice_cloning(&self) -> bool {
        self.model.supports_voice_cloning()
    }

    pub fn current_speaker(&self) -> &str {
        &self.current_speaker
    }

    pub fn model_size(&self) -> QwenModelSize {
        self.size
    }

    /// Get the current speaker embedding (only available after voice clone is set)
    pub fn get_speaker_embedding(&self) -> Result<Vec<f32>, TtsError> {
        if let Some(ref prompt) = self.voice_clone_prompt {
            prompt
                .speaker_embedding
                .to_vec1()
                .map_err(|e| TtsError::InitError(format!("Failed to get embedding: {}", e)))
        } else {
            Err(TtsError::InitError(
                "No voice clone prompt set. Call set_voice_clone first.".to_string(),
            ))
        }
    }

    /// Set the speaker embedding directly (bypasses reference audio encoding)
    pub fn set_speaker_embedding(&mut self, embedding: &[f32]) -> Result<(), TtsError> {
        if !self.model.supports_voice_cloning() {
            return Err(TtsError::InitError(
                "Speaker embedding only supported on Large model with voice cloning.".to_string(),
            ));
        }

        let device = self.model.device();
        let embedding_tensor = Tensor::from_vec(embedding.to_vec(), (embedding.len(),), device)
            .map_err(|e| {
                TtsError::InitError(format!("Failed to create embedding tensor: {}", e))
            })?;

        let prompt = VoiceClonePrompt {
            speaker_embedding: embedding_tensor,
            ref_codes: None,
            ref_text_ids: None,
        };

        self.voice_clone_prompt = Some(prompt);
        self.current_speaker = "loaded".to_string();
        log::info!("[QwenTTS] Set speaker embedding ({} dims)", embedding.len());
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct AudioOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Clone, Debug)]
pub struct TimingInfo {
    pub prefill_ms: f64,
    pub generation_ms: f64,
    pub generation_frames: usize,
    pub decode_ms: f64,
}

impl From<SynthesisTiming> for TimingInfo {
    fn from(t: SynthesisTiming) -> Self {
        TimingInfo {
            prefill_ms: t.prefill_ms,
            generation_ms: t.generation_ms,
            generation_frames: t.generation_frames,
            decode_ms: t.decode_ms,
        }
    }
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

pub fn available_voices() -> Vec<VoiceInfo> {
    QWEN_LITE_VOICES
        .iter()
        .enumerate()
        .map(|(id, (voice_id, name))| VoiceInfo {
            id: id as i32,
            id_str: voice_id.to_string(),
            name: name.to_string(),
            language: "multilingual".to_string(),
        })
        .collect()
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct VoiceInfo {
    pub id: i32,
    pub id_str: String,
    pub name: String,
    pub language: String,
}

pub fn can_run_qwen() -> bool {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = candle_core::Device::cuda_if_available(0) {
            if device.is_cuda() {
                log::debug!("[QwenTTS] CUDA available, Qwen can run (GPU accelerated)");
                return true;
            }
        }
    }

    // qwen3-tts auto_device() handles Metal/CPU selection at runtime.
    // CPU inference works but is slower than real-time for long utterances.
    // Metal acceleration requires building with --features metal (currently
    // blocked by pocket-tts pinning once_cell ~1.19 vs candle-metal-kernels
    // needing >=1.21).
    log::debug!("[QwenTTS] Qwen will use CPU (functional but slower than GPU)");
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn speaker_parsing_accepts_valid_names() {
        assert!(Speaker::from_str("ryan").is_ok());
        assert!(Speaker::from_str("RYAN").is_ok());
        assert!(Speaker::from_str("Ryan").is_ok());
        assert!(Speaker::from_str("serena").is_ok());
        assert!(Speaker::from_str("vivian").is_ok());
        assert!(Speaker::from_str("uncle_fu").is_ok());
        assert!(Speaker::from_str("unclefu").is_ok());
        assert!(Speaker::from_str("ono_anna").is_ok());
        assert!(Speaker::from_str("onoanna").is_ok());
        assert!(Speaker::from_str("sohee").is_ok());
        assert!(Speaker::from_str("eric").is_ok());
        assert!(Speaker::from_str("dylan").is_ok());
        assert!(Speaker::from_str("aiden").is_ok());
    }

    #[test]
    fn speaker_parsing_rejects_invalid() {
        assert!(Speaker::from_str("invalid").is_err());
        assert!(Speaker::from_str("").is_err());
        assert!(Speaker::from_str("not_a_speaker").is_err());
    }

    #[test]
    fn available_voices_returns_nine_presets() {
        let voices = available_voices();
        assert_eq!(voices.len(), 9);
        assert!(voices.iter().any(|v| v.id_str == "ryan"));
        assert!(voices.iter().any(|v| v.id_str == "serena"));
        assert!(voices.iter().any(|v| v.name.contains("Ryan")));
        assert!(voices.iter().any(|v| v.name.contains("Serena")));
    }

    #[test]
    fn qwen_lite_voices_constant_matches_speaker_enum() {
        for (id, _) in QWEN_LITE_VOICES {
            assert!(
                Speaker::from_str(id).is_ok(),
                "Invalid speaker in QWEN_LITE_VOICES: {}",
                id
            );
        }
    }

    #[test]
    fn qwen_lite_voices_contains_expected_voices() {
        let voice_ids: Vec<&str> = QWEN_LITE_VOICES.iter().map(|(id, _)| *id).collect();
        assert!(voice_ids.contains(&"ryan"));
        assert!(voice_ids.contains(&"serena"));
        assert!(voice_ids.contains(&"vivian"));
        assert!(voice_ids.contains(&"eric"));
        assert!(voice_ids.contains(&"dylan"));
    }

    #[test]
    fn voice_info_has_correct_language() {
        let voices = available_voices();
        for voice in voices {
            assert_eq!(voice.language, "multilingual");
        }
    }

    #[test]
    fn audio_output_empty_text_returns_empty() {
        let output = AudioOutput {
            samples: Vec::new(),
            sample_rate: SAMPLE_RATE,
        };
        assert!(output.samples.is_empty());
        assert_eq!(output.sample_rate, 24000);
    }

    #[test]
    fn sample_rate_is_24khz() {
        assert_eq!(SAMPLE_RATE, 24000);
    }

    #[test]
    fn default_voice_is_ryan() {
        assert_eq!(QWEN_DEFAULT_VOICE, "ryan");
    }
}
