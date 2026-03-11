//! Voice Pipeline - combines VAD, STT, LLM, and TTS
//!
//! Provides a unified interface for realtime voice conversations

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::{mpsc, RwLock};

use super::{
    ChatMessage, LlmClient, LlmConfig, RealtimeStt, RealtimeTts, RealtimeVad, TranscriptionEvent,
};
use crate::audio::AudioBuffer;

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub vad_enabled: bool,
    pub vad_threshold: f32,
    pub vad_silence_ms: u32,
    pub stt_mode: jack_voice::SttMode,
    pub llm_config: LlmConfig,
    pub tts_engine: jack_voice::TtsEngine,
    pub tts_voice: Option<String>,
    pub tts_language: Option<String>,
    pub turn_detection: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            vad_enabled: true,
            vad_threshold: 0.5,
            vad_silence_ms: 500,
            stt_mode: jack_voice::SttMode::Streaming,
            llm_config: LlmConfig::default(),
            tts_engine: jack_voice::TtsEngine::Auto,
            tts_voice: None,
            tts_language: None,
            turn_detection: true,
        }
    }
}

/// Pipeline events
#[derive(Debug, Clone)]
pub enum PipelineEvent {
    SpeechStarted { audio_start_ms: u32 },
    SpeechStopped { audio_end_ms: u32 },
    Transcription(TranscriptionEvent),
    TtsChunk(super::TtsChunk),
    TtsDone,
    Error(String),
}

/// Voice pipeline
pub struct VoicePipeline {
    vad: Arc<RwLock<Option<RealtimeVad>>>,
    stt: Arc<RwLock<Option<RealtimeStt>>>,
    tts: Arc<RwLock<Option<RealtimeTts>>>,
    llm: Arc<RwLock<Option<LlmClient>>>,
    config: PipelineConfig,
    input_buffer: Arc<RwLock<AudioBuffer>>,
    event_sender: mpsc::Sender<PipelineEvent>,
}

impl VoicePipeline {
    pub fn new(config: PipelineConfig, event_sender: mpsc::Sender<PipelineEvent>) -> Result<Self> {
        let vad = if config.vad_enabled {
            Some(RealtimeVad::with_config(
                config.vad_threshold,
                config.vad_silence_ms,
            )?)
        } else {
            None
        };

        let stt = Some(RealtimeStt::new(config.stt_mode)?);

        let mut tts = RealtimeTts::with_engine(config.tts_engine.clone())?;
        if let Some(language) = &config.tts_language {
            tts.set_language(language)?;
        }
        if let Some(voice) = &config.tts_voice {
            tts.set_voice(voice)?;
        }

        let llm = Some(LlmClient::new(config.llm_config.clone()));

        Ok(Self {
            vad: Arc::new(RwLock::new(vad)),
            stt: Arc::new(RwLock::new(stt)),
            tts: Arc::new(RwLock::new(Some(tts))),
            llm: Arc::new(RwLock::new(llm)),
            config,
            input_buffer: Arc::new(RwLock::new(AudioBuffer::new(16000, 1))),
            event_sender,
        })
    }

    /// Process incoming audio
    pub async fn process_audio(&self, audio_data: &[u8]) -> Result<()> {
        let mut buffer = self.input_buffer.write().await;
        buffer.append(audio_data)?;

        // Convert to f32 for VAD/STT
        let samples = crate::audio::pcm16_to_f32(audio_data);

        // Run VAD
        let mut vad_guard = self.vad.write().await;
        if let Some(ref mut vad) = *vad_guard {
            let result = vad.process(&samples);

            if result.is_speech {
                self.event_sender
                    .send(PipelineEvent::SpeechStarted { audio_start_ms: 0 })
                    .await
                    .ok();
            }
        }

        Ok(())
    }

    /// Commit buffered audio for transcription
    pub async fn commit_audio(&self) -> Result<String> {
        let mut buffer = self.input_buffer.write().await;

        let samples = buffer.get_f32();
        let item_id = format!("item_{}", uuid_simple());

        // Run STT
        let mut stt_guard = self.stt.write().await;
        if let Some(ref mut stt) = *stt_guard {
            let result = stt.transcribe(&samples)?;

            // Send transcription events
            if result.is_partial || result.is_final {
                self.event_sender
                    .send(PipelineEvent::Transcription(if result.is_final {
                        TranscriptionEvent::final_transcript(&item_id, result.text.clone())
                    } else {
                        TranscriptionEvent::partial(&item_id, result.text.clone())
                    }))
                    .await
                    .ok();
            }

            // Send speech stopped
            buffer.mark_speech_stopped();
            self.event_sender
                .send(PipelineEvent::SpeechStopped {
                    audio_end_ms: buffer.duration_ms() as u32,
                })
                .await
                .ok();

            buffer.clear();

            Ok(result.text)
        } else {
            anyhow::bail!("STT not initialized")
        }
    }

    /// Generate response from LLM and stream TTS
    pub async fn generate_response(&self, context: Vec<ChatMessage>) -> Result<()> {
        let llm_guard = self.llm.read().await;
        let llm = llm_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("LLM not initialized"))?;

        // Get LLM response
        let response = llm.chat(context).await?;

        // Stream TTS
        let mut tts_guard = self.tts.write().await;
        if let Some(ref mut tts) = *tts_guard {
            let mut index = 0u64;

            tts.synthesize_streaming(&response, |_samples, _sample_rate| {
                // In real implementation, this would send to event channel
                index += 1;
                true
            })?;

            self.event_sender.send(PipelineEvent::TtsDone).await.ok();
        }

        Ok(())
    }

    /// Get input buffer for external access
    pub fn input_buffer(&self) -> Arc<RwLock<AudioBuffer>> {
        self.input_buffer.clone()
    }
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:016x}", now)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();

        assert!(config.vad_enabled);
        assert_eq!(config.vad_threshold, 0.5);
        assert_eq!(config.stt_mode, jack_voice::SttMode::Streaming);
        assert_eq!(config.tts_engine, jack_voice::TtsEngine::Auto);
    }

    #[test]
    fn test_pipeline_config_custom() {
        let config = PipelineConfig {
            vad_enabled: false,
            vad_threshold: 0.3,
            vad_silence_ms: 1000,
            stt_mode: jack_voice::SttMode::Batch,
            llm_config: LlmConfig {
                base_url: "http://localhost:8080".to_string(),
                api_key: Some("test".to_string()),
                model: "custom-model".to_string(),
            },
            tts_engine: jack_voice::TtsEngine::Kokoro,
            tts_voice: Some("35".to_string()),
            tts_language: Some("it".to_string()),
            turn_detection: false,
        };

        assert!(!config.vad_enabled);
        assert_eq!(config.llm_config.model, "custom-model");
        assert_eq!(config.tts_voice, Some("35".to_string()));
        assert_eq!(config.tts_language, Some("it".to_string()));
    }
}
