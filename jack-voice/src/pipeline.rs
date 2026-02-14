//! Voice Pipeline - Framework-agnostic voice processing pipeline
//!
//! Replaces jack-desktop's Tauri-dependent VoicePipeline with a clean,
//! trait-based architecture. Consumers implement VoiceEventSink to receive
//! pipeline events.

use parking_lot::Mutex;
use std::sync::Arc;

use crate::audio_quality::{AudioQuality, AudioQualityGate};
use crate::speaker::SttMode;
use crate::stt::SpeechToText;
use crate::tts::{TextToSpeech, TtsEngine};
use crate::turn_detector::TurnDetector;
use crate::vad::{VadConfig, VoiceActivityDetector};

/// Events emitted by the voice pipeline
#[derive(Debug, Clone)]
pub enum VoiceEvent {
    /// Pipeline state changed
    StateChanged { state: PipelineState },
    /// Speech detected (VAD triggered)
    SpeechStart,
    /// Speech ended, transcription available
    Transcription {
        text: String,
        is_final: bool,
        latency_ms: u64,
    },
    /// Turn complete - full utterance ready
    TurnComplete { text: String },
    /// TTS started speaking
    SpeakingStart,
    /// TTS finished speaking
    SpeakingEnd,
    /// Error occurred
    Error { message: String },
    /// Audio health update
    AudioHealth { level: String, message: String },
}

/// Pipeline state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineState {
    Idle,
    Listening,
    Processing,
    Speaking,
}

/// Trait for receiving voice pipeline events
pub trait VoiceEventSink: Send + Sync {
    fn on_event(&self, event: VoiceEvent);
}

/// No-op event sink (silent operation)
pub struct NoopEventSink;
impl VoiceEventSink for NoopEventSink {
    fn on_event(&self, _event: VoiceEvent) {}
}

/// Logging event sink
pub struct LogEventSink;
impl VoiceEventSink for LogEventSink {
    fn on_event(&self, event: VoiceEvent) {
        match &event {
            VoiceEvent::StateChanged { state } => log::info!("[Pipeline] State: {:?}", state),
            VoiceEvent::SpeechStart => log::info!("[Pipeline] Speech detected"),
            VoiceEvent::Transcription {
                text,
                is_final,
                latency_ms,
            } => {
                log::info!(
                    "[Pipeline] STT: '{}' (final={}, {}ms)",
                    text,
                    is_final,
                    latency_ms
                );
            }
            VoiceEvent::TurnComplete { text } => log::info!("[Pipeline] Turn: '{}'", text),
            VoiceEvent::SpeakingStart => log::info!("[Pipeline] Speaking..."),
            VoiceEvent::SpeakingEnd => log::info!("[Pipeline] Done speaking"),
            VoiceEvent::Error { message } => log::error!("[Pipeline] Error: {}", message),
            VoiceEvent::AudioHealth { level, message } => {
                log::debug!("[Pipeline] Audio health: {} - {}", level, message);
            }
        }
    }
}

/// Voice pipeline configuration (replaces jack_core::settings dependency)
#[derive(Debug, Clone)]
pub struct VoicePipelineConfig {
    /// TTS engine to use
    pub tts_engine: TtsEngine,
    /// TTS voice/speaker ID
    pub tts_voice: String,
    /// TTS speech speed
    pub tts_speed: f32,
    /// STT mode preference
    pub stt_mode: SttMode,
    /// STT language (empty = auto-detect)
    pub stt_language: String,
    /// Enable barge-in (interrupt TTS when user speaks)
    pub barge_in_enabled: bool,
    /// Enable Smart Turn detection
    pub smart_turn_enabled: bool,
    /// VAD threshold override (None = use default)
    pub vad_threshold: Option<f32>,
}

impl Default for VoicePipelineConfig {
    fn default() -> Self {
        Self {
            tts_engine: TtsEngine::Pocket,
            tts_voice: "alba".to_string(),
            tts_speed: 1.0,
            stt_mode: SttMode::Streaming,
            stt_language: String::new(),
            barge_in_enabled: true,
            smart_turn_enabled: true,
            vad_threshold: None,
        }
    }
}

/// Voice pipeline that ties together VAD, STT, TTS, and turn detection
pub struct VoicePipeline {
    config: VoicePipelineConfig,
    state: Arc<Mutex<PipelineState>>,
    event_sink: Arc<dyn VoiceEventSink>,
    quality_gate: Arc<Mutex<AudioQualityGate>>,
}

impl VoicePipeline {
    /// Create a new voice pipeline with the given configuration
    pub fn new(config: VoicePipelineConfig, event_sink: Arc<dyn VoiceEventSink>) -> Self {
        Self {
            config,
            state: Arc::new(Mutex::new(PipelineState::Idle)),
            event_sink,
            quality_gate: Arc::new(Mutex::new(AudioQualityGate::new())),
        }
    }

    /// Get current pipeline state
    pub fn state(&self) -> PipelineState {
        *self.state.lock()
    }

    /// Update configuration
    pub fn set_config(&mut self, config: VoicePipelineConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &VoicePipelineConfig {
        &self.config
    }

    /// Create a VAD instance from current config
    pub fn create_vad(&self) -> Result<VoiceActivityDetector, crate::vad::VadError> {
        let mut vad_config = VadConfig::default();
        if let Some(threshold) = self.config.vad_threshold {
            vad_config.threshold = threshold;
        }
        VoiceActivityDetector::with_config(vad_config)
    }

    /// Create an STT instance from current config
    pub fn create_stt(&self) -> Result<SpeechToText, crate::stt::SttError> {
        let language = if self.config.stt_language.is_empty() {
            None
        } else {
            Some(self.config.stt_language.clone())
        };
        let voice = if self.config.tts_voice.is_empty() {
            None
        } else {
            Some(self.config.tts_voice.clone())
        };
        SpeechToText::with_language(self.config.stt_mode, language, voice)
    }

    /// Create a TTS instance from current config
    pub fn create_tts(&self) -> Result<TextToSpeech, crate::tts::TtsError> {
        let mut tts = TextToSpeech::with_engine(self.config.tts_engine.clone())?;
        tts.set_speed(self.config.tts_speed);
        if !self.config.tts_voice.is_empty() {
            let _ = tts.set_speaker(&self.config.tts_voice);
        }
        Ok(tts)
    }

    /// Create a turn detector
    pub fn create_turn_detector(
        &self,
    ) -> Result<TurnDetector, crate::turn_detector::SmartTurnError> {
        TurnDetector::new()
    }

    /// Emit a voice event
    pub fn emit(&self, event: VoiceEvent) {
        if let VoiceEvent::StateChanged { state } = &event {
            *self.state.lock() = *state;
        }
        self.event_sink.on_event(event);
    }

    /// Check audio quality
    pub fn check_audio_quality(&self, samples: &[f32]) -> AudioQuality {
        self.quality_gate.lock().check(samples)
    }
}
