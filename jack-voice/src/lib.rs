//! Jack Voice - Production-quality voice pipeline
//!
//! Provides VAD (Silero), STT (Whisper/Moonshine/Parakeet), TTS (Pocket/Supertonic/Kokoro),
//! turn detection (SmartTurn), audio capture/playback, and voice calibration.
//!
//! Extracted from jack-desktop as a standalone, framework-agnostic crate.

// Core modules
pub mod audio;
pub mod audio_quality;
pub mod calibration;
pub mod kokoro_tts;
pub mod models;
pub mod parakeet_stt;
pub mod pipeline;
pub mod speaker;
pub mod stt;
pub mod tts;
pub mod turn_detector;
pub mod vad;
pub mod watchdog;

// Re-export main types for convenience
pub use audio::{AudioCapture, AudioError, AudioHealthMonitor, AudioPlayer};
pub use audio_quality::{AudioQuality, AudioQualityGate};
pub use calibration::{CalibrationResults, CalibrationSession, VoiceConfig};
pub use models::{LogProgress, ModelError, ModelProgressCallback, NoopProgress};
pub use pipeline::{VoiceEvent, VoiceEventSink, VoicePipeline, VoicePipelineConfig};
pub use speaker::{SpeakerProfile, SttMode, TurnMetrics};
pub use stt::{SpeechToText, SttError, TranscriptionResult};
pub use tts::AudioOutput;
pub use tts::{TextToSpeech, TtsEngine, TtsError};
pub use turn_detector::{TurnDecision, TurnDetector};
pub use vad::{SpeechSegment, VadConfig, VadError, VoiceActivityDetector};
pub use watchdog::TimeoutTracker;
