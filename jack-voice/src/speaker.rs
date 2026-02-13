// Jack Desktop - Speaker Profile for Adaptive Voice Pipeline
// Learns speaker patterns to optimize latency and accuracy

use dirs::data_dir;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
const PROFILE_PATH: &str = "jack-desktop/speaker_profile.json";

const FAST_SPEAKER_WPS: f32 = 3.5;
#[allow(dead_code)]
const SLOW_SPEAKER_WPS: f32 = 2.0;

const EMA_ALPHA: f32 = 0.3;
const MIN_CALIBRATION_TURNS: usize = 3;

const DEFAULT_VAD_SILENCE_MS: f32 = 600.0;
const FAST_VAD_SILENCE_MS: f32 = 300.0;
const SLOW_VAD_SILENCE_MS: f32 = 800.0;

const DEFAULT_STEP_MS: u32 = 500;
const FAST_STEP_MS: u32 = 300;
#[allow(dead_code)]
const SLOW_STEP_MS: u32 = 800;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerProfile {
    pub avg_pause_ms: f32,
    pub avg_word_gap_ms: f32,
    pub speech_rate_wps: f32,
    pub turns_observed: usize,
    pub is_fast_speaker: bool,
    pub is_calibrated: bool,
    pub last_speech_time: Option<String>,
}

impl Default for SpeakerProfile {
    fn default() -> Self {
        Self {
            avg_pause_ms: DEFAULT_VAD_SILENCE_MS,
            avg_word_gap_ms: 100.0,
            speech_rate_wps: 2.5,
            turns_observed: 0,
            is_fast_speaker: false,
            is_calibrated: false,
            last_speech_time: None,
        }
    }
}

impl SpeakerProfile {
    pub fn new() -> Self {
        Self::load().unwrap_or_default()
    }

    pub fn load() -> Option<Self> {
        let path = Self::profile_path()?;
        if !path.exists() {
            return None;
        }
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }

    pub fn save(&self) -> Result<(), std::io::Error> {
        let path = Self::profile_path().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "Could not find profile path")
        })?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)
    }

    fn profile_path() -> Option<PathBuf> {
        data_dir().map(|p| p.join(PROFILE_PATH))
    }

    pub fn update_from_turn(&mut self, turn: &TurnMetrics) {
        let word_count = turn.word_count as f32;
        let turn_duration = turn.duration_secs;

        if word_count < 2.0 || turn_duration < 0.5 {
            return;
        }

        let current_wps = word_count / turn_duration;
        let current_pause_ms = turn.avg_pause_ms;

        self.speech_rate_wps = EMA_ALPHA * current_wps + (1.0 - EMA_ALPHA) * self.speech_rate_wps;
        self.avg_pause_ms = EMA_ALPHA * current_pause_ms + (1.0 - EMA_ALPHA) * self.avg_pause_ms;
        self.turns_observed += 1;

        if self.turns_observed >= MIN_CALIBRATION_TURNS {
            self.is_calibrated = true;
            self.is_fast_speaker = self.speech_rate_wps > FAST_SPEAKER_WPS;
        }

        self.last_speech_time = Some(chrono::Utc::now().to_rfc3339());

        let _ = self.save();
    }

    pub fn recommend_vad_silence_ms(&self) -> f32 {
        if !self.is_calibrated {
            return DEFAULT_VAD_SILENCE_MS;
        }

        let adaptive = (self.avg_pause_ms * 1.5).max(FAST_VAD_SILENCE_MS);
        adaptive.min(SLOW_VAD_SILENCE_MS)
    }

    pub fn recommend_step_ms(&self) -> u32 {
        if !self.is_calibrated {
            return DEFAULT_STEP_MS;
        }

        if self.is_fast_speaker {
            FAST_STEP_MS
        } else {
            DEFAULT_STEP_MS
        }
    }

    pub fn recommend_mode(&self) -> SttMode {
        // Use streaming mode (Moonshine) for ultra-low latency
        // Streaming reduces STT latency from 500-1500ms to 100-200ms
        // Trade-off: Less punctuation accuracy, but much faster response
        //
        // For formal/dictation use cases, could add a setting to force batch mode
        SttMode::Streaming
    }

    pub fn reset(&mut self) {
        self.turns_observed = 0;
        self.is_calibrated = false;
        self.is_fast_speaker = false;
        self.speech_rate_wps = 2.5;
        self.avg_pause_ms = DEFAULT_VAD_SILENCE_MS;
    }
}

#[derive(Debug, Clone)]
pub struct TurnMetrics {
    pub word_count: usize,
    pub duration_secs: f32,
    pub avg_pause_ms: f32,
    pub max_pause_ms: f32,
}

impl TurnMetrics {
    pub fn from_transcription(text: &str, duration_secs: f32, pauses: &[f32]) -> Self {
        let word_count = text.split_whitespace().count();
        let avg_pause_ms = if !pauses.is_empty() {
            pauses.iter().sum::<f32>() / pauses.len() as f32
        } else {
            duration_secs * 1000.0 / (word_count.max(1) as f32)
        };

        Self {
            word_count,
            duration_secs,
            avg_pause_ms,
            max_pause_ms: pauses.iter().cloned().fold(0.0f32, f32::max),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttMode {
    Batch,
    Streaming,
}

impl SttMode {
    pub fn as_str(&self) -> &'static str {
        match self {
            SttMode::Batch => "batch",
            SttMode::Streaming => "streaming",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "batch" => Some(SttMode::Batch),
            "streaming" => Some(SttMode::Streaming),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_profile() {
        let profile = SpeakerProfile::default();
        assert!(!profile.is_calibrated);
        assert_eq!(profile.turns_observed, 0);
        assert_eq!(profile.recommend_vad_silence_ms(), DEFAULT_VAD_SILENCE_MS);
    }

    #[test]
    fn test_fast_speaker_detection() {
        let mut profile = SpeakerProfile::default();

        for _ in 0..5 {
            profile.update_from_turn(&TurnMetrics {
                word_count: 20,
                duration_secs: 5.0,
                avg_pause_ms: 150.0,
                max_pause_ms: 300.0,
            });
        }

        assert!(profile.is_calibrated);
        assert!(profile.speech_rate_wps > 2.0);
    }

    #[test]
    fn test_stt_mode_from_str() {
        assert_eq!(SttMode::from_str("batch"), Some(SttMode::Batch));
        assert_eq!(SttMode::from_str("streaming"), Some(SttMode::Streaming));
        assert_eq!(SttMode::from_str("invalid"), None);
    }
}
