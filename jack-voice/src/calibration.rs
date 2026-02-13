// Jack Desktop - Voice Calibration System
// Guides users through tuning VAD/STT for their voice and environment

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::speaker::SttMode;
use crate::stt::SpeechToText;
use crate::vad::{VadConfig, VoiceActivityDetector};

/// Reference phrases for calibration (shorter than JFK for faster testing)
pub const CALIBRATION_PHRASES: &[(&str, &str)] = &[
    (
        "rainbow",
        "The rainbow is a division of white light into many beautiful colors.",
    ),
    (
        "quick_brown",
        "The quick brown fox jumps over the lazy dog.",
    ),
    (
        "north_wind",
        "The North Wind and the Sun were disputing which was the stronger.",
    ),
];

/// Default calibration phrase index
pub const DEFAULT_PHRASE_INDEX: usize = 0;

/// Current config version - bump when defaults change to force re-calibration
const CONFIG_VERSION: u32 = 2;

/// Voice pipeline configuration - all tunable parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    // Config version for migration
    #[serde(default = "default_version")]
    pub config_version: u32,

    // VAD settings
    pub vad_threshold: f32,      // Speech detection confidence (0.0-1.0)
    pub vad_min_silence_ms: u32, // Silence duration to end turn
    pub vad_min_speech_ms: u32,  // Minimum speech to trigger

    // Audio quality gates
    pub min_rms_threshold: f32,       // Minimum RMS energy
    pub min_amplitude_threshold: f32, // Minimum peak amplitude

    // STT settings
    pub preferred_stt_mode: String, // "streaming" or "batch"
    pub stt_step_ms: u32,           // Processing step size

    // Turn detection (Smart Turn v3 with fixed 0.5 threshold)
    pub smart_turn_enabled: bool, // Use Smart Turn model

    // Barge-in
    pub barge_in_enabled: bool,
    pub barge_in_cooldown_ms: u32,

    // Calibration metadata
    pub is_calibrated: bool,
    pub calibration_date: Option<String>,
    pub calibration_wer: Option<f32>,
}

fn default_version() -> u32 {
    0
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            // Config version
            config_version: CONFIG_VERSION,

            // VAD defaults (0.5 = Silero's recommended threshold)
            vad_threshold: 0.5,
            vad_min_silence_ms: 200, // 200ms - triggers SmartTurn quickly (Pipecat recommendation)
            vad_min_speech_ms: 150,

            // Audio quality defaults
            min_rms_threshold: 0.015,
            min_amplitude_threshold: 0.05,

            // STT defaults
            preferred_stt_mode: "streaming".to_string(),
            stt_step_ms: 500,

            // Turn detection
            smart_turn_enabled: true,

            // Barge-in
            barge_in_enabled: true,
            barge_in_cooldown_ms: 500,

            // Not calibrated by default
            is_calibrated: false,
            calibration_date: None,
            calibration_wer: None,
        }
    }
}

impl VoiceConfig {
    const CONFIG_PATH: &'static str = "jack-desktop/voice_config.json";

    pub fn load() -> Self {
        match Self::load_from_path() {
            Some(config) => {
                // Check version - force re-calibration if version mismatch
                if config.config_version < CONFIG_VERSION {
                    log::info!(
                        "Config version {} < {} - resetting to defaults",
                        config.config_version,
                        CONFIG_VERSION
                    );
                    return Self::default();
                }
                config
            }
            None => Self::default(),
        }
    }

    fn load_from_path() -> Option<Self> {
        let path = Self::config_path()?;
        if !path.exists() {
            return None;
        }
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }

    pub fn save(&self) -> Result<(), std::io::Error> {
        let path = Self::config_path().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "Could not find config path")
        })?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, data)
    }

    fn config_path() -> Option<PathBuf> {
        dirs::data_dir().map(|p| p.join(Self::CONFIG_PATH))
    }

    pub fn stt_mode(&self) -> SttMode {
        match self.preferred_stt_mode.as_str() {
            "batch" => SttMode::Batch,
            _ => SttMode::Streaming,
        }
    }
}

/// Result of a single calibration test run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationTestResult {
    pub vad_silence_ms: u32,
    pub segments_detected: usize,
    pub transcription: String,
    pub wer: f32,
    pub latency_ms: u64,
    pub is_fragmented: bool,
}

/// Full calibration session results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResults {
    pub reference_phrase: String,
    pub audio_duration_secs: f32,
    pub test_results: Vec<CalibrationTestResult>,
    pub recommended_config: VoiceConfig,
    pub best_wer: f32,
    pub best_silence_ms: u32,
}

/// Calibration session state
pub struct CalibrationSession {
    reference_text: String,
    audio_samples: Vec<f32>,
    sample_rate: u32,
}

impl CalibrationSession {
    pub fn new(phrase_index: usize) -> Self {
        let (_, text) = CALIBRATION_PHRASES
            .get(phrase_index)
            .unwrap_or(&CALIBRATION_PHRASES[0]);

        Self {
            reference_text: text.to_string(),
            audio_samples: Vec::new(),
            sample_rate: 16000,
        }
    }

    pub fn reference_text(&self) -> &str {
        &self.reference_text
    }

    pub fn set_audio(&mut self, samples: Vec<f32>, sample_rate: u32) {
        self.audio_samples = samples;
        self.sample_rate = sample_rate;
    }

    /// Run calibration with multiple VAD thresholds and find optimal settings
    pub fn run_calibration(&self) -> Result<CalibrationResults, CalibrationError> {
        if self.audio_samples.is_empty() {
            return Err(CalibrationError::NoAudio);
        }

        let silence_thresholds = [500, 800, 1000, 1200, 1500, 2000];
        let mut test_results = Vec::new();
        let audio_duration_secs = self.audio_samples.len() as f32 / self.sample_rate as f32;

        for &silence_ms in &silence_thresholds {
            match self.test_with_threshold(silence_ms) {
                Ok(result) => test_results.push(result),
                Err(e) => {
                    log::warn!("Calibration test failed for {}ms: {:?}", silence_ms, e);
                }
            }
        }

        if test_results.is_empty() {
            return Err(CalibrationError::AllTestsFailed);
        }

        // Find best result (lowest WER with single segment preferred)
        let best = test_results
            .iter()
            .filter(|r| !r.is_fragmented)
            .min_by(|a, b| a.wer.partial_cmp(&b.wer).unwrap())
            .or_else(|| {
                // If all fragmented, pick lowest WER anyway
                test_results
                    .iter()
                    .min_by(|a, b| a.wer.partial_cmp(&b.wer).unwrap())
            })
            .unwrap();

        // Extract values before moving test_results
        let best_wer = best.wer;
        let best_silence_ms = best.vad_silence_ms;
        let best_latency_ms = best.latency_ms;

        let mut recommended = VoiceConfig::default();
        recommended.vad_min_silence_ms = best_silence_ms;
        recommended.is_calibrated = true;
        recommended.calibration_date = Some(chrono::Utc::now().to_rfc3339());
        recommended.calibration_wer = Some(best_wer);

        // Adjust STT mode based on latency requirements
        if best_latency_ms > 500 {
            recommended.preferred_stt_mode = "streaming".to_string();
        }

        Ok(CalibrationResults {
            reference_phrase: self.reference_text.clone(),
            audio_duration_secs,
            test_results,
            recommended_config: recommended,
            best_wer,
            best_silence_ms,
        })
    }

    fn test_with_threshold(
        &self,
        silence_ms: u32,
    ) -> Result<CalibrationTestResult, CalibrationError> {
        // Create VAD with this threshold
        let vad_config = VadConfig {
            sample_rate: self.sample_rate,
            threshold: 0.5,
            min_silence_duration: silence_ms as f32 / 1000.0,
            min_speech_duration: 0.15,
            window_size: 512i32,
        };

        let mut vad = VoiceActivityDetector::with_config(vad_config)
            .map_err(|e| CalibrationError::VadError(e.to_string()))?;

        // Process audio through VAD in chunks
        let chunk_size = 512; // Match VAD window size
        let mut segments = Vec::new();

        for chunk in self.audio_samples.chunks(chunk_size) {
            if let Ok(Some(segment)) = vad.process(chunk) {
                segments.push(segment.samples);
            }
        }

        // Flush any remaining audio
        if let Ok(Some(segment)) = vad.flush() {
            segments.push(segment.samples);
        }

        let segments_detected = segments.len();
        let is_fragmented = segments_detected > 2; // More than 2 segments = fragmented

        // Transcribe all segments
        let start = std::time::Instant::now();
        let mut stt = SpeechToText::new(SttMode::Streaming)
            .map_err(|e| CalibrationError::SttError(e.to_string()))?;

        let mut combined_text = String::new();
        for segment_samples in &segments {
            if let Ok(result) = stt.transcribe(segment_samples) {
                if !result.text.is_empty() {
                    if !combined_text.is_empty() {
                        combined_text.push(' ');
                    }
                    combined_text.push_str(&result.text);
                }
            }
        }
        let latency_ms = start.elapsed().as_millis() as u64;

        // Calculate WER
        let wer = calculate_wer(&self.reference_text, &combined_text);

        Ok(CalibrationTestResult {
            vad_silence_ms: silence_ms,
            segments_detected,
            transcription: combined_text,
            wer,
            latency_ms,
            is_fragmented,
        })
    }
}

/// Calculate Word Error Rate using Levenshtein distance
pub fn calculate_wer(reference: &str, hypothesis: &str) -> f32 {
    // Normalize and tokenize reference
    let ref_lower = reference.to_lowercase();
    let ref_words: Vec<String> = ref_lower
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect();

    // Normalize and tokenize hypothesis
    let hyp_lower = hypothesis.to_lowercase();
    let hyp_words: Vec<String> = hyp_lower
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric())
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect();

    if ref_words.is_empty() {
        return if hyp_words.is_empty() { 0.0 } else { 1.0 };
    }

    let distance = levenshtein_distance(&ref_words, &hyp_words);
    distance as f32 / ref_words.len() as f32
}

fn levenshtein_distance<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let m = a.len();
    let n = b.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

#[derive(Debug, thiserror::Error)]
pub enum CalibrationError {
    #[error("No audio provided")]
    NoAudio,
    #[error("VAD error: {0}")]
    VadError(String),
    #[error("STT error: {0}")]
    SttError(String),
    #[error("All calibration tests failed")]
    AllTestsFailed,
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wer_perfect_match() {
        let wer = calculate_wer("hello world", "hello world");
        assert_eq!(wer, 0.0);
    }

    #[test]
    fn test_wer_case_insensitive() {
        let wer = calculate_wer("Hello World", "hello world");
        assert_eq!(wer, 0.0);
    }

    #[test]
    fn test_wer_with_substitution() {
        let wer = calculate_wer("hello world", "hello there");
        assert!((wer - 0.5).abs() < 0.01); // 1 substitution out of 2 words = 50%
    }

    #[test]
    fn test_wer_punctuation_ignored() {
        let wer = calculate_wer("Hello, world!", "hello world");
        assert_eq!(wer, 0.0);
    }

    #[test]
    fn test_default_config() {
        let config = VoiceConfig::default();
        assert!(!config.is_calibrated);
        assert_eq!(config.vad_min_silence_ms, 200);
    }

    #[test]
    fn test_calibration_phrases_exist() {
        assert!(!CALIBRATION_PHRASES.is_empty());
        for (id, text) in CALIBRATION_PHRASES {
            assert!(!id.is_empty());
            assert!(!text.is_empty());
        }
    }
}
