// Jack Desktop - Voice Activity Detection
// Uses Silero VAD via sherpa-rs for speech detection
//
// VAD ROLE: Detects speech START only. Turn END is determined by Smart Turn.
// This is a semantic shift: VAD finds when user begins speaking,
// Smart Turn determines when they're finished.

use sherpa_rs::silero_vad::{SileroVad, SileroVadConfig};

use crate::models;

/// Minimum RMS energy to consider audio as potentially containing speech
/// Raised to filter keyboard/mechanical noise which has lower sustained energy than voice
const MIN_RMS_THRESHOLD: f32 = 0.008;
/// Minimum peak amplitude for speech consideration
/// Raised to reject transient clicks from keyboards and mouse
const MIN_AMPLITUDE_THRESHOLD: f32 = 0.04;

/// Pre-speech buffer duration — captures speech onset that VAD misses
/// Matches TurnDetector's LOOKBACK_SAMPLES (300ms)
#[allow(dead_code)]
const PRE_SPEECH_BUFFER_SECS: f32 = 0.30;
/// Pre-speech buffer size in samples at 16kHz = 0.30s * 16000 = 4800 samples
const PRE_SPEECH_BUFFER_SIZE: usize = 4800;

/// Speech segment detected by VAD
#[derive(Clone, Debug)]
pub struct SpeechSegment {
    pub samples: Vec<f32>,
    pub start_time: f32,
    pub end_time: f32,
}

pub struct VoiceActivityDetector {
    vad: SileroVad,
    sample_rate: u32,
    /// Rolling buffer of recent audio to prepend to speech segments
    /// This prevents clipping the start of words
    pre_speech_buffer: Vec<f32>,
}

/// Configurable VAD parameters for calibration
#[derive(Clone, Debug)]
pub struct VadConfig {
    pub sample_rate: u32,
    pub threshold: f32,
    pub min_silence_duration: f32,
    pub min_speech_duration: f32,
    pub window_size: i32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            threshold: 0.30, // Lowered from 0.45: energy gating handles noise, threshold catches soft consonants
            min_silence_duration: 2.5, // 1.75s - generous pause tolerance for natural speech
            min_speech_duration: 0.15, // Increased from 0.15s to prevent detecting brief noise as speech
            window_size: 512i32,
        }
    }
}

impl VoiceActivityDetector {
    /// Create a new VAD instance with default config
    pub fn new() -> Result<Self, VadError> {
        Self::with_config(VadConfig::default())
    }

    /// Create VAD with custom configuration (for calibration)
    pub fn with_config(config: VadConfig) -> Result<Self, VadError> {
        let model_path = models::get_model_path("silero_vad.onnx")?;

        if !model_path.exists() {
            return Err(VadError::ModelNotFound(model_path.display().to_string()));
        }

        let silero_config = SileroVadConfig {
            model: model_path.to_string_lossy().to_string(),
            sample_rate: config.sample_rate,
            threshold: config.threshold,
            min_silence_duration: config.min_silence_duration,
            min_speech_duration: config.min_speech_duration,
            window_size: config.window_size,
            ..Default::default()
        };

        log::info!(
            "VAD config: sample_rate={}, threshold={}, min_silence={}s, window_size={}",
            config.sample_rate,
            config.threshold,
            config.min_silence_duration,
            config.window_size
        );

        // Buffer up to 60 seconds of audio for speech detection
        let vad =
            SileroVad::new(silero_config, 60.0).map_err(|e| VadError::InitError(e.to_string()))?;

        Ok(Self {
            vad,
            sample_rate: config.sample_rate,
            pre_speech_buffer: Vec::with_capacity(PRE_SPEECH_BUFFER_SIZE),
        })
    }

    /// Update the pre-speech rolling buffer with new samples
    /// IMPORTANT: Only buffer audio when NOT in active speech
    /// This prevents prepending speech from a previous segment to the current one
    fn update_pre_speech_buffer(&mut self, samples: &[f32]) {
        // Only buffer audio if VAD is not currently detecting speech
        // This ensures pre-speech buffer only contains audio BEFORE speech started
        if !self.vad.is_speech() {
            self.pre_speech_buffer.extend_from_slice(samples);
            // Keep only the most recent PRE_SPEECH_BUFFER_SIZE samples
            if self.pre_speech_buffer.len() > PRE_SPEECH_BUFFER_SIZE {
                let excess = self.pre_speech_buffer.len() - PRE_SPEECH_BUFFER_SIZE;
                self.pre_speech_buffer.drain(0..excess);
            }
        }
        // When speech is active, we DON'T add to the buffer
        // The buffer will be used once and cleared when the segment is emitted
    }

    /// Calculate RMS energy and max amplitude of samples
    pub fn calculate_energy(samples: &[f32]) -> (f32, f32) {
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt();
        let max_amp = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        (rms, max_amp)
    }

    /// Check if model is ready
    pub fn is_ready(&self) -> bool {
        true // If we got here, the model loaded successfully
    }

    /// Process audio samples and return any completed speech segments
    pub fn process(&mut self, samples: &[f32]) -> Result<Option<SpeechSegment>, VadError> {
        // Update pre-speech buffer BEFORE processing (captures audio before VAD triggers)
        self.update_pre_speech_buffer(samples);

        // Calculate energy metrics for the incoming samples
        let (rms, max_amp) = Self::calculate_energy(samples);
        let has_energy = rms >= MIN_RMS_THRESHOLD && max_amp >= MIN_AMPLITUDE_THRESHOLD;

        // Log periodically to verify audio is flowing
        static PROCESS_COUNT: std::sync::atomic::AtomicUsize =
            std::sync::atomic::AtomicUsize::new(0);
        let count = PROCESS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count % 500 == 0 {
            // Reduced from 50 to prevent log spam
            log::debug!(
                "VAD process #{}: {} samples, max={:.4}, rms={:.4}, has_energy={}, is_speech={}",
                count,
                samples.len(),
                max_amp,
                rms,
                has_energy,
                self.vad.is_speech()
            );
        }

        // Always feed samples to VAD (it needs continuous audio for state tracking)
        self.vad.accept_waveform(samples.to_vec());

        // Check if there are any completed segments
        if !self.vad.is_empty() {
            let segment = self.vad.front();

            // Calculate energy of the completed segment
            let (seg_rms, seg_max) = Self::calculate_energy(&segment.samples);

            // Reject segments that are effectively silence (energy gating)
            if seg_rms < MIN_RMS_THRESHOLD || seg_max < MIN_AMPLITUDE_THRESHOLD {
                log::debug!(
                    "VAD: Rejecting low-energy segment (rms={:.4}, max={:.4})",
                    seg_rms,
                    seg_max
                );
                self.vad.pop();
                // Clear pre-speech buffer after rejecting segment
                self.pre_speech_buffer.clear();
                return Ok(None);
            }

            // Prepend pre-speech buffer to capture speech onset
            // This prevents clipping the beginning of words
            let mut full_samples = self.pre_speech_buffer.clone();
            full_samples.extend_from_slice(&segment.samples);

            // Adjust timing to account for pre-speech buffer
            let pre_buffer_duration = self.pre_speech_buffer.len() as f32 / self.sample_rate as f32;
            let start_time = (segment.start as f32 / self.sample_rate as f32) - pre_buffer_duration;
            let end_time = start_time + full_samples.len() as f32 / self.sample_rate as f32;

            log::debug!(
                "VAD: Prepending {:.0}ms pre-speech buffer ({} samples) to segment",
                pre_buffer_duration * 1000.0,
                self.pre_speech_buffer.len()
            );

            let result = SpeechSegment {
                samples: full_samples,
                start_time: start_time.max(0.0), // Clamp to avoid negative time
                end_time,
            };

            self.vad.pop();
            // Clear pre-speech buffer after using it
            self.pre_speech_buffer.clear();
            return Ok(Some(result));
        }

        Ok(None)
    }

    /// Check if currently detecting speech
    pub fn is_speech(&mut self) -> bool {
        self.vad.is_speech()
    }

    /// Check if currently detecting speech with energy validation
    /// More reliable for barge-in detection as it rejects false positives on silence
    pub fn is_speech_with_energy(&mut self, samples: &[f32]) -> bool {
        let (rms, max_amp) = Self::calculate_energy(samples);
        let has_energy = rms >= MIN_RMS_THRESHOLD && max_amp >= MIN_AMPLITUDE_THRESHOLD;
        has_energy && self.vad.is_speech()
    }

    /// Flush any remaining audio and return final segment if any
    pub fn flush(&mut self) -> Result<Option<SpeechSegment>, VadError> {
        self.vad.flush();

        if !self.vad.is_empty() {
            let segment = self.vad.front();

            // Prepend pre-speech buffer (same as process()) to prevent onset clipping
            let mut full_samples = self.pre_speech_buffer.clone();
            full_samples.extend_from_slice(&segment.samples);

            let pre_buffer_duration = self.pre_speech_buffer.len() as f32 / self.sample_rate as f32;
            let start_time = (segment.start as f32 / self.sample_rate as f32) - pre_buffer_duration;
            let end_time = start_time + full_samples.len() as f32 / self.sample_rate as f32;

            let result = SpeechSegment {
                samples: full_samples,
                start_time: start_time.max(0.0),
                end_time,
            };

            self.vad.pop();
            self.pre_speech_buffer.clear();
            return Ok(Some(result));
        }

        Ok(None)
    }

    /// Reset the VAD state
    pub fn reset(&mut self) {
        self.vad.clear();
        self.pre_speech_buffer.clear();
    }
}

#[derive(Debug, thiserror::Error)]
pub enum VadError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Initialization error: {0}")]
    InitError(String),
    #[error("Processing error: {0}")]
    ProcessingError(String),
}

impl From<crate::models::ModelError> for VadError {
    fn from(e: crate::models::ModelError) -> Self {
        VadError::ModelNotFound(e.to_string())
    }
}
