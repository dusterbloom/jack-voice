// Jack Desktop - Audio Quality Gate
// Single unified gate for all audio quality checks
// Consolidates 3 scattered energy checks into one principled system

/// Unified audio quality gate - single place for all energy validation
/// Learns noise floor for adaptive thresholds
#[derive(Debug, Clone)]
pub struct AudioQualityGate {
    // Thresholds (adaptive)
    min_rms: f32,       // Minimum RMS energy for speech (default 0.015)
    min_amplitude: f32, // Minimum peak amplitude (default 0.05)
    target_rms: f32,    // Normalization target (default 0.1)

    // Learned noise floor
    noise_floor_rms: f32, // Rolling average of silence RMS
    noise_samples: usize, // Number of silence samples collected

    // Hysteresis to prevent rapid state changes
    consecutive_speech_frames: usize,
    consecutive_silence_frames: usize,

    // Audio health tracking
    total_frames: usize,
    rejected_frames: usize,
    low_energy_streak: usize,
}

/// Result of audio quality check
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioQuality {
    /// Audio passes quality gate - suitable for STT
    Good,
    /// Audio is too quiet - likely silence or background noise
    TooQuiet,
    /// Audio is clipping - too loud
    Clipping,
    /// Audio has no content (all zeros)
    Empty,
}

impl Default for AudioQualityGate {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioQualityGate {
    /// Create a new audio quality gate with default thresholds
    pub fn new() -> Self {
        Self {
            min_rms: 0.015,         // Higher threshold reduces false positives
            min_amplitude: 0.05,    // Ensures actual voice, not background
            target_rms: 0.1,        // Standard normalization target
            noise_floor_rms: 0.002, // Lower initial estimate for faster adaptation
            noise_samples: 0,
            consecutive_speech_frames: 0,
            consecutive_silence_frames: 0,
            total_frames: 0,
            rejected_frames: 0,
            low_energy_streak: 0,
        }
    }

    /// Learning rate for noise floor adaptation (EMA alpha)
    const NOISE_FLOOR_ALPHA: f32 = 0.3; // Faster learning (was 0.1)

    /// Minimum frames to collect before adapting thresholds
    const MIN_CALIBRATION_FRAMES: usize = 20; // Faster calibration (was 50, ~600ms now)

    /// Frames required for hysteresis (prevents rapid state changes)
    const SPEECH_HYSTERESIS_FRAMES: usize = 2;
    const SILENCE_HYSTERESIS_FRAMES: usize = 3;

    /// Calculate RMS energy of audio samples
    pub fn calculate_rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    /// Calculate peak amplitude
    pub fn calculate_peak(samples: &[f32]) -> f32 {
        samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max)
    }

    /// Check if audio passes quality gate
    /// This is the ONLY place where audio quality should be checked
    pub fn check(&mut self, samples: &[f32]) -> AudioQuality {
        self.total_frames += 1;

        if samples.is_empty() {
            self.rejected_frames += 1;
            return AudioQuality::Empty;
        }

        let rms = Self::calculate_rms(samples);
        let peak = Self::calculate_peak(samples);

        // Check for clipping (>0.99 indicates potential clipping)
        if peak > 0.99 {
            return AudioQuality::Clipping;
        }

        // Adaptive threshold: use 3x noise floor as minimum
        let adaptive_min_rms = (self.noise_floor_rms * 3.0).max(self.min_rms);

        // Check energy thresholds
        let has_energy = rms >= adaptive_min_rms && peak >= self.min_amplitude;

        if has_energy {
            self.consecutive_speech_frames += 1;
            self.consecutive_silence_frames = 0;
            self.low_energy_streak = 0;
            AudioQuality::Good
        } else {
            self.consecutive_silence_frames += 1;
            self.consecutive_speech_frames = 0;
            self.low_energy_streak += 1;
            self.rejected_frames += 1;

            // Learn noise floor from silence samples
            self.learn_noise_floor(rms);

            AudioQuality::TooQuiet
        }
    }

    /// Check with hysteresis - requires multiple consecutive frames
    /// Use for barge-in detection to prevent false triggers
    pub fn check_with_hysteresis(&mut self, samples: &[f32]) -> AudioQuality {
        let quality = self.check(samples);

        match quality {
            AudioQuality::Good => {
                if self.consecutive_speech_frames >= Self::SPEECH_HYSTERESIS_FRAMES {
                    AudioQuality::Good
                } else {
                    AudioQuality::TooQuiet // Not enough consecutive frames yet
                }
            }
            AudioQuality::TooQuiet => {
                if self.consecutive_silence_frames >= Self::SILENCE_HYSTERESIS_FRAMES {
                    AudioQuality::TooQuiet
                } else {
                    // Keep previous state during transition
                    if self.consecutive_speech_frames > 0 {
                        AudioQuality::Good
                    } else {
                        AudioQuality::TooQuiet
                    }
                }
            }
            other => other,
        }
    }

    /// Learn noise floor from silence samples
    fn learn_noise_floor(&mut self, rms: f32) {
        // Only learn from very quiet samples (likely actual silence)
        if rms < self.min_rms * 0.5 {
            if self.noise_samples == 0 {
                self.noise_floor_rms = rms;
            } else {
                // Exponential moving average
                self.noise_floor_rms = Self::NOISE_FLOOR_ALPHA * rms
                    + (1.0 - Self::NOISE_FLOOR_ALPHA) * self.noise_floor_rms;
            }
            self.noise_samples += 1;
        }
    }

    /// Normalize audio to target RMS level
    /// Returns normalized samples, or original if normalization would over-amplify
    pub fn normalize(&self, samples: &[f32]) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        let current_rms = Self::calculate_rms(samples);

        // Don't amplify very quiet audio too much (likely noise)
        if current_rms < 0.001 {
            return samples.to_vec();
        }

        // Calculate gain (limit to prevent over-amplification)
        let gain = (self.target_rms / current_rms).min(10.0);

        // Apply gain and clip to prevent distortion
        samples
            .iter()
            .map(|s| (s * gain).clamp(-1.0, 1.0))
            .collect()
    }

    /// Get adaptive barge-in threshold (3x noise floor)
    pub fn barge_in_threshold(&self) -> f32 {
        (self.noise_floor_rms * 3.0).max(self.min_rms)
    }

    /// Check if we have enough data for adaptive thresholds
    pub fn is_calibrated(&self) -> bool {
        self.noise_samples >= Self::MIN_CALIBRATION_FRAMES
    }

    /// Get current noise floor estimate
    pub fn noise_floor(&self) -> f32 {
        self.noise_floor_rms
    }

    /// Get quality stats for diagnostics
    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "totalFrames": self.total_frames,
            "rejectedFrames": self.rejected_frames,
            "rejectionRate": if self.total_frames > 0 {
                self.rejected_frames as f32 / self.total_frames as f32
            } else {
                0.0
            },
            "noiseFloorRms": self.noise_floor_rms,
            "noiseSamples": self.noise_samples,
            "isCalibrated": self.is_calibrated(),
            "bargeInThreshold": self.barge_in_threshold(),
            "lowEnergyStreak": self.low_energy_streak,
        })
    }

    /// Reset the gate state (e.g., when mic changes)
    pub fn reset(&mut self) {
        self.noise_floor_rms = 0.005;
        self.noise_samples = 0;
        self.consecutive_speech_frames = 0;
        self.consecutive_silence_frames = 0;
        self.total_frames = 0;
        self.rejected_frames = 0;
        self.low_energy_streak = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_audio() {
        let mut gate = AudioQualityGate::new();
        assert_eq!(gate.check(&[]), AudioQuality::Empty);
    }

    #[test]
    fn test_silence_detection() {
        let mut gate = AudioQualityGate::new();
        let silence = vec![0.001f32; 512];
        assert_eq!(gate.check(&silence), AudioQuality::TooQuiet);
    }

    #[test]
    fn test_speech_detection() {
        let mut gate = AudioQualityGate::new();
        // Simulated speech with moderate amplitude
        let speech: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin() * 0.3).collect();
        assert_eq!(gate.check(&speech), AudioQuality::Good);
    }

    #[test]
    fn test_noise_floor_learning() {
        let mut gate = AudioQualityGate::new();

        // Feed silence samples
        let silence = vec![0.002f32; 512];
        for _ in 0..100 {
            gate.check(&silence);
        }

        assert!(gate.is_calibrated());
        assert!(gate.noise_floor() < 0.01);
    }

    #[test]
    fn test_normalization() {
        let gate = AudioQualityGate::new();
        let quiet = vec![0.01f32; 512];
        let normalized = gate.normalize(&quiet);

        let original_rms = AudioQualityGate::calculate_rms(&quiet);
        let normalized_rms = AudioQualityGate::calculate_rms(&normalized);

        // Should be amplified toward target
        assert!(normalized_rms > original_rms);
    }
}
