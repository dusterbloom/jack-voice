//! VAD (Voice Activity Detection) integration
//!
//! Wraps jack-voice VAD for use in the realtime pipeline

use jack_voice::{VadError, VoiceActivityDetector};

/// VAD wrapper for realtime pipeline
pub struct RealtimeVad {
    inner: Option<VoiceActivityDetector>,
    threshold: f32,
    silence_duration_ms: u32,
}

impl RealtimeVad {
    pub fn new() -> Result<Self, VadError> {
        let inner = VoiceActivityDetector::new()?;
        Ok(Self {
            inner: Some(inner),
            threshold: 0.5,
            silence_duration_ms: 500,
        })
    }

    pub fn with_config(threshold: f32, silence_duration_ms: u32) -> Result<Self, VadError> {
        let inner = VoiceActivityDetector::new()?;
        Ok(Self {
            inner: Some(inner),
            threshold,
            silence_duration_ms,
        })
    }

    /// Process audio samples and detect speech
    pub fn process(&mut self, samples: &[f32]) -> VadResult {
        let is_speech = self
            .inner
            .as_mut()
            .map(|vad| vad.is_speech_with_energy(samples))
            .unwrap_or(false);

        VadResult {
            is_speech,
            samples: samples.to_vec(),
            sample_rate: 16000, // jack-voice uses 16kHz
        }
    }

    /// Get silence duration threshold
    pub fn silence_duration_ms(&self) -> u32 {
        self.silence_duration_ms
    }
}

#[derive(Debug, Clone)]
pub struct VadResult {
    pub is_speech: bool,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_creation() {
        let vad = RealtimeVad::new();
        // May fail if model not available, that's ok for unit test
        if vad.is_ok() {
            assert!(vad.unwrap().silence_duration_ms() == 500);
        }
    }

    #[test]
    fn test_vad_with_config() {
        let vad = RealtimeVad::with_config(0.3, 1000);
        if vad.is_ok() {
            assert_eq!(vad.unwrap().silence_duration_ms(), 1000);
        }
    }

    #[test]
    fn test_vad_result() {
        let result = VadResult {
            is_speech: true,
            samples: vec![0.1, 0.2, 0.3],
            sample_rate: 16000,
        };

        assert!(result.is_speech);
        assert_eq!(result.samples.len(), 3);
    }
}
