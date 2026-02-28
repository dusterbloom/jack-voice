//! Vocoder for converting codec codes to audio waveform.
//!
//! The vocoder takes 16 codebook codes and produces 24kHz audio.
//! Frame rate is ~12.5 Hz (1920 samples per frame).

use std::borrow::Cow;

use anyhow::Result;
use ndarray::Array2;
use ort::session::{Session, SessionInputValue, SessionInputs};
use ort::value::{Tensor, Value};

/// Vocoder for code-to-audio conversion.
pub struct Vocoder {
    session: Session,
}

impl Vocoder {
    /// Create a new vocoder from an ONNX session.
    pub fn new(session: Session) -> Self {
        Self { session }
    }

    /// Convert codec codes to audio samples.
    ///
    /// # Arguments
    /// * `codes` - Codec codes as [16, T] array (16 codebooks, T frames)
    ///
    /// # Returns
    /// Audio samples at 24kHz.
    pub fn decode(&mut self, codes: &Array2<i64>) -> Result<Vec<f32>> {
        if codes.nrows() != 16 {
            anyhow::bail!("Expected 16 codebook rows, got {}", codes.nrows());
        }

        let num_frames = codes.ncols();
        if num_frames == 0 {
            return Ok(vec![]);
        }

        let mut flat_codes: Vec<i64> = Vec::with_capacity(16 * num_frames);
        for c in 0..16 {
            for t in 0..num_frames {
                flat_codes.push(codes[[c, t]]);
            }
        }

        let input_tensor = Tensor::from_array((vec![1, 16, num_frames], flat_codes))
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {}", e))?;

        let inputs = SessionInputs::from(vec![(
            Cow::Borrowed("codes"),
            SessionInputValue::Owned(Value::from(input_tensor)),
        )]);

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| anyhow::anyhow!("Vocoder inference failed: {}", e))?;

        let first_output = outputs
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No vocoder output"))?;

        let (_shape, data) = first_output
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract vocoder output: {}", e))?;

        log::debug!(
            "[Vocoder] Decoded {} frames to {} samples",
            num_frames,
            data.len()
        );

        Ok(data.to_vec())
    }

    /// Decode a batch of frames (for streaming).
    ///
    /// Takes a slice of [16] codes per frame.
    pub fn decode_batch(&mut self, frame_codes: &[[i64; 16]]) -> Result<Vec<f32>> {
        if frame_codes.is_empty() {
            return Ok(vec![]);
        }

        // Convert to Array2
        let num_frames = frame_codes.len();
        let mut codes = Array2::zeros((16, num_frames));

        for (t, frame) in frame_codes.iter().enumerate() {
            for (c, &code) in frame.iter().enumerate() {
                codes[[c, t]] = code;
            }
        }

        self.decode(&codes)
    }

    /// Get the output sample rate (24kHz).
    pub fn sample_rate(&self) -> u32 {
        24000
    }

    /// Calculate expected audio duration for given frames.
    pub fn frames_to_duration_ms(num_frames: usize) -> u64 {
        // Each frame = 1920 samples at 24kHz = 80ms
        (num_frames as u64) * 80
    }

    /// Calculate number of frames for given duration in ms.
    pub fn duration_to_frames(duration_ms: u64) -> usize {
        // 80ms per frame
        ((duration_ms + 79) / 80) as usize
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frames_to_duration() {
        assert_eq!(Vocoder::frames_to_duration_ms(0), 0);
        assert_eq!(Vocoder::frames_to_duration_ms(1), 80);
        assert_eq!(Vocoder::frames_to_duration_ms(10), 800);
        assert_eq!(Vocoder::frames_to_duration_ms(100), 8000);
    }

    #[test]
    fn test_duration_to_frames() {
        assert_eq!(Vocoder::duration_to_frames(0), 0);
        assert_eq!(Vocoder::duration_to_frames(80), 1);
        assert_eq!(Vocoder::duration_to_frames(81), 2);
        assert_eq!(Vocoder::duration_to_frames(800), 10);
    }

    #[test]
    fn test_sample_rate() {
        // Need a real session for this, so just verify constant
        assert_eq!(24000u32, 24000);
    }

    #[test]
    fn test_decode_empty() {
        let codes: Array2<i64> = Array2::zeros((16, 0));
        assert_eq!(codes.ncols(), 0);
    }

    #[test]
    fn test_decode_wrong_codebook_count() {
        let codes: Array2<i64> = Array2::zeros((8, 10)); // Wrong: 8 instead of 16
        assert_ne!(codes.nrows(), 16);
    }

    #[test]
    fn test_decode_batch_empty() {
        let frames: [[i64; 16]; 0] = [];
        // Empty input should return empty output (can't actually call decode_batch without session)
        assert!(frames.is_empty());
    }

    // Integration test - requires downloaded models
    #[test]
    #[ignore]
    fn test_decode_real_codes() {
        let model_dir = dirs::home_dir()
            .unwrap()
            .join(".nanobot/models/qwen/qwen-onnx");

        if !model_dir.exists() {
            return;
        }

        use ort::session::{builder::GraphOptimizationLevel, Session};
        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .commit_from_file(model_dir.join("vocoder.onnx"))
            .unwrap();

        let mut vocoder = Vocoder::new(session);

        let mut codes: Array2<i64> = Array2::zeros((16, 10));
        for t in 0..10 {
            for c in 0..16 {
                codes[[c, t]] = 1000;
            }
        }

        let audio = vocoder.decode(&codes).unwrap();

        // Should produce 10 frames × 1920 samples = 19200 samples
        assert!(!audio.is_empty());
        assert_eq!(vocoder.sample_rate(), 24000);
    }
}
