//! Audio format conversion utilities
//!
//! Converts between:
//! - PCM16 LE (signed 16-bit)
//! - f32 LE (float32)
//! - base64 encoded variants

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};

/// Audio sample rate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleRate(pub u32);

impl SampleRate {
    pub const RATE_16K: Self = SampleRate(16000);
    pub const RATE_24K: Self = SampleRate(24000);
    pub const RATE_48K: Self = SampleRate(48000);
}

/// Audio format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioFormat {
    PcmS16Le,
    F32Le,
}

impl AudioFormat {
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            AudioFormat::PcmS16Le => 2,
            AudioFormat::F32Le => 4,
        }
    }
}

/// Convert PCM16 bytes to f32 samples (normalized to [-1.0, 1.0])
pub fn pcm16_to_f32(pcm_bytes: &[u8]) -> Vec<f32> {
    assert!(pcm_bytes.len() % 2 == 0, "PCM16 data must be even length");

    let samples: Vec<i16> = pcm_bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    samples
        .iter()
        .map(|&s| s as f32 / i16::MAX as f32)
        .collect()
}

/// Convert f32 samples to PCM16 bytes
pub fn f32_to_pcm16(samples: &[f32]) -> Vec<u8> {
    let pcm: Vec<i16> = samples
        .iter()
        .map(|&s| {
            let clamped = s.max(-1.0).min(1.0);
            (clamped * i16::MAX as f32) as i16
        })
        .collect();

    pcm.iter().flat_map(|&s| s.to_le_bytes()).collect()
}

/// Decode base64 PCM16 audio
pub fn decode_base64_pcm16(encoded: &str) -> Result<Vec<u8>, base64::DecodeError> {
    BASE64.decode(encoded)
}

/// Encode audio as base64
pub fn encode_base64_pcm16(pcm_bytes: &[u8]) -> String {
    BASE64.encode(pcm_bytes)
}

/// Decode base64 f32 audio
pub fn decode_base64_f32(encoded: &str) -> Result<Vec<f32>, base64::DecodeError> {
    let bytes = BASE64.decode(encoded)?;
    // Each f32 is 4 bytes
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    Ok(samples)
}

/// Encode f32 as base64
pub fn encode_base64_f32(samples: &[f32]) -> String {
    let bytes: Vec<u8> = samples
        .iter()
        .flat_map(|&s| {
            let bits = s.to_bits();
            bits.to_le_bytes()
        })
        .collect();
    BASE64.encode(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcm16_to_f32_conversion() {
        // Test known values
        let pcm: Vec<u8> = vec![
            0x00, 0x00, // 0
            0xFF, 0x7F, // 32767 (max positive)
            0x00, 0x80, // -32768 (min negative)
            0x00, 0x40, // 16384 = 0.5
        ];

        let f32 = pcm16_to_f32(&pcm);

        assert!((f32[0] - 0.0).abs() < 0.0001);
        assert!((f32[1] - 1.0).abs() < 0.0001);
        assert!((f32[2] - (-1.0)).abs() < 0.0001);
        assert!((f32[3] - 0.5).abs() < 0.0001);
    }

    #[test]
    fn test_f32_to_pcm16_conversion() {
        let f32_samples: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5];

        let pcm = f32_to_pcm16(&f32_samples);

        let reconstructed = pcm16_to_f32(&pcm);

        assert!((reconstructed[0] - 0.0).abs() < 0.0001);
        assert!((reconstructed[1] - 1.0).abs() < 0.0001);
        assert!((reconstructed[2] - (-1.0)).abs() < 0.0001);
        assert!((reconstructed[3] - 0.5).abs() < 0.0001);
        assert!((reconstructed[4] - (-0.5)).abs() < 0.0001);
    }

    #[test]
    fn test_roundtrip_pcm16() {
        let original: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let f32 = pcm16_to_f32(&original);
        let back = f32_to_pcm16(&f32);

        // Check first few values
        assert_eq!(original[0..4], back[0..4]);
    }

    #[test]
    fn test_base64_pcm16() {
        let original: Vec<u8> = vec![0x00, 0x01, 0x02, 0x03];

        let encoded = encode_base64_pcm16(&original);
        let decoded = decode_base64_pcm16(&encoded).unwrap();

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_base64_f32() {
        let original: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0];

        let encoded = encode_base64_f32(&original);
        let decoded = decode_base64_f32(&encoded).unwrap();

        for (o, d) in original.iter().zip(decoded.iter()) {
            assert!((o - d).abs() < 0.0001, "expected {}, got {}", o, d);
        }
    }

    #[test]
    fn test_silence() {
        let silence_pcm: Vec<u8> = vec![0x00; 100];
        let silence_f32 = pcm16_to_f32(&silence_pcm);

        for sample in &silence_f32 {
            assert!((sample - 0.0).abs() < 0.0001);
        }
    }

    #[test]
    fn test_clipping() {
        let over_max: Vec<f32> = vec![2.0, -2.0, 1.5];
        let clipped = f32_to_pcm16(&over_max);
        let reconstructed = pcm16_to_f32(&clipped);

        assert!((reconstructed[0] - 1.0).abs() < 0.0001);
        assert!((reconstructed[1] - (-1.0)).abs() < 0.0001);
        assert!((reconstructed[2] - 1.0).abs() < 0.0001);
    }
}
