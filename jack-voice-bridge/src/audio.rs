use base64::Engine;
use serde::Deserialize;

use crate::protocol::{ErrorCode, RpcError};

pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 16_000;
pub const DEFAULT_CHANNELS: u16 = 1;
const MAX_AUDIO_B64_BYTES: usize = 6 * 1024 * 1024;

#[derive(Debug, Deserialize)]
pub struct AudioPayload {
    pub audio_b64: String,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub sample_rate_hz: Option<u32>,
    #[serde(default)]
    pub channels: Option<u16>,
}

impl AudioPayload {
    pub fn format_or_default(&self) -> &str {
        self.format.as_deref().unwrap_or("pcm_s16le")
    }

    pub fn sample_rate_hz_or_default(&self) -> u32 {
        self.sample_rate_hz.unwrap_or(DEFAULT_SAMPLE_RATE_HZ)
    }

    pub fn channels_or_default(&self) -> u16 {
        self.channels.unwrap_or(DEFAULT_CHANNELS)
    }
}

pub fn decode_audio_to_f32(payload: &AudioPayload) -> Result<Vec<f32>, RpcError> {
    if payload.sample_rate_hz_or_default() != DEFAULT_SAMPLE_RATE_HZ {
        return Err(RpcError::new(
            ErrorCode::InvalidParams,
            format!(
                "Unsupported sample_rate_hz {} (expected {})",
                payload.sample_rate_hz_or_default(),
                DEFAULT_SAMPLE_RATE_HZ
            ),
        ));
    }

    if payload.channels_or_default() != DEFAULT_CHANNELS {
        return Err(RpcError::new(
            ErrorCode::InvalidParams,
            format!(
                "Unsupported channels {} (expected {})",
                payload.channels_or_default(),
                DEFAULT_CHANNELS
            ),
        ));
    }

    if payload.audio_b64.len() > MAX_AUDIO_B64_BYTES {
        return Err(RpcError::new(
            ErrorCode::PayloadTooLarge,
            format!(
                "audio_b64 exceeds max size ({} > {})",
                payload.audio_b64.len(),
                MAX_AUDIO_B64_BYTES
            ),
        ));
    }

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.audio_b64.as_bytes())
        .map_err(|e| {
            RpcError::new(
                ErrorCode::AudioDecodeFailed,
                format!("Invalid base64 audio payload: {e}"),
            )
        })?;

    match payload.format_or_default().to_ascii_lowercase().as_str() {
        "pcm_s16le" => decode_pcm_s16le(&bytes),
        "f32le" => decode_f32le(&bytes),
        other => Err(RpcError::new(
            ErrorCode::UnsupportedAudioFormat,
            format!("Unsupported audio format '{other}'"),
        )),
    }
}

pub fn encode_f32le_to_base64(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * std::mem::size_of::<f32>());
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn decode_pcm_s16le(bytes: &[u8]) -> Result<Vec<f32>, RpcError> {
    if bytes.len() % 2 != 0 {
        return Err(RpcError::new(
            ErrorCode::AudioDecodeFailed,
            format!(
                "pcm_s16le payload must be divisible by 2 bytes, got {}",
                bytes.len()
            ),
        ));
    }

    let mut samples = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let value = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples.push((value as f32 / i16::MAX as f32).clamp(-1.0, 1.0));
    }
    Ok(samples)
}

fn decode_f32le(bytes: &[u8]) -> Result<Vec<f32>, RpcError> {
    if bytes.len() % 4 != 0 {
        return Err(RpcError::new(
            ErrorCode::AudioDecodeFailed,
            format!(
                "f32le payload must be divisible by 4 bytes, got {}",
                bytes.len()
            ),
        ));
    }

    let mut samples = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !value.is_finite() {
            return Err(RpcError::new(
                ErrorCode::AudioDecodeFailed,
                "f32le payload contains non-finite samples",
            ));
        }
        samples.push(value.clamp(-1.0, 1.0));
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::{decode_audio_to_f32, encode_f32le_to_base64, AudioPayload};
    use base64::Engine;

    #[test]
    fn decodes_pcm_s16le_with_defaults() {
        let mut bytes = Vec::new();
        for s in [-32768i16, 0, 32767] {
            bytes.extend_from_slice(&s.to_le_bytes());
        }

        let payload = AudioPayload {
            audio_b64: base64::engine::general_purpose::STANDARD.encode(bytes),
            format: None,
            sample_rate_hz: None,
            channels: None,
        };

        let samples = decode_audio_to_f32(&payload).expect("pcm decode failed");
        assert_eq!(samples.len(), 3);
        assert!(samples[0] <= -0.99);
        assert!(samples[1].abs() < 0.001);
        assert!(samples[2] >= 0.99);
    }

    #[test]
    fn decodes_f32le() {
        let mut bytes = Vec::new();
        for s in [0.25f32, -0.5f32, 0.75f32] {
            bytes.extend_from_slice(&s.to_le_bytes());
        }

        let payload = AudioPayload {
            audio_b64: base64::engine::general_purpose::STANDARD.encode(bytes),
            format: Some("f32le".to_string()),
            sample_rate_hz: Some(16_000),
            channels: Some(1),
        };

        let samples = decode_audio_to_f32(&payload).expect("f32 decode failed");
        assert_eq!(samples.len(), 3);
        assert!((samples[0] - 0.25).abs() < 0.0001);
        assert!((samples[1] + 0.5).abs() < 0.0001);
        assert!((samples[2] - 0.75).abs() < 0.0001);
    }

    #[test]
    fn rejects_invalid_pcm_byte_count() {
        let payload = AudioPayload {
            audio_b64: base64::engine::general_purpose::STANDARD.encode([1u8, 2u8, 3u8]),
            format: Some("pcm_s16le".to_string()),
            sample_rate_hz: Some(16_000),
            channels: Some(1),
        };

        let err = decode_audio_to_f32(&payload).expect_err("expected decode failure");
        assert_eq!(err.code.as_str(), "AUDIO_DECODE_FAILED");
    }

    #[test]
    fn rejects_unknown_format() {
        let payload = AudioPayload {
            audio_b64: base64::engine::general_purpose::STANDARD.encode([0u8, 0u8]),
            format: Some("wav".to_string()),
            sample_rate_hz: Some(16_000),
            channels: Some(1),
        };

        let err = decode_audio_to_f32(&payload).expect_err("expected format failure");
        assert_eq!(err.code.as_str(), "UNSUPPORTED_AUDIO_FORMAT");
    }

    #[test]
    fn rejects_non_default_sample_rate_or_channels() {
        let payload = AudioPayload {
            audio_b64: base64::engine::general_purpose::STANDARD.encode([0u8, 0u8]),
            format: Some("pcm_s16le".to_string()),
            sample_rate_hz: Some(8_000),
            channels: Some(2),
        };

        let err = decode_audio_to_f32(&payload).expect_err("expected invalid params");
        assert_eq!(err.code.as_str(), "INVALID_PARAMS");
    }

    #[test]
    fn encodes_f32le_base64() {
        let encoded = encode_f32le_to_base64(&[1.0f32, -1.0f32]);
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .expect("decode output");

        assert_eq!(bytes.len(), 8);
        assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_le_bytes(bytes[4..8].try_into().unwrap()), -1.0);
    }
}
