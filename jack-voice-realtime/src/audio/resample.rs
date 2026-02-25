//! Sample rate conversion
//!
//! Handles conversion between different sample rates (16kHz, 24kHz, 48kHz)

use super::{f32_to_pcm16, pcm16_to_f32, AudioFormat};

/// Resample audio from one sample rate to another
///
/// Uses linear interpolation for simplicity - can be upgraded to sinc/soxr
pub fn resample_pcm16(audio: &[u8], from_rate: u32, to_rate: u32) -> Vec<u8> {
    if from_rate == to_rate {
        return audio.to_vec();
    }

    // Convert to f32 first
    let f32_samples = pcm16_to_f32(audio);

    // Resample
    let resampled = resample_f32(&f32_samples, from_rate, to_rate);

    // Convert back
    f32_to_pcm16(&resampled)
}

/// Resample f32 audio
pub fn resample_f32(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let new_len = (samples.len() as f64 * ratio).ceil() as usize;

    let mut output = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 / ratio;
        let src_idx_floor = src_idx.floor() as usize;

        if src_idx_floor >= samples.len() - 1 {
            if src_idx_floor < samples.len() {
                output.push(samples[src_idx_floor]);
            }
            break;
        }

        let frac = src_idx - src_idx_floor as f64;
        let sample = samples[src_idx_floor] * (1.0 - frac as f32)
            + samples[src_idx_floor + 1] * (frac as f32);
        output.push(sample);
    }

    output
}

/// Convert from one format to another with optional resampling
pub fn convert_and_resample(
    audio: &[u8],
    from_format: AudioFormat,
    from_rate: u32,
    to_format: AudioFormat,
    to_rate: u32,
) -> Vec<u8> {
    // Convert to f32 at source rate first
    let f32_samples = match from_format {
        AudioFormat::PcmS16Le => pcm16_to_f32(audio),
        AudioFormat::F32Le => audio
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
    };

    // Resample if needed
    let resampled = resample_f32(&f32_samples, from_rate, to_rate);

    // Convert to target format
    match to_format {
        AudioFormat::PcmS16Le => f32_to_pcm16(&resampled),
        AudioFormat::F32Le => resampled.iter().flat_map(|&s| s.to_le_bytes()).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_same_rate() {
        let samples: Vec<u8> = vec![0x00, 0x00, 0xFF, 0x7F];

        let result = resample_pcm16(&samples, 16000, 16000);

        assert_eq!(samples, result);
    }

    #[test]
    fn test_resample_16k_to_24k() {
        // 1 second of 16kHz = 16000 samples = 32000 bytes
        let samples_16k: Vec<u8> = (0..32000).map(|i| (i % 256) as u8).collect();

        let samples_24k = resample_pcm16(&samples_16k, 16000, 24000);

        // Should be approximately 1.5x = 48000 bytes
        assert!(samples_24k.len() >= 47000);
        assert!(samples_24k.len() <= 49000);
    }

    #[test]
    fn test_resample_24k_to_16k() {
        // 1 second of 24kHz = 24000 samples = 48000 bytes
        let samples_24k: Vec<u8> = (0..48000).map(|i| (i % 256) as u8).collect();

        let samples_16k = resample_pcm16(&samples_24k, 24000, 16000);

        // Should be approximately 0.666x = 32000 bytes
        assert!(samples_16k.len() >= 31000);
        assert!(samples_16k.len() <= 33000);
    }

    #[test]
    fn test_resample_f32() {
        let samples: Vec<f32> = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        let upsampled = resample_f32(&samples, 1000, 2000);

        // Should be approximately 2x = 10 samples (may be 9 due to ceil)
        assert!(upsampled.len() >= 9 && upsampled.len() <= 10);
    }

    #[test]
    fn test_resample_preserves_trends() {
        // Linear ramp from -1 to 1
        let samples: Vec<f32> = (-50..50).map(|i| i as f32 / 50.0).collect();

        let downsampled = resample_f32(&samples, 100, 50);

        // First and last should still be approximately -1 and 1
        assert!((downsampled[0] - (-1.0)).abs() < 0.1);
        assert!((downsampled[downsampled.len() - 1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_convert_and_resample() {
        // PCM16 16kHz to F32 24kHz
        // 100 bytes = 50 samples at PCM16
        // 50 samples * 1.5 (16k->24k ratio) = 75 samples
        // 75 samples * 4 bytes = 300 bytes
        let pcm_16k: Vec<u8> = (0..100).map(|i| i as u8).collect();

        let f32_24k = convert_and_resample(
            &pcm_16k,
            AudioFormat::PcmS16Le,
            16000,
            AudioFormat::F32Le,
            24000,
        );

        // Should be approximately 300 bytes
        assert!(f32_24k.len() >= 200 && f32_24k.len() <= 400);
    }
}
