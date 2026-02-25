//! Audio buffer for streaming input
//!
//! Manages the input audio buffer for the Realtime API

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use super::{f32_to_pcm16, pcm16_to_f32};

/// Maximum buffer size (10 seconds at 24kHz)
const MAX_BUFFER_SAMPLES: usize = 240000;

/// Input audio buffer for streaming
pub struct AudioBuffer {
    buffer: VecDeque<i16>,
    sample_rate: u32,
    channels: u32,
    bytes_per_sample: usize,
    speech_started: Arc<AtomicBool>,
    speech_stopped: Arc<AtomicBool>,
    commit_requested: Arc<AtomicBool>,
    size_bytes: Arc<AtomicUsize>,
}

impl AudioBuffer {
    pub fn new(sample_rate: u32, channels: u32) -> Self {
        Self {
            buffer: VecDeque::new(),
            sample_rate,
            channels,
            bytes_per_sample: 2, // PCM16
            speech_started: Arc::new(AtomicBool::new(false)),
            speech_stopped: Arc::new(AtomicBool::new(false)),
            commit_requested: Arc::new(AtomicBool::new(false)),
            size_bytes: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Append raw PCM16 bytes to buffer
    pub fn append(&mut self, audio_data: &[u8]) -> Result<(), BufferError> {
        if audio_data.len() % self.bytes_per_sample != 0 {
            return Err(BufferError::InvalidDataLength);
        }

        let new_samples = audio_data.len() / self.bytes_per_sample;
        let new_bytes = new_samples * self.channels as usize * self.bytes_per_sample;

        // Check if we need to make room
        while self.size_bytes.load(Ordering::Relaxed) + new_bytes > MAX_BUFFER_SAMPLES {
            let remove_samples =
                (MAX_BUFFER_SAMPLES - new_bytes) / (self.channels as usize * self.bytes_per_sample);
            if remove_samples == 0 {
                return Err(BufferError::BufferFull);
            }
            self.buffer.drain(..remove_samples);
            self.size_bytes.fetch_sub(
                remove_samples * self.channels as usize * self.bytes_per_sample,
                Ordering::Relaxed,
            );
        }

        // Convert bytes to samples and add
        let samples: Vec<i16> = audio_data
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();

        for sample in samples {
            self.buffer.push_back(sample);
        }

        self.size_bytes.fetch_add(new_bytes, Ordering::Relaxed);
        self.speech_started.store(true, Ordering::Relaxed);

        Ok(())
    }

    /// Append decoded PCM16 samples
    pub fn append_samples(&mut self, samples: &[i16]) -> Result<(), BufferError> {
        let new_bytes = samples.len() * self.channels as usize * self.bytes_per_sample;

        while self.size_bytes.load(Ordering::Relaxed) + new_bytes > MAX_BUFFER_SAMPLES {
            let remove_samples =
                (MAX_BUFFER_SAMPLES - new_bytes) / (self.channels as usize * self.bytes_per_sample);
            if remove_samples == 0 {
                return Err(BufferError::BufferFull);
            }
            self.buffer.drain(..remove_samples);
            self.size_bytes.fetch_sub(
                remove_samples * self.channels as usize * self.bytes_per_sample,
                Ordering::Relaxed,
            );
        }

        for &sample in samples {
            self.buffer.push_back(sample);
        }

        self.size_bytes.fetch_add(new_bytes, Ordering::Relaxed);
        self.speech_started.store(true, Ordering::Relaxed);

        Ok(())
    }

    /// Get all buffered audio as PCM16 bytes
    pub fn get_bytes(&self) -> Vec<u8> {
        self.buffer.iter().flat_map(|&s| s.to_le_bytes()).collect()
    }

    /// Get all buffered audio as f32 samples
    pub fn get_f32(&self) -> Vec<f32> {
        self.buffer
            .iter()
            .map(|&s| s as f32 / i16::MAX as f32)
            .collect()
    }

    /// Get a copy of buffered audio as PCM16 bytes
    pub fn get(&self) -> Vec<i16> {
        self.buffer.iter().copied().collect()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.size_bytes.store(0, Ordering::Relaxed);
        self.speech_started.store(false, Ordering::Relaxed);
        self.speech_stopped.store(true, Ordering::Relaxed);
    }

    /// Request commit (for VAD)
    pub fn request_commit(&self) {
        self.commit_requested.store(true, Ordering::Relaxed);
    }

    /// Check if commit was requested
    pub fn commit_requested(&self) -> bool {
        self.commit_requested.load(Ordering::Relaxed)
    }

    /// Reset commit flag
    pub fn reset_commit_flag(&self) {
        self.commit_requested.store(false, Ordering::Relaxed);
    }

    /// Get duration in milliseconds
    pub fn duration_ms(&self) -> u64 {
        let samples = self.buffer.len() / self.channels as usize;
        (samples as u64 * 1000) / self.sample_rate as u64
    }

    /// Get number of samples
    pub fn sample_count(&self) -> usize {
        self.buffer.len() / self.channels as usize
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get buffer size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes.load(Ordering::Relaxed)
    }

    /// Check speech detection state
    pub fn is_speech_started(&self) -> bool {
        self.speech_started.load(Ordering::Relaxed)
    }

    /// Mark speech as stopped
    pub fn mark_speech_stopped(&self) {
        self.speech_stopped.store(true, Ordering::Relaxed);
    }

    /// Check if speech stopped
    pub fn is_speech_stopped(&self) -> bool {
        self.speech_stopped.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferError {
    InvalidDataLength,
    BufferFull,
}

impl std::fmt::Display for BufferError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BufferError::InvalidDataLength => write!(f, "Invalid data length"),
            BufferError::BufferFull => write!(f, "Buffer is full"),
        }
    }
}

impl std::error::Error for BufferError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_append() {
        let mut buffer = AudioBuffer::new(16000, 1);

        // 100ms of 16kHz audio = 1600 samples = 3200 bytes
        let audio: Vec<u8> = (0..3200).map(|i| (i % 256) as u8).collect();

        buffer.append(&audio).unwrap();

        assert_eq!(buffer.sample_count(), 1600);
        assert!(!buffer.is_empty());
    }

    #[test]
    fn test_buffer_clear() {
        let mut buffer = AudioBuffer::new(16000, 1);

        let audio: Vec<u8> = vec![0x00; 3200];
        buffer.append(&audio).unwrap();

        buffer.clear();

        assert!(buffer.is_empty());
        assert!(buffer.is_speech_stopped());
    }

    #[test]
    fn test_buffer_commit_flag() {
        let buffer = AudioBuffer::new(16000, 1);

        assert!(!buffer.commit_requested());

        buffer.request_commit();

        assert!(buffer.commit_requested());

        buffer.reset_commit_flag();

        assert!(!buffer.commit_requested());
    }

    #[test]
    fn test_buffer_duration() {
        let mut buffer = AudioBuffer::new(16000, 1);

        // 16000 samples = 1 second
        let samples: Vec<i16> = vec![0; 16000];
        buffer.append_samples(&samples).unwrap();

        assert_eq!(buffer.duration_ms(), 1000);
    }

    #[test]
    fn test_buffer_roundtrip() {
        let mut buffer = AudioBuffer::new(24000, 1);

        let original: Vec<i16> = (0..24000).map(|i| (i as i16) % 1000 - 500).collect();
        buffer.append_samples(&original).unwrap();

        let retrieved = buffer.get();

        assert_eq!(original.len(), retrieved.len());
    }

    #[test]
    fn test_buffer_f32_conversion() {
        let mut buffer = AudioBuffer::new(16000, 1);

        let original: Vec<i16> = vec![0, 1000, 0, -1000, 0];
        buffer.append_samples(&original).unwrap();

        let f32_samples = buffer.get_f32();

        assert!((f32_samples[1] - 0.03).abs() < 0.01);
        assert!((f32_samples[3] - (-0.03)).abs() < 0.01);
    }

    #[test]
    fn test_invalid_length() {
        let mut buffer = AudioBuffer::new(16000, 1);

        let result = buffer.append(&[0x00, 0x01, 0x02]); // odd length

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), BufferError::InvalidDataLength);
    }
}
