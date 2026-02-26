//! TTS (Text-to-Speech) integration
//!
//! Wraps jack-voice TTS for use in the realtime pipeline

use jack_voice::{AudioOutput, TextToSpeech, TtsEngine, TtsError};

/// TTS wrapper for realtime pipeline
pub struct RealtimeTts {
    inner: Option<TextToSpeech>,
    engine: TtsEngine,
    voice: Option<String>,
    speed: f32,
}

impl RealtimeTts {
    pub fn new() -> Result<Self, TtsError> {
        let inner = TextToSpeech::new()?;
        Ok(Self {
            inner: Some(inner),
            engine: TtsEngine::Pocket,
            voice: None,
            speed: 1.0,
        })
    }

    pub fn with_engine(engine: TtsEngine) -> Result<Self, TtsError> {
        let inner = TextToSpeech::with_engine(engine.clone())?;
        Ok(Self {
            inner: Some(inner),
            engine,
            voice: None,
            speed: 1.0,
        })
    }

    /// Synthesize text (blocking)
    pub fn synthesize(&mut self, text: &str) -> Result<AudioOutput, TtsError> {
        match &mut self.inner {
            Some(tts) => {
                if let Some(voice) = &self.voice {
                    let _ = tts.set_speaker(voice);
                }
                tts.set_speed(self.speed);
                tts.synthesize(text)
            }
            None => Err(TtsError::InitError("TTS not initialized".to_string())),
        }
    }

    /// Synthesize with streaming callback
    pub fn synthesize_streaming<F>(&mut self, text: &str, mut callback: F) -> Result<u32, TtsError>
    where
        F: FnMut(&[f32], u32) -> bool,
    {
        match &mut self.inner {
            Some(tts) => {
                if let Some(voice) = &self.voice {
                    let _ = tts.set_speaker(voice);
                }
                tts.set_speed(self.speed);
                tts.synthesize_streaming(text, callback)
            }
            None => Err(TtsError::InitError("TTS not initialized".to_string())),
        }
    }

    /// Set voice
    pub fn set_voice(&mut self, voice: &str) -> Result<(), TtsError> {
        self.voice = Some(voice.to_string());
        if let Some(tts) = &mut self.inner {
            tts.set_speaker(voice)?;
        }
        Ok(())
    }

    /// Set speed
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    /// Get engine
    pub fn engine(&self) -> TtsEngine {
        self.engine.clone()
    }
}

/// TTS chunk event
#[derive(Debug, Clone)]
pub struct TtsChunk {
    pub index: u64,
    pub audio: Vec<u8>, // PCM16
    pub sample_rate: u32,
    pub duration_ms: u64,
}

impl TtsChunk {
    pub fn new(index: u64, samples: &[f32], sample_rate: u32) -> Self {
        let duration_ms = (samples.len() as u64 * 1000) / sample_rate as u64;

        // Convert f32 to PCM16
        let audio: Vec<i16> = samples
            .iter()
            .map(|&s| {
                let clamped = s.max(-1.0).min(1.0);
                (clamped * i16::MAX as f32) as i16
            })
            .collect();

        let audio_bytes: Vec<u8> = audio.iter().flat_map(|&s| s.to_le_bytes()).collect();

        Self {
            index,
            audio: audio_bytes,
            sample_rate,
            duration_ms,
        }
    }

    pub fn as_base64(&self) -> String {
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &self.audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_chunk_creation() {
        let samples: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0];

        let chunk = TtsChunk::new(0, &samples, 24000);

        assert_eq!(chunk.index, 0);
        assert_eq!(chunk.sample_rate, 24000);
        // 5 samples at 24kHz = ~0.2ms
        assert!(chunk.duration_ms <= 1);
    }

    #[test]
    fn test_tts_chunk_base64() {
        let samples: Vec<f32> = vec![0.0; 100];

        let chunk = TtsChunk::new(0, &samples, 16000);
        let encoded = chunk.as_base64();

        assert!(!encoded.is_empty());
    }
}
