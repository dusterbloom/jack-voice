//! STT (Speech-to-Text) integration
//!
//! Wraps jack-voice STT for use in the realtime pipeline

use jack_voice::{SpeechToText, SttError, SttMode, TranscriptionResult};

/// STT wrapper for realtime pipeline
pub struct RealtimeStt {
    inner: Option<SpeechToText>,
    mode: SttMode,
}

impl RealtimeStt {
    pub fn new(mode: SttMode) -> Result<Self, SttError> {
        let inner = SpeechToText::with_language(mode, None, None)?;
        Ok(Self {
            inner: Some(inner),
            mode,
        })
    }

    pub fn with_language(mode: SttMode, language: Option<String>) -> Result<Self, SttError> {
        let inner = SpeechToText::with_language(mode, language, None)?;
        Ok(Self {
            inner: Some(inner),
            mode,
        })
    }

    /// Transcribe audio samples
    pub fn transcribe(&mut self, samples: &[f32]) -> Result<TranscriptionResult, SttError> {
        match &mut self.inner {
            Some(stt) => stt.transcribe(samples),
            None => Err(SttError::ProcessingError("STT not initialized".to_string())),
        }
    }

    /// Get the current mode
    pub fn mode(&self) -> SttMode {
        self.mode
    }
}

/// Transcription event for streaming
#[derive(Debug, Clone)]
pub struct TranscriptionEvent {
    pub item_id: String,
    pub transcript: String,
    pub is_final: bool,
    pub delta: Option<String>,
}

impl TranscriptionEvent {
    pub fn partial(item_id: &str, delta: String) -> Self {
        Self {
            item_id: item_id.to_string(),
            transcript: delta.clone(),
            is_final: false,
            delta: Some(delta),
        }
    }

    pub fn final_transcript(item_id: &str, transcript: String) -> Self {
        Self {
            item_id: item_id.to_string(),
            transcript,
            is_final: true,
            delta: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcription_event_partial() {
        let event = TranscriptionEvent::partial("item_123", "Hello".to_string());

        assert_eq!(event.item_id, "item_123");
        assert!(!event.is_final);
        assert!(event.delta.is_some());
    }

    #[test]
    fn test_transcription_event_final() {
        let event = TranscriptionEvent::final_transcript("item_123", "Hello world".to_string());

        assert_eq!(event.item_id, "item_123");
        assert!(event.is_final);
        assert!(event.delta.is_none());
    }
}
