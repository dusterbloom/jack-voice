//! Shared types for Qwen ONNX TTS.

/// Supported languages for TTS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    English,
    Chinese,
    Spanish,
    French,
    Japanese,
    Korean,
}

impl Language {
    /// Get the language name for token ID lookup.
    pub fn name(&self) -> &'static str {
        match self {
            Language::English => "english",
            Language::Chinese => "chinese",
            Language::Spanish => "spanish",
            Language::French => "french",
            Language::Japanese => "japanese",
            Language::Korean => "korean",
        }
    }

    /// Get language from ISO code.
    pub fn from_iso(code: &str) -> Option<Self> {
        match code.to_lowercase().as_str() {
            "en" | "english" => Some(Language::English),
            "zh" | "chinese" => Some(Language::Chinese),
            "es" | "spanish" => Some(Language::Spanish),
            "fr" | "french" => Some(Language::French),
            "ja" | "japanese" => Some(Language::Japanese),
            "ko" | "korean" => Some(Language::Korean),
            _ => None,
        }
    }
}

/// Preset speaker voices (CustomVoice model).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    Ryan,
    Serena,
    Vivian,
    Aiden,
    Eric,
    Dylan,
    UncleFu,
    OnoAnna,
    Sohee,
}

impl Speaker {
    /// Get speaker name for token ID lookup.
    pub fn name(&self) -> &'static str {
        match self {
            Speaker::Ryan => "ryan",
            Speaker::Serena => "serena",
            Speaker::Vivian => "vivian",
            Speaker::Aiden => "aiden",
            Speaker::Eric => "eric",
            Speaker::Dylan => "dylan",
            Speaker::UncleFu => "uncle_fu",
            Speaker::OnoAnna => "ono_anna",
            Speaker::Sohee => "sohee",
        }
    }

    /// Get speaker from name string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "ryan" => Some(Speaker::Ryan),
            "serena" => Some(Speaker::Serena),
            "vivian" => Some(Speaker::Vivian),
            "aiden" => Some(Speaker::Aiden),
            "eric" => Some(Speaker::Eric),
            "dylan" => Some(Speaker::Dylan),
            "uncle_fu" | "unclefu" => Some(Speaker::UncleFu),
            "ono_anna" | "onoanna" => Some(Speaker::OnoAnna),
            "sohee" => Some(Speaker::Sohee),
            _ => None,
        }
    }

    /// Get all available speakers.
    pub fn all() -> &'static [Speaker] {
        &[
            Speaker::Ryan,
            Speaker::Serena,
            Speaker::Vivian,
            Speaker::Aiden,
            Speaker::Eric,
            Speaker::Dylan,
            Speaker::UncleFu,
            Speaker::OnoAnna,
            Speaker::Sohee,
        ]
    }
}

/// Configuration for ONNX TTS synthesis.
#[derive(Debug, Clone)]
pub struct OnnxTtsConfig {
    /// Maximum frames to generate (default: 2000, ~160 seconds).
    pub max_frames: usize,
    /// Sampling configuration.
    pub sampling: super::SamplingConfig,
    /// Number of frames to batch before vocoding (default: 10, ~80ms).
    pub frames_per_chunk: usize,
}

impl Default for OnnxTtsConfig {
    fn default() -> Self {
        Self {
            max_frames: 2000,
            sampling: super::SamplingConfig::default(),
            frames_per_chunk: 10,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_name() {
        assert_eq!(Language::English.name(), "english");
        assert_eq!(Language::Chinese.name(), "chinese");
    }

    #[test]
    fn test_language_from_iso() {
        assert_eq!(Language::from_iso("en"), Some(Language::English));
        assert_eq!(Language::from_iso("zh"), Some(Language::Chinese));
        assert_eq!(Language::from_iso("unknown"), None);
    }

    #[test]
    fn test_speaker_name() {
        assert_eq!(Speaker::Ryan.name(), "ryan");
        assert_eq!(Speaker::Serena.name(), "serena");
        assert_eq!(Speaker::UncleFu.name(), "uncle_fu");
    }

    #[test]
    fn test_speaker_from_name() {
        assert_eq!(Speaker::from_name("ryan"), Some(Speaker::Ryan));
        assert_eq!(Speaker::from_name("SERENA"), Some(Speaker::Serena));
        assert_eq!(Speaker::from_name("unknown"), None);
    }

    #[test]
    fn test_speaker_all_count() {
        assert_eq!(Speaker::all().len(), 9);
    }

    #[test]
    fn test_onnx_tts_config_default() {
        let config = OnnxTtsConfig::default();
        assert_eq!(config.max_frames, 2000);
        assert_eq!(config.frames_per_chunk, 10);
    }
}
