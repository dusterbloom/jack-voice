// Jack Desktop - Text-to-Speech
// Supports multiple TTS engines:
// - Auto (language-aware default selection)
// - Pocket (fast English, pure Rust Candle)
// - Supertonic (fast English)
// - Kokoro (local multilingual)

use pocket_tts::{ModelState as PocketModelState, TTSModel as PocketTtsModel};
use std::path::Path;
use supertonic::{TextToSpeech as SupertonicTts, VoiceStyleData};

use crate::kokoro_tts::KokoroTts;
use crate::models;

const POCKET_DEFAULT_VOICE: &str = "alba";
const POCKET_DEFAULT_VARIANT: &str = "english";
const KOKORO_DEFAULT_VOICE: &str = "0";
const SUPERTONIC_DEFAULT_VOICE: &str = "F1";

/// All known model variants with their language code and description.
const POCKET_VARIANTS: &[(&str, &str, &str)] = &[
    ("b6369a24", "en", "Legacy English"),
    ("english", "en", "English"),
    ("english_2026-01", "en", "English (Jan 2026)"),
    ("english_2026-04", "en", "English (Apr 2026)"),
    ("french_24l", "fr", "French (24-layer)"),
    ("german", "de", "German"),
    ("german_24l", "de", "German (24-layer)"),
    ("italian", "it", "Italian"),
    ("italian_24l", "it", "Italian (24-layer)"),
    ("portuguese", "pt", "Portuguese"),
    ("portuguese_24l", "pt", "Portuguese (24-layer)"),
    ("spanish", "es", "Spanish"),
    ("spanish_24l", "es", "Spanish (24-layer)"),
];

/// Default voice per language code.
const POCKET_LANG_DEFAULT_VOICES: &[(&str, &str)] = &[
    ("en", "alba"),
    ("fr", "estelle"),
    ("de", "juergen"),
    ("it", "giovanni"),
    ("pt", "rafael"),
    ("es", "lola"),
];
const POCKET_VOICES: &[&str] = &[
    "alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma",
    "estelle", "juergen", "giovanni", "rafael", "lola",
    "arnaud", "fabienne", "hans", "katja", "lucia", "marco", "miguel", "sofia",
    "alejandro", "camille", "giuseppe", "heinrich", "isabela",
];

/// Audio output from TTS
#[derive(Clone, Debug)]
pub struct AudioOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// TTS Engine type
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TtsEngine {
    Auto,
    Pocket,
    Supertonic,
    Kokoro,
}

/// Internal TTS implementation
enum TtsImpl {
    Pocket(PocketTts),
    Supertonic(SupertonicTts),
    Kokoro(KokoroTts),
}

struct PocketTts {
    model: PocketTtsModel,
    voice_state: PocketModelState,
    voice_id: String,
    variant: String,
}

#[derive(Debug)]
enum PocketVoiceInput<'a> {
    Preset(&'a str),
    VoiceCloneWav(&'a Path),
    PromptStateFile(&'a Path),
}

impl PocketTts {
    fn new_with_voice(variant: &str, voice_id: &str) -> Result<Self, anyhow::Error> {
        let model = PocketTtsModel::load(variant)?;
        let voice_state = load_pocket_voice_state(&model, voice_id, variant)?;

        Ok(Self {
            model,
            voice_state,
            voice_id: voice_id.to_string(),
            variant: variant.to_string(),
        })
    }

    fn set_voice(&mut self, voice_id: &str) -> Result<(), TtsError> {
        self.voice_state = load_pocket_voice_state(&self.model, voice_id, &self.variant)?;
        self.voice_id = voice_id.to_string();
        Ok(())
    }

    fn synthesize(&self, text: &str) -> Result<AudioOutput, TtsError> {
        let audio = self
            .model
            .generate(text, &self.voice_state)
            .map_err(|e| TtsError::SynthesisError(format!("Pocket synthesis failed: {}", e)))?;
        let channels = audio
            .to_vec2::<f32>()
            .map_err(|e| TtsError::SynthesisError(format!("Pocket output decode failed: {}", e)))?;
        let samples = channels.into_iter().next().unwrap_or_default();

        Ok(AudioOutput {
            samples,
            sample_rate: self.sample_rate(),
        })
    }

    fn synthesize_streaming<F>(&self, text: &str, on_chunk: &mut F) -> Result<u32, TtsError>
    where
        F: FnMut(&[f32], u32) -> bool,
    {
        for chunk in self.model.generate_stream_long(text, &self.voice_state) {
            let chunk = chunk.map_err(|e| {
                TtsError::SynthesisError(format!("Pocket streaming synthesis failed: {}", e))
            })?;
            let chunk = chunk.squeeze(0).map_err(|e| {
                TtsError::SynthesisError(format!("Pocket streaming chunk decode failed: {}", e))
            })?;
            let channels = chunk.to_vec2::<f32>().map_err(|e| {
                TtsError::SynthesisError(format!("Pocket streaming chunk decode failed: {}", e))
            })?;
            let samples = channels.into_iter().next().unwrap_or_default();

            if !on_chunk(&samples, self.sample_rate()) {
                break;
            }
        }

        Ok(self.sample_rate())
    }

    fn sample_rate(&self) -> u32 {
        self.model.sample_rate as u32
    }
}

fn pocket_voice_embedding_hf_path(voice_id: &str, variant: &str) -> String {
    if variant == "b6369a24" {
        format!(
            "hf://kyutai/pocket-tts-without-voice-cloning/embeddings/{}.safetensors",
            voice_id
        )
    } else {
        format!(
            "hf://kyutai/pocket-tts-without-voice-cloning/languages/{}/embeddings/{}.safetensors",
            variant, voice_id
        )
    }
}

fn load_pocket_voice_state(
    model: &PocketTtsModel,
    voice_id: &str,
    variant: &str,
) -> Result<PocketModelState, TtsError> {
    match classify_pocket_voice_input(voice_id)? {
        PocketVoiceInput::Preset(preset_voice_id) => {
            let prompt_hf_path = pocket_voice_embedding_hf_path(preset_voice_id, variant);
            let prompt_path =
                pocket_tts::weights::download_if_necessary(&prompt_hf_path).map_err(|e| {
                    TtsError::ModelNotFound(format!(
                        "Pocket voice '{}' download failed: {}",
                        preset_voice_id, e
                    ))
                })?;

            model
                .get_voice_state_from_prompt_file(&prompt_path)
                .map_err(|e| {
                    TtsError::InitError(format!(
                        "Pocket voice '{}' load failed: {}",
                        preset_voice_id, e
                    ))
                })
        }
        PocketVoiceInput::VoiceCloneWav(path) => model.get_voice_state(path).map_err(|e| {
            TtsError::InitError(format!(
                "Pocket voice cloning failed from '{}': {}",
                path.display(),
                e
            ))
        }),
        PocketVoiceInput::PromptStateFile(path) => {
            model.get_voice_state_from_prompt_file(path).map_err(|e| {
                TtsError::InitError(format!(
                    "Pocket prompt state load failed from '{}': {}",
                    path.display(),
                    e
                ))
            })
        }
    }
}

fn classify_pocket_voice_input<'a>(voice_id: &'a str) -> Result<PocketVoiceInput<'a>, TtsError> {
    let voice_id = voice_id.trim();
    if voice_id.is_empty() {
        return Err(TtsError::ModelNotFound(
            "Pocket voice cannot be empty".to_string(),
        ));
    }

    if POCKET_VOICES.contains(&voice_id) {
        return Ok(PocketVoiceInput::Preset(voice_id));
    }

    let path = Path::new(voice_id);
    if !path.exists() {
        return Err(TtsError::ModelNotFound(format!(
            "Unknown Pocket voice '{}'. Expected one of: {} or an existing .wav/.safetensors file path",
            voice_id,
            POCKET_VOICES.join(", ")
        )));
    }

    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match extension.as_str() {
        "wav" => Ok(PocketVoiceInput::VoiceCloneWav(path)),
        "safetensors" => Ok(PocketVoiceInput::PromptStateFile(path)),
        _ => Err(TtsError::ModelNotFound(format!(
            "Unsupported Pocket voice file '{}'. Expected .wav for cloning or .safetensors for prompt state",
            path.display()
        ))),
    }
}

fn pocket_variant_for_language(lang: &str) -> Result<(&'static str, &'static str), TtsError> {
    // Prefer 24-layer variants when available (higher quality)
    match lang {
        "en" | "english" => Ok(("english", "alba")),
        "fr" | "french" => Ok(("french_24l", "estelle")),
        "de" | "german" => Ok(("german_24l", "juergen")),
        "it" | "italian" => Ok(("italian_24l", "giovanni")),
        "pt" | "portuguese" => Ok(("portuguese_24l", "rafael")),
        "es" | "spanish" => Ok(("spanish_24l", "lola")),
        _ => Err(TtsError::InitError(format!(
            "Unsupported Pocket TTS language: {}",
            lang
        ))),
    }
}

fn pocket_default_voice_for_variant(variant: &str) -> &'static str {
    let lang = POCKET_VARIANTS
        .iter()
        .find(|(v, _, _)| *v == variant)
        .map(|(_, l, _)| *l)
        .unwrap_or("en");
    POCKET_LANG_DEFAULT_VOICES
        .iter()
        .find(|(l, _)| *l == lang)
        .map(|(_, v)| *v)
        .unwrap_or("alba")
}

#[cfg(test)]
mod tests {
    use super::{
        classify_pocket_voice_input, pocket_default_voice_for_variant, pocket_variant_for_language,
        PocketVoiceInput, TextToSpeech, TtsEngine, TtsError,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn pocket_variant_for_language_maps_known_languages() {
        let cases = [
            ("en", "english", "alba"),
            ("english", "english", "alba"),
            ("fr", "french_24l", "estelle"),
            ("french", "french_24l", "estelle"),
            ("de", "german_24l", "juergen"),
            ("it", "italian_24l", "giovanni"),
            ("pt", "portuguese_24l", "rafael"),
            ("es", "spanish_24l", "lola"),
        ];
        for (lang, expected_variant, expected_voice) in cases {
            let (variant, voice) =
                pocket_variant_for_language(lang).expect("known language should succeed");
            assert_eq!(variant, expected_variant, "variant mismatch for lang={}", lang);
            assert_eq!(voice, expected_voice, "voice mismatch for lang={}", lang);
        }
    }

    #[test]
    fn pocket_variant_for_language_rejects_unknown() {
        let err =
            pocket_variant_for_language("zz").expect_err("unknown language should fail");
        match err {
            TtsError::InitError(msg) => {
                assert!(msg.contains("Unsupported"), "unexpected message: {}", msg);
            }
            other => panic!("expected InitError, got: {:?}", other),
        }
    }

    #[test]
    fn pocket_default_voice_for_variant_returns_expected_defaults() {
        assert_eq!(pocket_default_voice_for_variant("english"), "alba");
        assert_eq!(pocket_default_voice_for_variant("french_24l"), "estelle");
        assert_eq!(pocket_default_voice_for_variant("german_24l"), "juergen");
        assert_eq!(pocket_default_voice_for_variant("italian_24l"), "giovanni");
        assert_eq!(pocket_default_voice_for_variant("portuguese_24l"), "rafael");
        assert_eq!(pocket_default_voice_for_variant("spanish_24l"), "lola");
    }

    #[test]
    fn pocket_default_voice_for_variant_falls_back_to_alba() {
        // Unknown variant should fall back to "alba"
        assert_eq!(pocket_default_voice_for_variant("nonexistent"), "alba");
    }

    #[test]
    fn available_pocket_languages_returns_six_entries() {
        let langs = TextToSpeech::available_pocket_languages();
        assert_eq!(langs.len(), 6, "expected 6 languages");
        let codes: Vec<&str> = langs.iter().map(|(code, _)| *code).collect();
        assert!(codes.contains(&"en"));
        assert!(codes.contains(&"fr"));
        assert!(codes.contains(&"de"));
        assert!(codes.contains(&"it"));
        assert!(codes.contains(&"pt"));
        assert!(codes.contains(&"es"));
    }

    fn unique_temp_file(ext: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join("jack-voice-tests");
        fs::create_dir_all(&dir).expect("failed to create temp test dir");

        let path = dir.join(format!("pocket-voice-{nanos}.{ext}"));
        fs::write(&path, b"test").expect("failed to write temp test file");
        path
    }

    #[test]
    fn classify_pocket_voice_input_accepts_preset() {
        match classify_pocket_voice_input("alba").expect("preset should be accepted") {
            PocketVoiceInput::Preset("alba") => {}
            _ => panic!("expected preset voice classification"),
        }
    }

    #[test]
    fn classify_pocket_voice_input_accepts_wav_file() {
        let path = unique_temp_file("WAV");
        let voice = path.to_string_lossy().to_string();

        match classify_pocket_voice_input(&voice).expect("wav path should be accepted") {
            PocketVoiceInput::VoiceCloneWav(classified_path) => {
                assert_eq!(classified_path, path.as_path());
            }
            _ => panic!("expected wav voice clone path classification"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn classify_pocket_voice_input_accepts_prompt_state_file() {
        let path = unique_temp_file("safetensors");
        let voice = path.to_string_lossy().to_string();

        match classify_pocket_voice_input(&voice).expect("prompt path should be accepted") {
            PocketVoiceInput::PromptStateFile(classified_path) => {
                assert_eq!(classified_path, path.as_path());
            }
            _ => panic!("expected prompt state file classification"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn classify_pocket_voice_input_rejects_unknown_voice() {
        let err = classify_pocket_voice_input("not-a-real-voice")
            .expect_err("unknown non-path voice should fail");
        match err {
            TtsError::ModelNotFound(message) => {
                assert!(message.contains("Unknown Pocket voice"));
            }
            _ => panic!("expected model-not-found error"),
        }
    }

    #[test]
    fn classify_pocket_voice_input_rejects_unsupported_file_extension() {
        let path = unique_temp_file("txt");
        let voice = path.to_string_lossy().to_string();

        let err = classify_pocket_voice_input(&voice).expect_err("txt file should be rejected");
        match err {
            TtsError::ModelNotFound(message) => {
                assert!(message.contains("Unsupported Pocket voice file"));
            }
            _ => panic!("expected model-not-found error"),
        }

        let _ = fs::remove_file(path);
    }

    #[test]
    fn tts_engine_all_variants_distinct() {
        let engines = vec![
            TtsEngine::Auto,
            TtsEngine::Pocket,
            TtsEngine::Supertonic,
            TtsEngine::Kokoro,
        ];
        for (i, e1) in engines.iter().enumerate() {
            for (j, e2) in engines.iter().enumerate() {
                if i != j {
                    assert_ne!(e1, e2, "Engine variants should be distinct");
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn sine_wave_24khz_plays_clean() {
        use crate::AudioPlayer;
        let player = AudioPlayer::new().expect("Failed to create AudioPlayer");
        let sample_rate = 24000u32;
        let duration_secs = 1.0f32;
        let freq = 440.0f32;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3
            })
            .collect();
        eprintln!("Playing 440Hz sine wave at 24kHz for 1 second...");
        player.play(samples, sample_rate);
        player.wait();
        eprintln!("Done. Did you hear a clean 440Hz tone?");
    }

    #[test]
    #[ignore]
    fn sine_wave_44100hz_plays_clean() {
        use crate::AudioPlayer;
        let player = AudioPlayer::new().expect("Failed to create AudioPlayer");
        let sample_rate = 44100u32;
        let duration_secs = 1.0f32;
        let freq = 440.0f32;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.3
            })
            .collect();
        eprintln!("Playing 440Hz sine wave at 44100Hz for 1 second...");
        player.play(samples, sample_rate);
        player.wait();
        eprintln!("Done. Did you hear a clean 440Hz tone?");
    }

    #[test]
    fn normalize_language_collapses_locale_tags() {
        assert_eq!(
            super::normalize_language(Some("it-IT")),
            Some("it".to_string())
        );
        assert_eq!(
            super::normalize_language(Some("FR_fr")),
            Some("fr".to_string())
        );
        assert_eq!(
            super::normalize_language(Some(" en ")),
            Some("en".to_string())
        );
        assert_eq!(super::normalize_language(Some("   ")), None);
    }

    #[test]
    fn auto_candidates_for_english_prefer_english_stack() {
        assert_eq!(
            TextToSpeech::auto_candidates(Some("en")),
            vec![TtsEngine::Pocket, TtsEngine::Supertonic, TtsEngine::Kokoro]
        );
        assert_eq!(
            TextToSpeech::auto_candidates(None),
            vec![TtsEngine::Pocket, TtsEngine::Supertonic, TtsEngine::Kokoro]
        );
    }

    #[test]
    fn auto_candidates_for_unsupported_multilingual_fallback_to_kokoro() {
        assert_eq!(
            TextToSpeech::auto_candidates(Some("es")),
            vec![TtsEngine::Kokoro]
        );
    }

    #[test]
    fn auto_candidates_for_italian_returns_kokoro_only() {
        assert_eq!(
            TextToSpeech::auto_candidates(Some("it")),
            vec![TtsEngine::Kokoro]
        );
    }

    #[test]
    fn kokoro_voices_cover_all_53_speakers() {
        let voices = TextToSpeech::available_kokoro_voices();
        assert_eq!(voices.len(), 53, "Kokoro v1.0 has 53 voices across 9 languages");

        // Verify language diversity
        let languages: std::collections::HashSet<&str> =
            voices.iter().map(|v| v.language.as_str()).collect();
        for lang in &["en-us", "en-gb", "es", "fr", "hi", "it", "ja", "pt", "zh"] {
            assert!(languages.contains(lang), "Missing language: {}", lang);
        }

        // Spot check specific voices
        assert!(voices.iter().any(|v| v.id == 35 && v.name == "if_sara"));
        assert!(voices.iter().any(|v| v.id == 36 && v.name == "im_nicola"));
        assert!(voices.iter().any(|v| v.id == 0 && v.name == "af_alloy"));
    }

}

pub struct TextToSpeech {
    engine: TtsImpl,
    requested_engine: TtsEngine,
    resolved_engine: TtsEngine,
    speaker_id: String,
    speed: f32,
    sample_rate: u32,
    language: Option<String>,
}

impl TtsEngine {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Pocket => "pocket",
            Self::Supertonic => "supertonic",
            Self::Kokoro => "kokoro",
        }
    }
}

impl TextToSpeech {
    /// Create a new TTS instance with language-aware auto selection.
    pub fn new() -> Result<Self, TtsError> {
        Self::with_engine(TtsEngine::Auto)
    }

    /// Create TTS with specific engine
    pub fn with_engine(engine_type: TtsEngine) -> Result<Self, TtsError> {
        match engine_type {
            TtsEngine::Auto => Self::new_auto(),
            TtsEngine::Pocket => Self::new_pocket(),
            TtsEngine::Supertonic => Self::new_supertonic(),
            TtsEngine::Kokoro => Self::new_kokoro(),
        }
    }

    fn new_auto() -> Result<Self, TtsError> {
        Self::build_auto(None)
    }

    /// Create Pocket TTS instance
    fn new_pocket() -> Result<Self, TtsError> {
        Self::new_pocket_with_voice(POCKET_DEFAULT_VOICE)
    }

    /// Create Pocket TTS instance with specific preset voice (defaults to "english" variant)
    pub fn new_pocket_with_voice(voice_id: &str) -> Result<Self, TtsError> {
        Self::new_pocket_with_variant_and_voice(POCKET_DEFAULT_VARIANT, voice_id)
    }

    /// Create Pocket TTS instance for a specific language (picks best variant and default voice)
    pub fn new_pocket_with_language(lang: &str) -> Result<Self, TtsError> {
        let (variant, default_voice) = pocket_variant_for_language(lang)?;
        Self::new_pocket_with_variant_and_voice(variant, default_voice)
    }

    /// Create Pocket TTS instance with a specific variant (uses default voice for that variant's language)
    pub fn new_pocket_with_variant(variant: &str) -> Result<Self, TtsError> {
        let default_voice = pocket_default_voice_for_variant(variant);
        Self::new_pocket_with_variant_and_voice(variant, default_voice)
    }

    /// Create Pocket TTS instance with a specific variant and voice
    pub fn new_pocket_with_variant_and_voice(variant: &str, voice_id: &str) -> Result<Self, TtsError> {
        let pocket = PocketTts::new_with_voice(variant, voice_id)
            .map_err(|e| TtsError::InitError(format!("Pocket TTS ({}): {}", variant, e)))?;
        let sample_rate = pocket.sample_rate();
        let lang = POCKET_VARIANTS.iter()
            .find(|(v, _, _)| *v == variant)
            .map(|(_, l, _)| Some(l.to_string()))
            .unwrap_or(Some("en".to_string()));

        Ok(Self {
            engine: TtsImpl::Pocket(pocket),
            requested_engine: TtsEngine::Pocket,
            resolved_engine: TtsEngine::Pocket,
            speaker_id: voice_id.to_string(),
            speed: 1.0,
            sample_rate,
            language: lang,
        })
    }

    /// Create Supertonic TTS instance
    fn new_supertonic() -> Result<Self, TtsError> {
        let paths = models::get_supertonic_paths()?;
        Self::with_supertonic_paths(&paths)
    }

    /// Create Kokoro TTS instance
    fn new_kokoro() -> Result<Self, TtsError> {
        Self::new_kokoro_with_voice(KOKORO_DEFAULT_VOICE)
    }

    /// Create Kokoro TTS instance with specific voice
    pub fn new_kokoro_with_voice(voice_id: &str) -> Result<Self, TtsError> {
        let voice_num = voice_id.parse::<i32>().unwrap_or(0);
        let language = crate::kokoro_tts::voice_id_to_language(voice_num);
        Self::build(
            TtsEngine::Kokoro,
            TtsEngine::Kokoro,
            voice_id.to_string(),
            Some(language),
        )
    }

    /// Create TTS with specific Supertonic model paths
    pub fn with_supertonic_paths(paths: &models::SupertonicPaths) -> Result<Self, TtsError> {
        let (engine, sample_rate) = build_supertonic(paths, SUPERTONIC_DEFAULT_VOICE)?;
        Ok(Self {
            engine,
            requested_engine: TtsEngine::Supertonic,
            resolved_engine: TtsEngine::Supertonic,
            speaker_id: SUPERTONIC_DEFAULT_VOICE.to_string(),
            speed: 1.0,
            sample_rate,
            language: Some("en".to_string()),
        })
    }

    fn build(
        resolved_engine: TtsEngine,
        requested_engine: TtsEngine,
        speaker_id: String,
        language: Option<&str>,
    ) -> Result<Self, TtsError> {
        let language = normalize_language(language);
        let (engine, sample_rate) =
            build_engine(&resolved_engine, &speaker_id, language.as_deref())?;
        Ok(Self {
            engine,
            requested_engine,
            resolved_engine,
            speaker_id,
            speed: 1.0,
            sample_rate,
            language,
        })
    }

    fn build_auto(language: Option<&str>) -> Result<Self, TtsError> {
        let language = normalize_language(language);
        let mut last_error = None;

        for resolved_engine in Self::auto_candidates(language.as_deref()) {
            let speaker_id = default_voice_for_engine(&resolved_engine, language.as_deref());
            match build_engine(&resolved_engine, &speaker_id, language.as_deref()) {
                Ok((engine, sample_rate)) => {
                    return Ok(Self {
                        engine,
                        requested_engine: TtsEngine::Auto,
                        resolved_engine,
                        speaker_id,
                        speed: 1.0,
                        sample_rate,
                        language,
                    });
                }
                Err(err) => {
                    log::warn!(
                        "[TTS] auto init fallback: {} failed for language {:?}: {}",
                        resolved_engine.as_str(),
                        language,
                        err
                    );
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            TtsError::InitError("No available TTS engine for auto mode".to_string())
        }))
    }

    fn auto_candidates(language: Option<&str>) -> Vec<TtsEngine> {
        match normalize_language(language) {
            Some(lang) if lang != "en" => vec![TtsEngine::Kokoro],
            _ => vec![TtsEngine::Pocket, TtsEngine::Supertonic, TtsEngine::Kokoro],
        }
    }

    fn apply_language_to_engine(&mut self) -> Result<(), TtsError> {
        let language = self.language.clone();
        if let (TtsImpl::Kokoro(kokoro), Some(language)) = (&mut self.engine, language.as_deref()) {
            kokoro
                .set_language(kokoro_language(language))
                .map_err(|e| TtsError::InitError(format!("Kokoro language: {}", e)))?;
        }
        Ok(())
    }

    /// Check if model is ready
    pub fn is_ready(&self) -> bool {
        true
    }

    pub fn set_language(&mut self, language: &str) -> Result<(), TtsError> {
        self.language = normalize_language(Some(language));
        if self.requested_engine == TtsEngine::Auto {
            let candidates = Self::auto_candidates(self.language.as_deref());
            if !candidates
                .iter()
                .any(|engine| *engine == self.resolved_engine)
            {
                let replacement = Self::build_auto(self.language.as_deref())?;
                self.engine = replacement.engine;
                self.resolved_engine = replacement.resolved_engine;
                self.speaker_id = replacement.speaker_id;
                self.sample_rate = replacement.sample_rate;
                if let TtsImpl::Supertonic(tts) = &mut self.engine {
                    tts.set_speed(self.speed);
                }
            } else {
                self.apply_language_to_engine()?;
            }
            return Ok(());
        }

        self.apply_language_to_engine()
    }

    /// Set speaker voice.
    /// Pocket accepts preset names (`alba`, `marius`, etc.) or local `.wav`/`.safetensors` paths.
    /// Supertonic accepts voice IDs like `F1`/`M2`; Kokoro accepts numeric IDs as strings.
    pub fn set_speaker(&mut self, speaker_id: &str) -> Result<(), TtsError> {
        match &mut self.engine {
            TtsImpl::Pocket(pocket) => {
                pocket.set_voice(speaker_id)?;
            }
            TtsImpl::Supertonic(tts) => {
                let paths = models::get_supertonic_paths()?;
                let voice_path = paths.voice_path(speaker_id);
                if !voice_path.exists() {
                    return Err(TtsError::ModelNotFound(format!(
                        "Voice file not found: {}",
                        voice_path.display()
                    )));
                }
                let voice_data =
                    VoiceStyleData::from_json_file(&voice_path, speaker_id, speaker_id)
                        .map_err(|e| TtsError::InitError(format!("Failed to load voice: {}", e)))?;
                tts.set_voice_style(&voice_data);
            }
            TtsImpl::Kokoro(kokoro) => {
                let voice_id = speaker_id.parse::<i32>().map_err(|_| {
                    TtsError::InitError(format!("Invalid Kokoro speaker ID: {}", speaker_id))
                })?;
                kokoro.set_language_for_voice(voice_id).map_err(|e| {
                    TtsError::InitError(format!(
                        "Failed to set language for voice {}: {}",
                        voice_id, e
                    ))
                })?;
            }
        }

        self.speaker_id = speaker_id.to_string();
        Ok(())
    }

    /// Set the speaker voice by numeric ID (for backwards compatibility)
    pub fn set_speaker_id(&mut self, id: i32) {
        let voice = match &self.engine {
            TtsImpl::Pocket(_) => match id {
                0 => "alba",
                1 => "marius",
                2 => "javert",
                3 => "jean",
                4 => "fantine",
                5 => "cosette",
                6 => "eponine",
                7 => "azelma",
                _ => POCKET_DEFAULT_VOICE,
            },
            TtsImpl::Supertonic(_) => match id {
                0 => "F1",
                1 => "F2",
                2 => "M1",
                3 => "M2",
                _ => SUPERTONIC_DEFAULT_VOICE,
            },
            TtsImpl::Kokoro(_) => {
                let voice_id = default_kokoro_voice_for_id(id);
                let voice_id = voice_id.to_string();
                if let Err(e) = self.set_speaker(&voice_id) {
                    log::warn!("Failed to set speaker {}: {}", voice_id, e);
                }
                return;
            }
        };

        if let Err(e) = self.set_speaker(voice) {
            log::warn!("Failed to set speaker {}: {}", voice, e);
        }
    }

    /// Set the speech speed (0.5 = half speed, 2.0 = double speed)
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed.clamp(0.25, 4.0);
        if let TtsImpl::Supertonic(tts) = &mut self.engine {
            tts.set_speed(self.speed);
        }
    }

    /// Synthesize text to audio samples
    pub fn synthesize(&mut self, text: &str) -> Result<AudioOutput, TtsError> {
        if text.is_empty() {
            return Ok(AudioOutput {
                samples: Vec::new(),
                sample_rate: self.sample_rate,
            });
        }

        match &mut self.engine {
            TtsImpl::Pocket(tts) => tts.synthesize(text),
            TtsImpl::Supertonic(tts) => {
                let audio = tts
                    .synthesize(text)
                    .map_err(|e| TtsError::SynthesisError(e.to_string()))?;
                Ok(AudioOutput {
                    samples: audio.samples,
                    sample_rate: audio.sample_rate,
                })
            }
            TtsImpl::Kokoro(tts) => {
                let speaker_id = self.speaker_id.parse::<i32>().unwrap_or(0);
                let audio = tts
                    .synthesize(text, speaker_id, self.speed)
                    .map_err(|e| TtsError::SynthesisError(e.to_string()))?;
                Ok(AudioOutput {
                    samples: audio.samples,
                    sample_rate: audio.sample_rate,
                })
            }
        }
    }

    /// Synthesize text and stream audio chunks to a callback.
    pub fn synthesize_streaming<F>(&mut self, text: &str, mut on_chunk: F) -> Result<u32, TtsError>
    where
        F: FnMut(&[f32], u32) -> bool,
    {
        let text = text.trim();
        if text.is_empty() {
            return Ok(self.sample_rate);
        }

        match &mut self.engine {
            TtsImpl::Pocket(tts) => tts.synthesize_streaming(text, &mut on_chunk),
            TtsImpl::Kokoro(tts) => {
                let speaker_id = self.speaker_id.parse::<i32>().unwrap_or(0);
                tts.synthesize_streaming(text, speaker_id, self.speed, &mut on_chunk)
                    .map_err(|e| TtsError::SynthesisError(e.to_string()))
            }
            _ => {
                let audio = self.synthesize(text)?;
                let _ = on_chunk(&audio.samples, audio.sample_rate);
                Ok(audio.sample_rate)
            }
        }
    }

    /// Get the output sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get current speaker ID
    pub fn current_speaker(&self) -> &str {
        &self.speaker_id
    }

    pub fn current_language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    pub fn resolved_engine(&self) -> TtsEngine {
        self.resolved_engine.clone()
    }

    /// Get current engine type
    pub fn engine_type(&self) -> &str {
        self.resolved_engine.as_str()
    }

    /// Get available voices for Pocket
    pub fn available_pocket_voices() -> Vec<VoiceInfo> {
        vec![
            VoiceInfo {
                id: 0,
                id_str: "alba".to_string(),
                name: "Alba".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 1,
                id_str: "marius".to_string(),
                name: "Marius".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 2,
                id_str: "javert".to_string(),
                name: "Javert".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 3,
                id_str: "jean".to_string(),
                name: "Jean".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 4,
                id_str: "fantine".to_string(),
                name: "Fantine".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 5,
                id_str: "cosette".to_string(),
                name: "Cosette".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 6,
                id_str: "eponine".to_string(),
                name: "Eponine".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 7,
                id_str: "azelma".to_string(),
                name: "Azelma".to_string(),
                language: "en".to_string(),
            },
        ]
    }

    /// Get available languages for Pocket TTS
    pub fn available_pocket_languages() -> Vec<(&'static str, &'static str)> {
        vec![
            ("en", "English"),
            ("fr", "French"),
            ("de", "German"),
            ("it", "Italian"),
            ("pt", "Portuguese"),
            ("es", "Spanish"),
        ]
    }

    /// Get available voices for Supertonic
    pub fn available_supertonic_voices() -> Vec<VoiceInfo> {
        vec![
            VoiceInfo {
                id: 0,
                id_str: "F1".to_string(),
                name: "Female 1 (F1)".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 1,
                id_str: "F2".to_string(),
                name: "Female 2 (F2)".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 2,
                id_str: "M1".to_string(),
                name: "Male 1 (M1)".to_string(),
                language: "en".to_string(),
            },
            VoiceInfo {
                id: 3,
                id_str: "M2".to_string(),
                name: "Male 2 (M2)".to_string(),
                language: "en".to_string(),
            },
        ]
    }

    /// Get available voices for Kokoro (all 53 voices across 9 languages)
    pub fn available_kokoro_voices() -> Vec<VoiceInfo> {
        (0..53)
            .map(|id| {
                let name = crate::kokoro_tts::voice_id_to_name(id);
                let language = crate::kokoro_tts::voice_id_to_language(id);
                VoiceInfo {
                    id,
                    id_str: id.to_string(),
                    name: name.to_string(),
                    language: language.to_string(),
                }
            })
            .collect()
    }

    /// Get available voices (legacy method, returns current engine's voices)
    pub fn available_voices() -> Vec<VoiceInfo> {
        Self::available_pocket_voices()
    }
}

fn build_engine(
    engine_type: &TtsEngine,
    speaker_id: &str,
    _language: Option<&str>,
) -> Result<(TtsImpl, u32), TtsError> {
    match engine_type {
        TtsEngine::Auto => unreachable!("auto must be resolved before engine build"),
        TtsEngine::Pocket => {
            let pocket = PocketTts::new_with_voice(POCKET_DEFAULT_VARIANT, speaker_id)
                .map_err(|e| TtsError::InitError(format!("Pocket TTS: {}", e)))?;
            let sample_rate = pocket.sample_rate();
            Ok((TtsImpl::Pocket(pocket), sample_rate))
        }
        TtsEngine::Supertonic => {
            let paths = models::get_supertonic_paths()?;
            build_supertonic(&paths, speaker_id)
        }
        TtsEngine::Kokoro => {
            let voice_num = speaker_id.parse::<i32>().unwrap_or(0);
            let language = crate::kokoro_tts::voice_id_to_language(voice_num);
            log::info!(
                "[TTS] Initializing Kokoro with voice {} (language: {})",
                speaker_id,
                language
            );
            let kokoro = KokoroTts::new_with_language(language)
                .map_err(|e| TtsError::InitError(format!("Kokoro init failed: {}", e)))?;
            Ok((TtsImpl::Kokoro(kokoro), 24000))
        }
    }
}

fn build_supertonic(
    paths: &models::SupertonicPaths,
    speaker_id: &str,
) -> Result<(TtsImpl, u32), TtsError> {
    if !paths.all_exist() {
        return Err(TtsError::ModelNotFound(
            "Supertonic models not fully downloaded".to_string(),
        ));
    }

    let mut tts =
        SupertonicTts::new(&paths.model_dir).map_err(|e| TtsError::InitError(e.to_string()))?;
    let voice_path = paths.voice_path(speaker_id);

    if !voice_path.exists() {
        return Err(TtsError::ModelNotFound(format!(
            "Voice file not found: {}",
            voice_path.display()
        )));
    }

    let voice_data = VoiceStyleData::from_json_file(&voice_path, speaker_id, speaker_id)
        .map_err(|e| TtsError::InitError(format!("Failed to load voice: {}", e)))?;
    tts.set_voice_style(&voice_data);
    Ok((TtsImpl::Supertonic(tts), supertonic::SAMPLE_RATE))
}

fn normalize_language(language: Option<&str>) -> Option<String> {
    let language = language?.trim();
    if language.is_empty() {
        return None;
    }
    let base = language
        .split(['-', '_'])
        .next()
        .unwrap_or(language)
        .to_ascii_lowercase();
    if base == "en" {
        Some("en".to_string())
    } else {
        Some(base)
    }
}

fn kokoro_language(language: &str) -> &str {
    match language {
        "en" => "en-us",
        other => other,
    }
}

fn default_voice_for_engine(engine: &TtsEngine, language: Option<&str>) -> String {
    match engine {
        TtsEngine::Auto => POCKET_DEFAULT_VOICE.to_string(),
        TtsEngine::Pocket => POCKET_DEFAULT_VOICE.to_string(),
        TtsEngine::Supertonic => SUPERTONIC_DEFAULT_VOICE.to_string(),
        TtsEngine::Kokoro => default_kokoro_voice_for_language(language).to_string(),
    }
}

fn default_kokoro_voice_for_language(language: Option<&str>) -> &'static str {
    match normalize_language(language).as_deref() {
        Some("es") => "28",
        Some("fr") => "30",
        Some("hi") => "31",
        Some("it") => "35",
        Some("ja") => "37",
        Some("pt") => "42",
        Some("zh") => "45",
        _ => KOKORO_DEFAULT_VOICE,
    }
}

fn default_kokoro_voice_for_id(id: i32) -> i32 {
    if (0..=52).contains(&id) {
        id
    } else {
        0
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct VoiceInfo {
    pub id: i32,
    pub id_str: String,
    pub name: String,
    pub language: String,
}

#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Initialization error: {0}")]
    InitError(String),
    #[error("Synthesis error: {0}")]
    SynthesisError(String),
}

impl From<crate::models::ModelError> for TtsError {
    fn from(e: crate::models::ModelError) -> Self {
        TtsError::ModelNotFound(e.to_string())
    }
}
