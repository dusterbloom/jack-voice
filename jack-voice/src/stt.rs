// Jack Desktop - Speech-to-Text with sherpa-rs
// Uses Moonshine for streaming transcription
// Also supports batch Whisper for high-accuracy mode

use sherpa_rs::moonshine::{MoonshineConfig, MoonshineRecognizer};
use sherpa_rs::whisper::{WhisperConfig, WhisperRecognizer};

use crate::models;
use crate::speaker::SttMode;

/// Detect language from TTS voice name or ID.
/// Returns a language code compatible with Whisper (e.g., "en", "it", "es").
fn detect_language_from_voice(voice_name: &str) -> &'static str {
    // Try to parse voice name as integer (Kokoro voice ID)
    if let Ok(voice_id) = voice_name.parse::<i32>() {
        // Use voice_id_to_language mapping (from kokoro_tts.rs pattern)
        return match voice_id {
            0..=19 => "en",  // American English
            20..=27 => "en", // British English
            28..=29 => "es", // Spanish
            30 => "fr",      // French
            31..=34 => "hi", // Hindi
            35..=36 => "it", // Italian
            37..=41 => "ja", // Japanese
            42..=44 => "pt", // Portuguese
            45..=52 => "zh", // Mandarin Chinese
            _ => "en",
        };
    }

    // Check for language prefixes in voice name (if_sara, im_nicola, etc.)
    let lower = voice_name.to_lowercase();
    if lower.starts_with("if_") || lower.starts_with("im_") || lower.contains("italian") {
        "it"
    } else if lower.starts_with("af_")
        || lower.starts_with("am_")
        || lower.starts_with("bf_")
        || lower.starts_with("bm_")
        || lower.contains("english")
    {
        "en"
    } else if lower.contains("spanish") || lower.contains("es_") {
        "es"
    } else if lower.contains("french") || lower.contains("fr_") {
        "fr"
    } else if lower.contains("hindi") || lower.contains("hi_") {
        "hi"
    } else if lower.contains("japanese") || lower.contains("ja_") {
        "ja"
    } else if lower.contains("portuguese") || lower.contains("pt_") {
        "pt"
    } else if lower.contains("chinese") || lower.contains("zh_") {
        "zh"
    } else {
        "en" // Default to English
    }
}

/// Create Moonshine recognizer with the best available provider
/// Tries GPU (DirectML/CUDA) first, falls back to CPU if unavailable
fn create_moonshine_with_best_provider(
    paths: &models::MoonshinePaths,
) -> Result<(MoonshineRecognizer, &'static str), SttError> {
    // Determine which GPU provider to try based on compile-time features
    let gpu_provider = if cfg!(feature = "directml") {
        Some(("DirectML", "directml"))
    } else if cfg!(feature = "cuda") {
        Some(("CUDA", "cuda"))
    } else {
        None
    };

    // Try GPU provider first if available
    if let Some((provider_name, provider_string)) = gpu_provider {
        log::info!(
            "[STT] Attempting {} provider for Moonshine...",
            provider_name
        );

        let gpu_config = MoonshineConfig {
            preprocessor: paths.preprocessor.to_string_lossy().to_string(),
            encoder: paths.encoder.to_string_lossy().to_string(),
            cached_decoder: paths.cached_decoder.to_string_lossy().to_string(),
            uncached_decoder: paths.uncached_decoder.to_string_lossy().to_string(),
            tokens: paths.tokens.to_string_lossy().to_string(),
            provider: Some(provider_string.to_string()),
            num_threads: Some(4),
            debug: false,
        };

        match MoonshineRecognizer::new(gpu_config) {
            Ok(rec) => {
                log::info!("[STT] Moonshine using {} (GPU accelerated)", provider_name);
                return Ok((rec, provider_name));
            }
            Err(e) => {
                log::warn!("[STT] {} unavailable ({}), using CPU", provider_name, e);
            }
        }
    }

    // Fallback to CPU
    log::info!("[STT] Using CPU provider for Moonshine");
    let cpu_config = MoonshineConfig {
        preprocessor: paths.preprocessor.to_string_lossy().to_string(),
        encoder: paths.encoder.to_string_lossy().to_string(),
        cached_decoder: paths.cached_decoder.to_string_lossy().to_string(),
        uncached_decoder: paths.uncached_decoder.to_string_lossy().to_string(),
        tokens: paths.tokens.to_string_lossy().to_string(),
        provider: Some("cpu".to_string()),
        num_threads: Some(4),
        debug: false,
    };

    let rec =
        MoonshineRecognizer::new(cpu_config).map_err(|e| SttError::InitError(e.to_string()))?;
    Ok((rec, "CPU"))
}

/// Create Whisper recognizer with the best available provider
fn create_whisper_with_best_provider(
    paths: &models::WhisperPaths,
) -> Result<(WhisperRecognizer, &'static str), SttError> {
    create_whisper_with_language(paths, "en")
}

/// Create Whisper Turbo recognizer with language hint support
fn create_whisper_turbo_with_language(
    paths: &models::WhisperPaths,
    language: &str,
) -> Result<(WhisperRecognizer, &'static str), SttError> {
    create_whisper_with_language(paths, language)
}

/// Create Whisper recognizer with language hint and best available provider
fn create_whisper_with_language(
    paths: &models::WhisperPaths,
    language: &str,
) -> Result<(WhisperRecognizer, &'static str), SttError> {
    let gpu_provider = if cfg!(feature = "directml") {
        Some(("DirectML", "directml"))
    } else if cfg!(feature = "cuda") {
        Some(("CUDA", "cuda"))
    } else {
        None
    };

    if let Some((provider_name, provider_string)) = gpu_provider {
        log::info!(
            "[STT] Attempting {} provider for Whisper (language: {})...",
            provider_name,
            language
        );

        let gpu_config = WhisperConfig {
            encoder: paths.encoder.to_string_lossy().to_string(),
            decoder: paths.decoder.to_string_lossy().to_string(),
            tokens: paths.tokens.to_string_lossy().to_string(),
            language: language.to_string(),
            provider: Some(provider_string.to_string()),
            num_threads: Some(4),
            tail_paddings: Some(4800),
            ..Default::default()
        };

        match WhisperRecognizer::new(gpu_config) {
            Ok(rec) => {
                log::info!(
                    "[STT] Whisper using {} (GPU accelerated, language: {})",
                    provider_name,
                    language
                );
                return Ok((rec, provider_name));
            }
            Err(e) => {
                log::warn!("[STT] {} unavailable ({}), using CPU", provider_name, e);
            }
        }
    }

    log::info!(
        "[STT] Using CPU provider for Whisper (language: {})",
        language
    );
    let cpu_config = WhisperConfig {
        encoder: paths.encoder.to_string_lossy().to_string(),
        decoder: paths.decoder.to_string_lossy().to_string(),
        tokens: paths.tokens.to_string_lossy().to_string(),
        language: language.to_string(),
        provider: Some("cpu".to_string()),
        num_threads: Some(4),
        tail_paddings: Some(4800),
        ..Default::default()
    };

    let rec = WhisperRecognizer::new(cpu_config).map_err(|e| SttError::InitError(e.to_string()))?;
    Ok((rec, "CPU"))
}

#[derive(Clone, Debug)]
pub struct TranscriptionResult {
    pub text: String,
    pub is_final: bool,
    pub is_partial: bool,
    pub latency_ms: u64,
}

pub enum SttBackend {
    Batch(BatchTranscriber),
    Streaming(StreamingTranscriber),
    /// Parakeet TDT offline (multilingual, 25 langs)
    ParakeetTdt(crate::parakeet_stt::ParakeetOfflineStt),
    /// Parakeet EOU streaming (160ms chunks)
    ParakeetEou(crate::parakeet_stt::ParakeetStreamingStt),
    /// Whisper Turbo (multilingual, fast, with language hints)
    WhisperTurbo(BatchTranscriber),
}

pub struct SpeechToText {
    pub backend: SttBackend,
    mode: SttMode,
}

impl SpeechToText {
    pub fn new(mode: SttMode) -> Result<Self, SttError> {
        Self::with_language(mode, None, None)
    }

    /// Create STT with language hint from settings
    ///
    /// # Arguments
    /// * `mode` - Batch or Streaming mode
    /// * `stt_language` - Optional language code from settings (empty string = auto-detect)
    /// * `tts_voice` - Optional TTS voice for auto-detection
    pub fn with_language(
        mode: SttMode,
        stt_language: Option<String>,
        tts_voice: Option<String>,
    ) -> Result<Self, SttError> {
        // Determine language hint
        let language = match stt_language {
            Some(lang) if !lang.is_empty() => lang,
            _ => {
                // Auto-detect from TTS voice
                if let Some(voice) = tts_voice {
                    detect_language_from_voice(&voice).to_string()
                } else {
                    "en".to_string()
                }
            }
        };

        match mode {
            SttMode::Batch => {
                // Try WhisperTurbo first if model exists (multilingual with language hints)
                if let Ok(paths) = models::get_whisper_turbo_paths() {
                    match BatchTranscriber::with_language(&paths, &language) {
                        Ok(transcriber) => {
                            log::info!("[STT] Using Whisper Turbo (language: {})", language);
                            return Ok(Self {
                                backend: SttBackend::WhisperTurbo(transcriber),
                                mode,
                            });
                        }
                        Err(e) => log::warn!("[STT] Whisper Turbo failed: {}", e),
                    }
                }

                // Try Parakeet TDT (multilingual), fall back to Whisper
                if let Ok(paths) = models::get_parakeet_tdt_paths() {
                    match crate::parakeet_stt::ParakeetOfflineStt::new(
                        paths.model_dir.to_str().unwrap_or("."),
                    ) {
                        Ok(tdt) => {
                            log::info!("[STT] Using Parakeet TDT (multilingual, 25 langs)");
                            return Ok(Self {
                                backend: SttBackend::ParakeetTdt(tdt),
                                mode,
                            });
                        }
                        Err(e) => {
                            log::warn!("[STT] Parakeet TDT failed, falling back to Whisper: {}", e)
                        }
                    }
                }
                let transcriber = BatchTranscriber::new()?;
                Ok(Self {
                    backend: SttBackend::Batch(transcriber),
                    mode,
                })
            }
            SttMode::Streaming => {
                // Use ParakeetTDT for streaming mode too
                // WHY: ParakeetTDT handles batch transcription of complete audio buffers,
                // which matches our architecture (accumulate audio → transcribe complete turn).
                // ParakeetEOU is designed for continuous 160ms chunk streaming, causing
                // empty transcriptions when fed large buffers all at once.
                if let Ok(paths) = models::get_parakeet_tdt_paths() {
                    match crate::parakeet_stt::ParakeetOfflineStt::new(
                        paths.model_dir.to_str().unwrap_or("."),
                    ) {
                        Ok(tdt) => {
                            log::info!("[STT] Using Parakeet TDT for streaming (works for both incomplete/final)");
                            return Ok(Self {
                                backend: SttBackend::ParakeetTdt(tdt),
                                mode,
                            });
                        }
                        Err(e) => log::warn!(
                            "[STT] Parakeet TDT failed, falling back to Moonshine: {}",
                            e
                        ),
                    }
                }
                // Fall back to Moonshine streaming
                let transcriber = StreamingTranscriber::new()?;
                Ok(Self {
                    backend: SttBackend::Streaming(transcriber),
                    mode,
                })
            }
        }
    }

    pub fn with_config(mode: SttMode, _config: StreamingConfig) -> Result<Self, SttError> {
        Self::new(mode)
    }

    pub fn is_ready(&self) -> bool {
        match &self.backend {
            SttBackend::Batch(b) => b.is_ready(),
            SttBackend::Streaming(s) => s.is_ready(),
            SttBackend::ParakeetTdt(p) => p.is_ready(),
            SttBackend::ParakeetEou(p) => p.is_ready(),
            SttBackend::WhisperTurbo(w) => w.is_ready(),
        }
    }

    pub fn mode(&self) -> SttMode {
        self.mode
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<TranscriptionResult, SttError> {
        match &mut self.backend {
            SttBackend::Batch(b) => b.transcribe(samples),
            SttBackend::WhisperTurbo(w) => w.transcribe(samples),
            SttBackend::ParakeetTdt(p) => p.transcribe(samples, 16000),
            SttBackend::ParakeetEou(p) => match p.feed_chunk(samples, 16000)? {
                Some(r) => Ok(r),
                None => Ok(TranscriptionResult {
                    text: String::new(),
                    is_final: false,
                    is_partial: true,
                    latency_ms: 0,
                }),
            },
            SttBackend::Streaming(s) => {
                s.feed_audio(samples);
                match s.transcribe_buffer() {
                    Ok(Some(t)) => Ok(TranscriptionResult {
                        text: t.text,
                        is_final: t.is_final,
                        is_partial: t.is_partial,
                        latency_ms: 0,
                    }),
                    Ok(None) => Ok(TranscriptionResult {
                        text: String::new(),
                        is_final: false,
                        is_partial: true,
                        latency_ms: 0,
                    }),
                    Err(e) => Err(SttError::ProcessingError(e.to_string())),
                }
            }
        }
    }

    pub fn next_transcription(&mut self) -> Result<Option<TranscriptionResult>, SttError> {
        match &mut self.backend {
            SttBackend::Batch(_) | SttBackend::ParakeetTdt(_) | SttBackend::WhisperTurbo(_) => Err(
                SttError::ProcessingError("Use transcribe() for batch backend".to_string()),
            ),
            SttBackend::ParakeetEou(_) => Ok(None), // EOU uses feed_chunk via transcribe()
            SttBackend::Streaming(s) => {
                let start = std::time::Instant::now();
                match s.next_transcription() {
                    Ok(Some(t)) => {
                        let latency_ms = start.elapsed().as_millis() as u64;
                        Ok(Some(TranscriptionResult {
                            text: t.text,
                            is_final: t.is_final,
                            is_partial: t.is_partial,
                            latency_ms,
                        }))
                    }
                    Ok(None) => Ok(None),
                    Err(e) => Err(SttError::ProcessingError(e.to_string())),
                }
            }
        }
    }

    pub fn try_next_transcription(&mut self) -> Result<Option<TranscriptionResult>, SttError> {
        match &mut self.backend {
            SttBackend::Batch(_) | SttBackend::ParakeetTdt(_) | SttBackend::WhisperTurbo(_) => Err(
                SttError::ProcessingError("Use transcribe() for batch backend".to_string()),
            ),
            SttBackend::ParakeetEou(_) => Ok(None),
            SttBackend::Streaming(s) => {
                let start = std::time::Instant::now();
                match s.try_next() {
                    Ok(Some(t)) => {
                        let latency_ms = start.elapsed().as_millis() as u64;
                        Ok(Some(TranscriptionResult {
                            text: t.text,
                            is_final: t.is_final,
                            is_partial: t.is_partial,
                            latency_ms,
                        }))
                    }
                    Ok(None) => Ok(None),
                    Err(StreamingSttError::NoData) => Ok(None),
                    Err(e) => Err(SttError::ProcessingError(e.to_string())),
                }
            }
        }
    }

    pub fn is_streaming(&self) -> bool {
        matches!(
            self.backend,
            SttBackend::Streaming(_) | SttBackend::ParakeetEou(_)
        )
    }
}

// StreamingConfig for API compatibility
#[derive(Clone, Debug)]
pub struct StreamingConfig {
    pub step_ms: u32,
    pub length_ms: u32,
    pub keep_ms: u32,
}

impl StreamingConfig {
    pub fn fast() -> Self {
        Self {
            step_ms: 300,
            length_ms: 2000,
            keep_ms: 150,
        }
    }

    pub fn balanced() -> Self {
        Self {
            step_ms: 500,
            length_ms: 3000,
            keep_ms: 200,
        }
    }
}

// Streaming transcriber using sherpa-rs Moonshine
// Moonshine is designed for streaming transcription with low latency
pub struct StreamingTranscriber {
    recognizer: MoonshineRecognizer,
    sample_rate: u32,
    audio_buffer: Vec<f32>,
}

impl StreamingTranscriber {
    pub fn new() -> Result<Self, SttError> {
        let paths = models::get_moonshine_paths()?;

        // Select provider: GPU if available, otherwise CPU
        // DirectML (Windows) works on any GPU - NVIDIA, AMD, Intel
        // Falls back to CPU automatically if GPU init fails
        let (recognizer, provider_used) = create_moonshine_with_best_provider(&paths)?;
        log::info!(
            "[STT] Moonshine initialized with {} provider",
            provider_used
        );

        Ok(Self {
            recognizer,
            sample_rate: 16000,
            audio_buffer: Vec::new(),
        })
    }

    pub fn is_ready(&self) -> bool {
        true
    }

    pub fn feed_audio(&mut self, samples: &[f32]) {
        self.audio_buffer.extend_from_slice(samples);
    }

    pub fn transcribe_buffer(
        &mut self,
    ) -> Result<Option<StreamingTranscription>, StreamingSttError> {
        if self.audio_buffer.is_empty() {
            return Err(StreamingSttError::NoData);
        }

        let start = std::time::Instant::now();
        let result = self
            .recognizer
            .transcribe(self.sample_rate, &self.audio_buffer);
        let elapsed = start.elapsed();

        log::debug!("Moonshine transcription in {:?}", elapsed);

        let text = result.text.trim().to_string();
        self.audio_buffer.clear();

        if text.is_empty() {
            return Ok(None);
        }

        Ok(Some(StreamingTranscription {
            text,
            is_final: true,
            is_partial: false,
        }))
    }

    pub fn next_transcription(
        &mut self,
    ) -> Result<Option<StreamingTranscription>, StreamingSttError> {
        self.transcribe_buffer()
    }

    pub fn try_next(&mut self) -> Result<Option<StreamingTranscription>, StreamingSttError> {
        self.transcribe_buffer()
    }
}

#[derive(Clone, Debug)]
pub struct StreamingTranscription {
    pub text: String,
    pub is_final: bool,
    pub is_partial: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum StreamingSttError {
    #[error("No data available")]
    NoData,
    #[error("Processing error: {0}")]
    ProcessingError(String),
}

// Batch transcriber using sherpa-rs Whisper
pub struct BatchTranscriber {
    recognizer: WhisperRecognizer,
    sample_rate: u32,
}

const STT_MIN_RMS_THRESHOLD: f32 = 0.01;
const STT_MIN_AMPLITUDE_THRESHOLD: f32 = 0.03;

impl BatchTranscriber {
    pub fn new() -> Result<Self, SttError> {
        let paths = models::get_whisper_paths()?;
        Self::with_paths(&paths)
    }

    pub fn with_paths(paths: &models::WhisperPaths) -> Result<Self, SttError> {
        Self::with_language(paths, "en")
    }

    pub fn with_language(paths: &models::WhisperPaths, language: &str) -> Result<Self, SttError> {
        if !paths.encoder.exists() {
            return Err(SttError::ModelNotFound(paths.encoder.display().to_string()));
        }
        if !paths.decoder.exists() {
            return Err(SttError::ModelNotFound(paths.decoder.display().to_string()));
        }
        if !paths.tokens.exists() {
            return Err(SttError::ModelNotFound(paths.tokens.display().to_string()));
        }

        // Use best available provider (GPU with CPU fallback)
        let (recognizer, provider_used) = create_whisper_with_language(paths, language)?;
        log::info!(
            "[STT] Whisper initialized with {} provider (language: {})",
            provider_used,
            language
        );

        Ok(Self {
            recognizer,
            sample_rate: 16000,
        })
    }

    pub fn is_ready(&self) -> bool {
        true
    }

    pub fn transcribe(&mut self, samples: &[f32]) -> Result<TranscriptionResult, SttError> {
        if samples.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                is_final: true,
                is_partial: false,
                latency_ms: 0,
            });
        }

        let max_val = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        if rms < STT_MIN_RMS_THRESHOLD || max_val < STT_MIN_AMPLITUDE_THRESHOLD {
            log::info!(
                "STT: Rejecting low-energy audio (rms={:.4}, max={:.4})",
                rms,
                max_val
            );
            return Ok(TranscriptionResult {
                text: String::new(),
                is_final: true,
                is_partial: false,
                latency_ms: 0,
            });
        }

        let normalized_samples = Self::normalize_audio(samples, 0.1);

        let start = std::time::Instant::now();
        let result = self
            .recognizer
            .transcribe(self.sample_rate, &normalized_samples);
        let elapsed = start.elapsed();

        let raw_text = result.text.trim();
        let cleaned_text = Self::clean_transcription(raw_text);
        let audio_duration_secs = samples.len() as f32 / self.sample_rate as f32;

        let final_text = if !cleaned_text.is_empty()
            && !Self::validate_wps(&cleaned_text, audio_duration_secs)
        {
            log::info!("STT: WPS validation failed, returning empty");
            String::new()
        } else {
            cleaned_text.clone()
        };

        log::info!("STT: Transcribed in {:?}, text='{}'", elapsed, final_text);

        Ok(TranscriptionResult {
            text: final_text,
            is_final: true,
            is_partial: false,
            latency_ms: elapsed.as_millis() as u64,
        })
    }

    fn normalize_audio(samples: &[f32], target_rms: f32) -> Vec<f32> {
        if samples.is_empty() {
            return Vec::new();
        }

        let current_rms =
            (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        if current_rms < 0.001 {
            return samples.to_vec();
        }

        let gain = (target_rms / current_rms).min(10.0);

        samples
            .iter()
            .map(|s| (s * gain).clamp(-1.0, 1.0))
            .collect()
    }

    fn clean_transcription(text: &str) -> String {
        let mut cleaned = text.to_string();

        let artifacts = [
            "[BLANK_AUDIO]",
            "(inaudible)",
            "[inaudible]",
            "(music)",
            "[music]",
            "(silence)",
            "[silence]",
            "(static)",
            "[static]",
            "...",
        ];
        for artifact in artifacts {
            cleaned = cleaned.replace(artifact, "");
        }

        let youtube_hallucinations = [
            "thank you for watching",
            "thanks for watching",
            "subscribe",
            "like and subscribe",
            "hit the bell",
            "notification bell",
            "don't forget to subscribe",
            "see you in the next",
            "bye bye",
            "goodbye everyone",
        ];

        let lower = cleaned.to_lowercase();
        for marker in youtube_hallucinations {
            if lower.contains(marker) {
                log::info!("STT: Rejecting YouTube hallucination: '{}'", marker);
                return String::new();
            }
        }

        cleaned = Self::remove_repeated_phrases(&cleaned);
        cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");

        cleaned
    }

    fn validate_wps(text: &str, audio_duration_secs: f32) -> bool {
        if audio_duration_secs < 0.1 {
            return false;
        }

        let word_count = text.split_whitespace().count();
        let wps = word_count as f32 / audio_duration_secs;

        if wps < 0.3 || wps > 8.0 {
            log::info!(
                "STT: Rejecting WPS outlier: {:.1} wps ({} words in {:.2}s)",
                wps,
                word_count,
                audio_duration_secs
            );
            return false;
        }

        true
    }

    fn remove_repeated_phrases(text: &str) -> String {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.len() < 4 {
            return text.to_string();
        }

        for phrase_len in (2..=5).rev() {
            if words.len() < phrase_len * 2 {
                continue;
            }

            let mut result_words = Vec::new();
            let mut i = 0;

            while i < words.len() {
                if i + phrase_len * 2 <= words.len() {
                    let phrase1 = &words[i..i + phrase_len];
                    let phrase2 = &words[i + phrase_len..i + phrase_len * 2];

                    let phrases_match = phrase1
                        .iter()
                        .zip(phrase2.iter())
                        .all(|(a, b)| a.to_lowercase() == b.to_lowercase());

                    if phrases_match {
                        result_words.extend_from_slice(phrase1);
                        i += phrase_len * 2;

                        while i + phrase_len <= words.len() {
                            let next_phrase = &words[i..i + phrase_len];
                            let still_repeating = phrase1
                                .iter()
                                .zip(next_phrase.iter())
                                .all(|(a, b)| a.to_lowercase() == b.to_lowercase());
                            if still_repeating {
                                i += phrase_len;
                            } else {
                                break;
                            }
                        }
                        continue;
                    }
                }

                result_words.push(words[i]);
                i += 1;
            }

            if result_words.len() < words.len() {
                return result_words.join(" ");
            }
        }

        text.to_string()
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SttError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Initialization error: {0}")]
    InitError(String),
    #[error("Processing error: {0}")]
    ProcessingError(String),
}

impl From<crate::models::ModelError> for SttError {
    fn from(e: crate::models::ModelError) -> Self {
        SttError::ModelNotFound(e.to_string())
    }
}
