// Jack Desktop - Kokoro TTS via kokoro-tiny + direct ONNX pipeline
// Provides fast multilingual text-to-speech with REAL CUDA support
//
// For English voices: uses kokoro-tiny's synthesize() (espeak-ng English)
// For non-English voices: bypasses kokoro-tiny and runs espeak-ng (with language flag) → tokenize → ONNX directly
// Italian uses a dedicated rule-based G2P; all others use espeak-ng with --ipa -v <lang>

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use kokoro_tiny::TtsEngine;
use ndarray::{ArrayBase, IxDyn, OwnedRepr};
use ndarray_npy::NpzReader;
use ort::{
    session::{builder::GraphOptimizationLevel, Session, SessionInputValue, SessionInputs},
    value::{Tensor, Value},
};

use crate::models;

/// Map Kokoro voice ID to voice name for kokoro-tiny
/// Voice IDs 0-52 represent different language/gender combinations
///
/// ITALIAN VOICES (35-36): Use if_sara and im_nicola for proper Italian pronunciation
/// These voices use espeak-ng Italian phonemizer via direct ONNX pipeline
pub fn voice_id_to_name(voice_id: i32) -> &'static str {
    match voice_id {
        // American English Female (0-10)
        0 => "af_alloy",
        1 => "af_aoede",
        2 => "af_bella",
        3 => "af_heart",
        4 => "af_jessica",
        5 => "af_kore",
        6 => "af_nicole",
        7 => "af_nova",
        8 => "af_river",
        9 => "af_sarah",
        10 => "af_sky",
        // American English Male (11-19)
        11 => "am_adam",
        12 => "am_echo",
        13 => "am_eric",
        14 => "am_fenrir",
        15 => "am_liam",
        16 => "am_michael",
        17 => "am_onyx",
        18 => "am_puck",
        19 => "am_santa",
        // British English Female (20-23)
        20 => "bf_alice",
        21 => "bf_emma",
        22 => "bf_isabella",
        23 => "bf_lily",
        // British English Male (24-27)
        24 => "bm_daniel",
        25 => "bm_fable",
        26 => "bm_george",
        27 => "bm_lewis",
        // Spanish (28-29)
        28 => "ef_dora",
        29 => "em_alex",
        // French (30)
        30 => "ff_siwis",
        // Hindi (31-34)
        31 => "hf_alpha",
        32 => "hf_beta",
        33 => "hm_omega",
        34 => "hm_psi",
        // Italian (35-36)
        35 => "if_sara",
        36 => "im_nicola",
        // Japanese (37-41)
        37 => "jf_alpha",
        38 => "jf_gongitsune",
        39 => "jf_nezumi",
        40 => "jf_tebukuro",
        41 => "jm_kumo",
        // Portuguese (42-44)
        42 => "pf_dora",
        43 => "pm_alex",
        44 => "pm_santa",
        // Mandarin Chinese (45-52)
        45 => "zf_xiaobei",
        46 => "zf_xiaoni",
        47 => "zf_xiaoxiao",
        48 => "zf_xiaoyi",
        49 => "zm_yunjian",
        50 => "zm_yunxi",
        51 => "zm_yunxia",
        52 => "zm_yunyang",
        // Default fallback
        _ => "af_bella",
    }
}

/// Map Kokoro voice ID to language code
pub fn voice_id_to_language(voice_id: i32) -> &'static str {
    match voice_id {
        0..=19 => "en-us",
        20..=27 => "en-gb",
        28..=29 => "es",
        30 => "fr",
        31..=34 => "hi",
        35..=36 => "it",
        37..=41 => "ja",
        42..=44 => "pt",
        45..=52 => "zh",
        _ => "en-us",
    }
}

/// Split text into chunks for ONNX inference.
///
/// Kokoro 82M handles up to 510 tokens. Optimal range on CPU is 150-200 tokens
/// (~250-350 chars). We merge short sentences together and only split when a
/// chunk would exceed the target. This gives the model enough context for
/// natural prosody while keeping latency under ~1s per chunk on CPU.
const MAX_CHUNK_CHARS: usize = 300;

fn split_into_sentences(text: &str) -> Vec<String> {
    // Split on sentence endings, then merge small sentences into larger chunks
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    if sentences.is_empty() {
        return vec![text.to_string()];
    }

    // Merge sentences into chunks up to MAX_CHUNK_CHARS
    let mut merged = Vec::new();
    let mut buf = String::new();

    for sentence in sentences {
        if buf.is_empty() {
            buf = sentence;
        } else if buf.len() + 1 + sentence.len() <= MAX_CHUNK_CHARS {
            buf.push(' ');
            buf.push_str(&sentence);
        } else {
            merged.push(buf);
            buf = sentence;
        }
    }
    if !buf.is_empty() {
        merged.push(buf);
    }

    // Safety split: if any chunk is still very long, split at comma/semicolon
    let mut final_chunks = Vec::new();
    for chunk in merged {
        if chunk.len() <= MAX_CHUNK_CHARS {
            final_chunks.push(chunk);
        } else {
            let mut sub = String::new();
            for ch in chunk.chars() {
                sub.push(ch);
                if matches!(ch, ',' | ';' | ':') && sub.len() >= 100 {
                    let trimmed = sub.trim().to_string();
                    if !trimmed.is_empty() {
                        final_chunks.push(trimmed);
                    }
                    sub.clear();
                }
            }
            let trimmed = sub.trim().to_string();
            if !trimmed.is_empty() {
                final_chunks.push(trimmed);
            }
        }
    }

    if final_chunks.is_empty() {
        final_chunks.push(text.to_string());
    }

    final_chunks
}

/// Check if a voice ID requires the direct pipeline
fn needs_direct_pipeline(voice_id: i32) -> bool {
    // Non-English voices use the direct pipeline with espeak-ng phonemization + ONNX inference
    // 0-27: English (built-in pipeline), 28-52: multilingual direct pipeline
    matches!(voice_id, 28..=52)
}

/// Map Kokoro language code to espeak-ng voice/language identifier
fn language_to_espeak_voice(lang: &str) -> &'static str {
    match lang {
        "en-us" => "en-us",
        "en-gb" => "en-gb",
        "es" => "es",
        "fr" => "fr-fr",
        "hi" => "hi",
        "it" => "it",
        "ja" => "ja",
        "pt" => "pt-br",
        "zh" => "cmn",
        _ => "en",
    }
}

/// Generate IPA phonemes using espeak-ng for any supported language.
/// This is the general-purpose G2P used for all non-English languages
/// except Italian (which has a dedicated rule-based G2P).
fn espeak_g2p(text: &str, lang: &str, vocab: &HashMap<char, i64>) -> Result<String, String> {
    let espeak_voice = language_to_espeak_voice(lang);

    let output = std::process::Command::new("espeak-ng")
        .args(["--ipa", "-q", "-v", espeak_voice, text])
        .output()
        .map_err(|e| {
            format!(
                "espeak-ng failed: {}. Install: sudo apt install espeak-ng",
                e
            )
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "espeak-ng error (lang={}): {}",
            lang,
            stderr.trim()
        ));
    }

    let raw = String::from_utf8_lossy(&output.stdout).trim().to_string();

    // espeak-ng outputs newlines between clauses — join them
    let joined = raw.replace('\n', " ");

    // Filter to only characters present in the Kokoro vocabulary.
    // espeak-ng may output diacritics or symbols Kokoro doesn't recognize.
    let filtered: String = joined
        .chars()
        .filter(|c| c.is_whitespace() || vocab.contains_key(c))
        .collect();

    // Collapse multiple spaces
    let phonemes = filtered.split_whitespace().collect::<Vec<_>>().join(" ");

    let text_preview: String = text.chars().take(40).collect();
    let phon_preview: String = phonemes.chars().take(80).collect();
    log::debug!(
        "[espeak G2P] lang={} '{}' → '{}'",
        lang,
        text_preview,
        phon_preview
    );

    Ok(phonemes)
}

/// Build the Kokoro vocabulary (same as kokoro-tiny's internal build_vocab)
fn build_vocab() -> HashMap<char, i64> {
    let pad = "$";
    let punctuation = r#";:,.!?¡¿—…"«»"" "#;
    let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

    let symbols: String = [pad, punctuation, letters, letters_ipa].concat();

    symbols
        .chars()
        .enumerate()
        .map(|(idx, c)| (c, idx as i64))
        .collect()
}

/// Load voice style data from voices NPZ file
fn load_voices(path: &str) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut npz =
        NpzReader::new(File::open(path).map_err(|e| format!("Failed to open voices file: {}", e))?)
            .map_err(|e| format!("Failed to read NPZ: {:?}", e))?;

    let mut voices = HashMap::new();
    for name in npz
        .names()
        .map_err(|e| format!("Failed to get NPZ names: {:?}", e))?
    {
        let arr: ArrayBase<OwnedRepr<f32>, IxDyn> = npz
            .by_name(&name)
            .map_err(|e| format!("Failed to read voice {}: {:?}", name, e))?;

        let shape = arr.shape();
        if shape.len() == 3 && shape[1] == 1 && shape[2] == 256 {
            let data = arr
                .as_slice()
                .ok_or_else(|| format!("Failed to get slice for voice {}", name))?[..256]
                .to_vec();
            voices.insert(name.trim_end_matches(".npy").to_string(), data);
        }
    }

    Ok(voices)
}

/// Get kokoro cache directory (same as kokoro-tiny)
fn get_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Path::new(&home).join(".cache").join("kokoros")
}

/// Direct ONNX pipeline for non-English synthesis
/// Bypasses kokoro-tiny's English-hardcoded phonemizer
struct DirectPipeline {
    session: Session,
    voices: HashMap<String, Vec<f32>>,
    vocab: HashMap<char, i64>,
    /// Input tensor name for token IDs: "tokens" (kokoro-tiny f32) or "input_ids" (onnx-community int8/fp16)
    tokens_input_name: String,
}

impl DirectPipeline {
    fn new() -> Result<Self, String> {
        let cache_dir = get_cache_dir();
        let voices_path = cache_dir.join("voices-v1.0.bin");

        // Prefer int8 quantized model (~88MB, ~2-3x faster on CPU) over f32 (~310MB).
        // Download int8 from: https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX
        let int8_path = cache_dir.join("kokoro-v1.0.int8.onnx");
        let f32_path = cache_dir.join("kokoro-v1.0.onnx");
        let (model_path, model_variant) = if int8_path.exists() {
            (int8_path, "int8")
        } else if f32_path.exists() {
            log::info!("[DirectPipeline] int8 model not found, using f32. For ~2-3x speedup, download kokoro-v1.0.int8.onnx to {:?}", cache_dir);
            (f32_path, "f32")
        } else {
            return Err(format!(
                "No Kokoro model found at {:?} (tried int8 and f32)",
                cache_dir
            ));
        };

        if !voices_path.exists() {
            return Err(format!("Voices not found at {:?}", voices_path));
        }

        let model_bytes =
            std::fs::read(&model_path).map_err(|e| format!("Failed to read model: {}", e))?;

        // Use available CPU cores for intra-op parallelism (within operators like matmul)
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let session = Session::builder()
            .map_err(|e| format!("Failed to create session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization level: {}", e))?
            .with_intra_threads(num_threads)
            .map_err(|e| format!("Failed to set intra threads: {}", e))?
            .commit_from_memory(&model_bytes)
            .map_err(|e| format!("Failed to load model: {}", e))?;

        let voices = load_voices(voices_path.to_str().unwrap_or(""))?;
        let vocab = build_vocab();

        // Detect the token input name: kokoro-tiny's f32 model uses "tokens",
        // onnx-community's int8/fp16 models use "input_ids"
        let tokens_input_name = session
            .inputs()
            .iter()
            .find(|i| i.name() == "input_ids" || i.name() == "tokens")
            .map(|i| i.name().to_string())
            .unwrap_or_else(|| "tokens".to_string());

        log::info!(
            "[DirectPipeline] Loaded {} ONNX model ({} threads) + {} voices (token input: '{}')",
            model_variant,
            num_threads,
            voices.len(),
            tokens_input_name
        );

        Ok(Self {
            session,
            voices,
            vocab,
            tokens_input_name,
        })
    }

    fn tokenize(&self, phonemes: &str) -> Vec<i64> {
        phonemes
            .chars()
            .filter_map(|c| self.vocab.get(&c).copied())
            .collect()
    }

    /// Run ONNX inference on a single chunk of tokens.
    /// Uses catch_unwind to prevent panics in ort from killing the TTS thread.
    fn infer_chunk(
        &mut self,
        tokens: Vec<i64>,
        style: &[f32],
        speed: f32,
    ) -> Result<Vec<f32>, String> {
        use std::borrow::Cow;

        let tokens_shape = [1_usize, tokens.len()];
        let tokens_tensor = Tensor::from_array((tokens_shape, tokens))
            .map_err(|e| format!("Failed to create tokens tensor: {}", e))?;

        let style_shape = [1_usize, style.len()];
        let style_tensor = Tensor::from_array((style_shape, style.to_vec()))
            .map_err(|e| format!("Failed to create style tensor: {}", e))?;

        let speed_tensor = Tensor::from_array(([1_usize], vec![speed]))
            .map_err(|e| format!("Failed to create speed tensor: {}", e))?;

        let token_name = self.tokens_input_name.clone();
        let inputs = SessionInputs::from(vec![
            (
                Cow::Owned(token_name),
                SessionInputValue::Owned(Value::from(tokens_tensor)),
            ),
            (
                Cow::Borrowed("style"),
                SessionInputValue::Owned(Value::from(style_tensor)),
            ),
            (
                Cow::Borrowed("speed"),
                SessionInputValue::Owned(Value::from(speed_tensor)),
            ),
        ]);

        // Wrap ONNX run in catch_unwind to prevent panics from killing TTS thread
        let session_ptr = &mut self.session as *mut Session;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let session = unsafe { &mut *session_ptr };
            session.run(inputs)
        }));

        let outputs = match result {
            Ok(Ok(outputs)) => outputs,
            Ok(Err(e)) => return Err(format!("ONNX inference failed: {}", e)),
            Err(_) => return Err("ONNX inference panicked".to_string()),
        };

        // Access output by index (0) instead of name — model variants use different
        // output names ("audio" for kokoro-tiny f32, different for onnx-community int8)
        let first_output = outputs
            .iter()
            .next()
            .ok_or("ONNX inference produced no outputs")?;
        let (_shape, data) = first_output
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract audio tensor: {}", e))?;

        Ok(data.to_vec())
    }

    fn synthesize(
        &mut self,
        text: &str,
        voice_name: &str,
        language: &str,
        speed: f32,
    ) -> Result<Vec<f32>, String> {
        log::info!(
            "[DirectPipeline] Synthesizing with voice '{}', language '{}'",
            voice_name,
            language
        );

        // Split text into clauses at sentence/clause boundaries to keep each ONNX
        // inference call short (~50 tokens). This prevents slowdown on long text
        // where 170+ tokens can take 1.8s+ in a single call.
        let chunks = split_into_sentences(text);

        let style = self
            .voices
            .get(voice_name)
            .ok_or_else(|| format!("Voice '{}' not found in voices data", voice_name))?
            .clone();

        let mut audio = Vec::new();

        for (i, chunk) in chunks.iter().enumerate() {
            let chunk = chunk.trim();
            if chunk.is_empty() {
                continue;
            }

            // Use espeak-ng G2P for all non-English languages (including Italian)
            let phonemes_str = espeak_g2p(chunk, language, &self.vocab)?;
            if phonemes_str.trim().is_empty() {
                continue;
            }

            let mut tokens = self.tokenize(&phonemes_str);
            if tokens.is_empty() {
                continue;
            }

            // Wrap tokens with pad token (0) at start and end.
            // The Kokoro Python reference does: [0, *chunk, 0]
            // This gives the model proper sequence boundaries, preventing
            // first-word cutoff and last-word truncation.
            tokens.insert(0, 0);
            tokens.push(0);

            log::debug!(
                "[DirectPipeline] Chunk {}/{}: '{}' → {} tokens (padded)",
                i + 1,
                chunks.len(),
                &chunk.chars().take(30).collect::<String>(),
                tokens.len(),
            );

            match self.infer_chunk(tokens, &style, speed) {
                Ok(chunk_audio) => audio.extend_from_slice(&chunk_audio),
                Err(e) => {
                    log::warn!(
                        "[DirectPipeline] Chunk {}/{} failed, skipping: {}",
                        i + 1,
                        chunks.len(),
                        e
                    );
                }
            }
        }

        if audio.is_empty() {
            return Err("No audio produced from any chunk".to_string());
        }

        Ok(audio)
    }
}

/// Kokoro TTS wrapper using kokoro-tiny (English) + direct pipeline (other languages)
pub struct KokoroTts {
    engine: Arc<Mutex<TtsEngine>>,
    runtime: tokio::runtime::Runtime,
    direct_pipeline: Option<DirectPipeline>,
    sample_rate: u32,
    current_language: String,
}

impl KokoroTts {
    /// Create a new Kokoro TTS instance
    pub fn new() -> Result<Self, TtsError> {
        Self::new_with_language("en-us")
    }

    /// Create Kokoro TTS with specific language
    pub fn new_with_language(language: &str) -> Result<Self, TtsError> {
        log::info!(
            "[Kokoro TTS] Loading kokoro-tiny with language: {}",
            language
        );

        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| TtsError::InitError(format!("Failed to create runtime: {}", e)))?;

        let engine = runtime.block_on(async {
            TtsEngine::new()
                .await
                .map_err(|e| TtsError::InitError(format!("kokoro-tiny init failed: {}", e)))
        })?;

        log::info!("[Kokoro TTS] kokoro-tiny initialized successfully");

        // Initialize direct pipeline for non-English voices
        let direct_pipeline = match DirectPipeline::new() {
            Ok(dp) => {
                log::info!("[Kokoro TTS] Direct pipeline ready for non-English voices");
                Some(dp)
            }
            Err(e) => {
                log::warn!(
                    "[Kokoro TTS] Direct pipeline unavailable (non-English voices won't work): {}",
                    e
                );
                None
            }
        };

        Ok(Self {
            engine: Arc::new(Mutex::new(engine)),
            runtime,
            direct_pipeline,
            sample_rate: 24000,
            current_language: language.to_string(),
        })
    }

    /// Change the language
    pub fn set_language(&mut self, language: &str) -> Result<(), TtsError> {
        if self.current_language == language {
            return Ok(());
        }
        log::info!(
            "[Kokoro TTS] Language changed from {} to {}",
            self.current_language,
            language
        );
        self.current_language = language.to_string();
        Ok(())
    }

    /// Set language based on voice ID
    pub fn set_language_for_voice(&mut self, voice_id: i32) -> Result<(), TtsError> {
        let language = voice_id_to_language(voice_id);
        self.set_language(language)
    }

    /// Synthesize speech from text
    ///
    /// For English voices: uses kokoro-tiny (espeak-ng English)
    /// For non-English voices: uses direct pipeline (espeak-ng with language flag → ONNX)
    /// Italian uses a dedicated rule-based G2P; all others use espeak-ng
    pub fn synthesize(
        &mut self,
        text: &str,
        speaker_id: i32,
        speed: f32,
    ) -> Result<AudioOutput, TtsError> {
        let voice_name = voice_id_to_name(speaker_id);

        log::debug!(
            "[Kokoro TTS] Synthesizing with voice {} (ID {})",
            voice_name,
            speaker_id
        );

        let samples = if needs_direct_pipeline(speaker_id) {
            // Non-English: use direct pipeline with language-specific phonemization
            let pipeline = self.direct_pipeline.as_mut().ok_or_else(|| {
                TtsError::SynthesisError(
                    "Direct pipeline not available for non-English voice".to_string(),
                )
            })?;

            let language = voice_id_to_language(speaker_id);
            log::info!(
                "[Kokoro TTS] Using direct pipeline: voice '{}', language '{}'",
                voice_name,
                language
            );

            pipeline
                .synthesize(text, voice_name, language, speed)
                .map_err(|e| TtsError::SynthesisError(e))?
        } else {
            // English: use kokoro-tiny's built-in pipeline (reuse stored runtime)
            let engine = self.engine.clone();
            let voice = voice_name.to_string();
            let text_owned = text.to_string();

            self.runtime.block_on(async move {
                let mut engine = engine.lock().expect("kokoro engine lock poisoned");
                let primary = engine.synthesize(&text_owned, Some(&voice));
                match primary {
                    Ok(samples) => Ok(samples),
                    Err(e) => {
                        let msg = e.to_string();
                        let lower = msg.to_lowercase();
                        if lower.contains("voice") && lower.contains("not found") {
                            let fallback_voice = "af_bella";
                            log::warn!(
                                "[Kokoro TTS] Voice '{}' not found, retrying with fallback '{}'",
                                voice,
                                fallback_voice
                            );
                            engine
                                .synthesize(&text_owned, Some(fallback_voice))
                                .map_err(|e| {
                                    TtsError::SynthesisError(format!(
                                        "kokoro-tiny synthesis failed: {}",
                                        e
                                    ))
                                })
                        } else {
                            Err(TtsError::SynthesisError(format!(
                                "kokoro-tiny synthesis failed: {}",
                                msg
                            )))
                        }
                    }
                }
            })?
        };

        Ok(AudioOutput {
            samples,
            sample_rate: self.sample_rate,
        })
    }

    /// Synthesize speech with per-chunk streaming callback.
    ///
    /// Each sentence/clause is synthesized independently and yielded to `on_chunk`
    /// as soon as it's ready, dramatically reducing time-to-first-audio for
    /// multi-sentence text.
    pub fn synthesize_streaming<F>(
        &mut self,
        text: &str,
        speaker_id: i32,
        speed: f32,
        on_chunk: &mut F,
    ) -> Result<u32, TtsError>
    where
        F: FnMut(&[f32], u32) -> bool,
    {
        let voice_name = voice_id_to_name(speaker_id);

        if needs_direct_pipeline(speaker_id) {
            let pipeline = self.direct_pipeline.as_mut().ok_or_else(|| {
                TtsError::SynthesisError(
                    "Direct pipeline not available for non-English voice".to_string(),
                )
            })?;

            let language = voice_id_to_language(speaker_id);
            let chunks = split_into_sentences(text);

            let style = pipeline
                .voices
                .get(voice_name)
                .ok_or_else(|| {
                    TtsError::SynthesisError(format!(
                        "Voice '{}' not found in voices data",
                        voice_name
                    ))
                })?
                .clone();

            for (i, chunk) in chunks.iter().enumerate() {
                let chunk = chunk.trim();
                if chunk.is_empty() {
                    continue;
                }

                let phonemes_str = espeak_g2p(chunk, language, &pipeline.vocab)
                    .map_err(|e| TtsError::SynthesisError(e))?;
                if phonemes_str.trim().is_empty() {
                    continue;
                }

                let mut tokens = pipeline.tokenize(&phonemes_str);
                if tokens.is_empty() {
                    continue;
                }

                tokens.insert(0, 0);
                tokens.push(0);

                log::debug!(
                    "[DirectPipeline] Streaming chunk {}/{}: {} tokens",
                    i + 1,
                    chunks.len(),
                    tokens.len(),
                );

                match pipeline.infer_chunk(tokens, &style, speed) {
                    Ok(chunk_audio) => {
                        if !on_chunk(&chunk_audio, self.sample_rate) {
                            break;
                        }
                    }
                    Err(e) => {
                        log::warn!(
                            "[DirectPipeline] Streaming chunk {}/{} failed, skipping: {}",
                            i + 1,
                            chunks.len(),
                            e
                        );
                    }
                }
            }
        } else {
            // English: kokoro-tiny doesn't support streaming, synthesize as one block
            let result = self.synthesize(text, speaker_id, speed)?;
            on_chunk(&result.samples, result.sample_rate);
        }

        Ok(self.sample_rate)
    }

    /// Get sample rate (always 24kHz for Kokoro)
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Audio output from TTS synthesis
pub struct AudioOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
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

impl From<models::ModelError> for TtsError {
    fn from(e: models::ModelError) -> Self {
        TtsError::ModelNotFound(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voice_id_mapping() {
        assert_eq!(voice_id_to_language(0), "en-us");
        assert_eq!(voice_id_to_language(35), "it");
        assert_eq!(voice_id_to_name(0), "af_alloy");
        assert_eq!(voice_id_to_name(2), "af_bella");
        assert_eq!(voice_id_to_name(10), "af_sky");
        assert_eq!(voice_id_to_name(16), "am_michael");
        assert_eq!(voice_id_to_name(35), "if_sara");
        assert_eq!(voice_id_to_name(36), "im_nicola");
        // Default fallback
        assert_eq!(voice_id_to_name(999), "af_bella");
    }

    #[test]
    fn test_direct_pipeline_needed() {
        assert!(!needs_direct_pipeline(0)); // English
        assert!(!needs_direct_pipeline(10)); // English
        assert!(!needs_direct_pipeline(27)); // British English
        assert!(needs_direct_pipeline(28)); // Spanish
        assert!(needs_direct_pipeline(30)); // French
        assert!(needs_direct_pipeline(31)); // Hindi
        assert!(needs_direct_pipeline(35)); // Italian
        assert!(needs_direct_pipeline(36)); // Italian
        assert!(needs_direct_pipeline(37)); // Japanese
        assert!(needs_direct_pipeline(42)); // Portuguese
        assert!(needs_direct_pipeline(45)); // Chinese
        assert!(needs_direct_pipeline(52)); // Chinese
    }

    #[test]
    fn test_vocab_build() {
        let vocab = build_vocab();
        // Should contain IPA characters
        assert!(vocab.contains_key(&'ɑ'));
        assert!(vocab.contains_key(&'ʃ'));
        assert!(vocab.contains_key(&'ˈ'));
        // Should contain basic ASCII
        assert!(vocab.contains_key(&'a'));
        assert!(vocab.contains_key(&' '));
    }

    #[test]
    #[ignore]
    fn test_kokoro_synthesis() {
        if let Ok(mut tts) = KokoroTts::new() {
            let result = tts.synthesize("Hello world", 0, 1.0);
            assert!(result.is_ok());
            let audio = result.unwrap();
            assert!(!audio.samples.is_empty());
            assert_eq!(audio.sample_rate, 24000);
        }
    }

    #[test]
    #[ignore]
    fn test_kokoro_streaming_yields_per_chunk() {
        let mut tts =
            KokoroTts::new_with_language("it").expect("Failed to init Kokoro TTS for streaming");
        let text = "Prima frase. Seconda frase. Terza frase.";
        let mut chunk_count = 0usize;
        let mut total_samples = 0usize;
        let sr = tts
            .synthesize_streaming(text, 35, 1.0, &mut |samples: &[f32], _sr: u32| {
                chunk_count += 1;
                total_samples += samples.len();
                true
            })
            .expect("streaming synthesis");
        assert_eq!(sr, 24000);
        assert!(
            chunk_count >= 2,
            "Expected multiple streaming chunks for 3 sentences, got {}",
            chunk_count
        );
        assert!(total_samples > 0);
    }
}
