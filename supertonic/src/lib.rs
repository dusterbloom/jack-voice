// Supertonic TTS - Fast diffusion-based text-to-speech
// Native Rust implementation using ONNX Runtime
// Based on Supertone/supertonic architecture

use ndarray::Array3;
use ort::ep::ExecutionProvider;
use ort::session::Session;
use ort::value::Tensor;
use rand_distr::{Distribution, Normal};
use std::path::Path;

mod phonemizer;
mod voice_style;

pub use phonemizer::{chunk_text, UnicodeIndexer};
pub use voice_style::{VoiceStyle, VoiceStyleData};

/// Sample rate for Supertonic output (44.1kHz per actual ONNX model)
pub const SAMPLE_RATE: u32 = 44100;

/// Model constants derived from actual ONNX model inspection
/// vector_estimator expects noisy_latent: [batch, 144, latent_len]
/// vocoder expects latent: [batch, 144, latent_len]
const BASE_CHUNK_SIZE: usize = 512;
const CHUNK_COMPRESS_FACTOR: usize = 6;
const LATENT_SIZE: usize = BASE_CHUNK_SIZE * CHUNK_COMPRESS_FACTOR; // 3072
const LATENT_DIM: usize = 24;
const LATENT_CHANNELS: usize = LATENT_DIM * CHUNK_COMPRESS_FACTOR; // 144

/// Audio output from TTS
#[derive(Clone, Debug)]
pub struct AudioOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// Text-to-speech configuration
#[derive(Clone, Debug)]
pub struct TtsConfig {
    /// Number of flow-matching inference steps (higher = better quality, slower)
    pub num_inference_steps: u32,
    /// Speech speed multiplier (1.0 = normal)
    pub speed: f32,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            num_inference_steps: 5, // Official default (supertonic-2)
            speed: 1.05,            // Match Python default (slightly faster than 1.0)
        }
    }
}

// ============================================
// Mid-sentence gap detection
// ============================================
// The flow-matching denoiser starts from random noise, so each synthesis
// produces different audio. Occasionally it drops a word, leaving a silent
// gap in the middle of a sentence. We detect this by scanning for windows
// with near-zero energy within the "audible body" of the audio.

/// Info about a detected mid-sentence gap
#[derive(Debug, Clone)]
pub struct AudioGap {
    /// Start of gap in milliseconds
    pub start_ms: u32,
    /// End of gap in milliseconds
    pub end_ms: u32,
    /// Peak amplitude in the gap window
    pub peak: f32,
}

/// Minimum audio duration to bother checking for gaps (200ms)
const GAP_MIN_AUDIO_MS: f32 = 200.0;
/// Window size for gap scanning (50ms)
const GAP_WINDOW_MS: f32 = 50.0;
/// Peak amplitude below this in a window = silence
const GAP_SILENCE_PEAK: f32 = 0.02;
/// Number of consecutive silent windows to count as a gap
const GAP_MIN_SILENT_WINDOWS: usize = 2;
/// Amplitude threshold to find the "audible body" boundaries
const GAP_AUDIBLE_THRESHOLD: f32 = 0.01;
/// Skip this many ms from the start of audible region (onset tolerance)
const GAP_ONSET_SKIP_MS: f32 = 80.0;
/// Skip this many ms from the end of audible region (tail tolerance)
const GAP_TAIL_SKIP_MS: f32 = 80.0;

/// Detect a mid-sentence silent gap in synthesized audio.
///
/// Scans the "audible body" (between first and last audible samples,
/// with onset/tail margins trimmed) for consecutive silent windows.
/// Returns `Some(AudioGap)` if a suspicious gap is found, `None` if clean.
pub fn detect_mid_sentence_gap(samples: &[f32], sample_rate: u32) -> Option<AudioGap> {
    let duration_ms = samples.len() as f32 / sample_rate as f32 * 1000.0;
    if duration_ms < GAP_MIN_AUDIO_MS {
        return None; // Too short to have meaningful gaps
    }

    // Find audible body boundaries
    let first_audible = samples
        .iter()
        .position(|&s| s.abs() > GAP_AUDIBLE_THRESHOLD)?;
    let last_audible = samples
        .iter()
        .rposition(|&s| s.abs() > GAP_AUDIBLE_THRESHOLD)?;

    if last_audible <= first_audible {
        return None;
    }

    // Apply onset/tail margins
    let onset_skip_samples = (GAP_ONSET_SKIP_MS / 1000.0 * sample_rate as f32) as usize;
    let tail_skip_samples = (GAP_TAIL_SKIP_MS / 1000.0 * sample_rate as f32) as usize;
    let scan_start = first_audible.saturating_add(onset_skip_samples);
    let scan_end = last_audible.saturating_sub(tail_skip_samples);

    if scan_end <= scan_start {
        return None; // Body too short after margins
    }

    // Scan with sliding windows
    let window_size = (GAP_WINDOW_MS / 1000.0 * sample_rate as f32) as usize;
    if window_size == 0 {
        return None;
    }

    let mut consecutive_silent = 0usize;
    let mut gap_start_sample = 0usize;

    let mut pos = scan_start;
    while pos + window_size <= scan_end {
        let window = &samples[pos..pos + window_size];
        let w_peak = window.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        if w_peak < GAP_SILENCE_PEAK {
            if consecutive_silent == 0 {
                gap_start_sample = pos;
            }
            consecutive_silent += 1;

            if consecutive_silent >= GAP_MIN_SILENT_WINDOWS {
                let gap_end_sample = pos + window_size;
                return Some(AudioGap {
                    start_ms: (gap_start_sample as f32 / sample_rate as f32 * 1000.0) as u32,
                    end_ms: (gap_end_sample as f32 / sample_rate as f32 * 1000.0) as u32,
                    peak: w_peak,
                });
            }
        } else {
            consecutive_silent = 0;
        }

        pos += window_size;
    }

    None
}

/// Supertonic TTS engine
pub struct TextToSpeech {
    // ONNX sessions for the model pipeline
    duration_predictor: Session,
    text_encoder: Session,
    vector_estimator: Session,
    vocoder: Session,

    // Unicode tokenizer
    unicode_indexer: UnicodeIndexer,

    // Current voice style embeddings
    style_ttl: Option<Array3<f32>>,
    style_dp: Option<Array3<f32>>,
    style_ttl_shape: [usize; 3],
    style_dp_shape: [usize; 3],

    // Configuration
    config: TtsConfig,
}

impl TextToSpeech {
    /// Create a new TTS instance from model directory
    pub fn new<P: AsRef<Path>>(model_dir: P) -> Result<Self, TtsError> {
        let model_dir = model_dir.as_ref();

        log::info!("Loading Supertonic TTS models from {:?}", model_dir);

        // Load ONNX models (from Supertone/supertonic repo)
        let duration_predictor = Self::load_session(model_dir.join("duration_predictor.onnx"))?;
        let text_encoder = Self::load_session(model_dir.join("text_encoder.onnx"))?;
        let vector_estimator = Self::load_session(model_dir.join("vector_estimator.onnx"))?;
        let vocoder = Self::load_session(model_dir.join("vocoder.onnx"))?;

        // Load unicode indexer
        let unicode_indexer = UnicodeIndexer::from_file(model_dir.join("unicode_indexer.json"))?;

        log::info!("Supertonic TTS models loaded successfully");

        Ok(Self {
            duration_predictor,
            text_encoder,
            vector_estimator,
            vocoder,
            unicode_indexer,
            style_ttl: None,
            style_dp: None,
            style_ttl_shape: [1, 50, 256], // Default expected shape
            style_dp_shape: [1, 8, 16],    // Default expected shape
            config: TtsConfig::default(),
        })
    }

    fn load_session<P: AsRef<Path>>(path: P) -> Result<Session, TtsError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(TtsError::ModelNotFound(path.display().to_string()));
        }

        let builder = ort::session::Session::builder()
            .map_err(|e| TtsError::OrtError(e.to_string()))?
            .with_intra_threads(2)
            .map_err(|e| TtsError::OrtError(e.to_string()))?;

        // Try CUDA first, then fall back to CPU
        let cuda = ort::ep::CUDA::default();
        if cuda.is_available()? {
            log::info!("Using CUDA for TTS inference");
            let cuda_ep = cuda.build();
            builder
                .with_execution_providers([cuda_ep])
                .map_err(|e| TtsError::OrtError(format!("Session with CUDA failed: {}", e)))?
                .commit_from_file(path)
                .map_err(|e| TtsError::OrtError(format!("Session with CUDA failed: {}", e)))
        } else {
            // Fall back to CPU
            log::info!("CUDA not available, using CPU for TTS");
            builder
                .commit_from_file(path)
                .map_err(|e| TtsError::OrtError(format!("Session failed: {}", e)))
        }
    }

    /// Load voice style from VoiceStyleData
    pub fn set_voice_style(&mut self, voice_data: &VoiceStyleData) {
        self.style_ttl = Some(voice_data.style_ttl.clone());
        self.style_dp = Some(voice_data.style_dp.clone());
        self.style_ttl_shape = voice_data.style_ttl_shape();
        self.style_dp_shape = voice_data.style_dp_shape();

        log::info!(
            "Set voice style: {} (ttl: {:?}, dp: {:?})",
            voice_data.style.id,
            self.style_ttl_shape,
            self.style_dp_shape
        );
    }

    /// Load voice style from a JSON file
    pub fn load_voice_style<P: AsRef<Path>>(&mut self, path: P) -> Result<(), TtsError> {
        let path = path.as_ref();
        let voice_data = VoiceStyleData::from_json_file(path, "custom", "Custom Voice")
            .map_err(|e| TtsError::IoError(e.to_string()))?;
        self.set_voice_style(&voice_data);
        Ok(())
    }

    /// Set speech speed (0.5 = half speed, 2.0 = double speed)
    pub fn set_speed(&mut self, speed: f32) {
        self.config.speed = speed.clamp(0.25, 4.0);
    }

    /// Set number of inference steps
    pub fn set_inference_steps(&mut self, steps: u32) {
        self.config.num_inference_steps = steps.clamp(1, 20);
    }

    /// Synthesize text to audio (with automatic chunking for long text)
    ///
    /// Long text is automatically split into ~300 character chunks (matching official
    /// supertonic-2 implementation) to avoid exceeding the model's sequence length limit.
    pub fn synthesize(&mut self, text: &str) -> Result<AudioOutput, TtsError> {
        if text.is_empty() {
            return Ok(AudioOutput {
                samples: Vec::new(),
                sample_rate: SAMPLE_RATE,
            });
        }

        // Chunk long text to avoid sequence length limits
        // Official supertonic-2 uses 300 chars max (120 for Korean)
        const MAX_CHUNK_CHARS: usize = 300;
        let chunks = phonemizer::chunk_text(text, MAX_CHUNK_CHARS);

        if chunks.len() == 1 {
            // Short text - process directly (with gap retry)
            return self.synthesize_chunk_with_retry(&chunks[0]);
        }

        // Long text - synthesize each chunk and concatenate with inter-chunk silence
        log::info!("TTS: Splitting long text into {} chunks", chunks.len());
        let mut all_samples = Vec::new();

        // 0.3s silence between chunks (matches official implementation)
        let silence_samples = (0.3 * SAMPLE_RATE as f32) as usize;

        for (i, chunk) in chunks.iter().enumerate() {
            log::debug!(
                "TTS: Processing chunk {}/{} ({} chars)",
                i + 1,
                chunks.len(),
                chunk.len()
            );
            let audio = self.synthesize_chunk_with_retry(chunk)?;

            if i > 0 {
                // Add silence between chunks
                all_samples.extend(std::iter::repeat(0.0f32).take(silence_samples));
            }
            all_samples.extend(audio.samples);
        }

        log::info!(
            "TTS: Concatenated {} samples from {} chunks",
            all_samples.len(),
            chunks.len()
        );

        Ok(AudioOutput {
            samples: all_samples,
            sample_rate: SAMPLE_RATE,
        })
    }

    /// Synthesize a chunk with automatic retry if a mid-sentence gap is detected.
    ///
    /// Flow-matching denoising starts from random Gaussian noise, so each synthesis
    /// produces different audio. Occasionally the model drops a word, creating a
    /// silent gap mid-sentence. This wrapper detects such gaps and re-synthesizes
    /// (up to MAX_GAP_RETRIES times) to get a clean result.
    fn synthesize_chunk_with_retry(&mut self, text: &str) -> Result<AudioOutput, TtsError> {
        const MAX_GAP_RETRIES: u32 = 2;

        // Very short text (< 3 words) is unlikely to have mid-sentence gaps
        let word_count = text.split_whitespace().count();
        if word_count < 3 {
            return self.synthesize_chunk(text);
        }

        for attempt in 0..=MAX_GAP_RETRIES {
            let audio = self.synthesize_chunk(text)?;

            match detect_mid_sentence_gap(&audio.samples, audio.sample_rate) {
                None => return Ok(audio), // Clean — no gap detected
                Some(gap) => {
                    if attempt < MAX_GAP_RETRIES {
                        log::warn!(
                            "[TTS] Mid-sentence gap at {}-{}ms (peak={:.4}), retrying ({}/{}): '{}'",
                            gap.start_ms, gap.end_ms, gap.peak,
                            attempt + 1, MAX_GAP_RETRIES,
                            if text.len() > 40 { &text[..40] } else { text }
                        );
                        // Next call to synthesize_chunk will use a different RNG seed
                        continue;
                    } else {
                        log::warn!(
                            "[TTS] Mid-sentence gap persists after {} retries at {}-{}ms (peak={:.4}), using best effort: '{}'",
                            MAX_GAP_RETRIES, gap.start_ms, gap.end_ms, gap.peak,
                            if text.len() > 40 { &text[..40] } else { text }
                        );
                        return Ok(audio);
                    }
                }
            }
        }

        // Unreachable, but satisfy the compiler
        self.synthesize_chunk(text)
    }

    /// Synthesize a single chunk of text (internal, no chunking)
    fn synthesize_chunk(&mut self, text: &str) -> Result<AudioOutput, TtsError> {
        if text.is_empty() {
            return Ok(AudioOutput {
                samples: Vec::new(),
                sample_rate: SAMPLE_RATE,
            });
        }

        // Ensure we have voice styles loaded
        let style_ttl = self
            .style_ttl
            .as_ref()
            .ok_or(TtsError::NoSpeakerEmbeddings)?;
        let style_dp = self
            .style_dp
            .as_ref()
            .ok_or(TtsError::NoSpeakerEmbeddings)?;

        // Step 1: Convert text to token IDs (with language tags for v2 model)
        let (text_ids, text_mask_1d, seq_len) = self.unicode_indexer.text_to_ids(text, "en");

        if seq_len == 0 {
            return Ok(AudioOutput {
                samples: Vec::new(),
                sample_rate: SAMPLE_RATE,
            });
        }

        log::debug!("Text tokenized: {} tokens", seq_len);

        // Prepare tensors
        // text_ids: [batch, seq_len]
        let text_ids_tensor = Tensor::from_array(([1, seq_len], text_ids.clone()))
            .map_err(|e| TtsError::OrtError(e.to_string()))?;

        // text_mask: [batch, 1, seq_len] - 3D tensor for attention
        let text_mask_3d: Vec<f32> = text_mask_1d.clone();
        let text_mask_tensor = Tensor::from_array(([1, 1, seq_len], text_mask_3d.clone()))
            .map_err(|e| TtsError::OrtError(e.to_string()))?;

        // style_dp: [batch, dim1, dim2]
        let style_dp_flat: Vec<f32> = style_dp.iter().copied().collect();
        let style_dp_tensor = Tensor::from_array((
            [
                self.style_dp_shape[0],
                self.style_dp_shape[1],
                self.style_dp_shape[2],
            ],
            style_dp_flat.clone(),
        ))
        .map_err(|e| TtsError::OrtError(e.to_string()))?;

        // style_ttl: [batch, dim1, dim2]
        let style_ttl_flat: Vec<f32> = style_ttl.iter().copied().collect();
        let _style_ttl_tensor = Tensor::from_array((
            [
                self.style_ttl_shape[0],
                self.style_ttl_shape[1],
                self.style_ttl_shape[2],
            ],
            style_ttl_flat.clone(),
        ))
        .map_err(|e| TtsError::OrtError(e.to_string()))?;

        // Step 2: Run duration predictor
        // Inputs: text_ids, style_dp, text_mask
        let dp_outputs = self
            .duration_predictor
            .run(ort::inputs![
                "text_ids" => text_ids_tensor.view(),
                "style_dp" => style_dp_tensor.view(),
                "text_mask" => text_mask_tensor.view()
            ])
            .map_err(|e| TtsError::OrtError(format!("Duration predictor failed: {}", e)))?;

        // Get duration output (in seconds)
        let duration_output = dp_outputs
            .values()
            .next()
            .ok_or_else(|| TtsError::OrtError("No duration output".to_string()))?;
        let (dp_shape, duration_slice) = duration_output
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::OrtError(e.to_string()))?;
        let dp_shape_vec: Vec<i64> = dp_shape.to_vec();
        let dp_all_values: Vec<f32> = duration_slice.iter().copied().collect();

        log::debug!(
            "Duration predictor output: shape={:?}, {} elements",
            dp_shape_vec,
            dp_all_values.len()
        );

        let duration_seconds: f32 = if dp_all_values.len() > 1 && dp_all_values.len() == seq_len {
            // Per-token durations — sum them for total
            dp_all_values.iter().sum()
        } else {
            // Single scalar duration
            dp_all_values.first().copied().unwrap_or(1.0)
        };

        // Apply speed adjustment
        let adjusted_duration = duration_seconds / self.config.speed;
        log::debug!(
            "Duration: {:.2}s (adjusted: {:.2}s)",
            duration_seconds,
            adjusted_duration
        );

        // Step 3: Run text encoder
        // Need fresh tensors since the previous ones were consumed
        let text_ids_tensor2 = Tensor::from_array(([1, seq_len], text_ids.clone()))
            .map_err(|e| TtsError::OrtError(e.to_string()))?;
        let text_mask_tensor2 = Tensor::from_array(([1, 1, seq_len], text_mask_3d.clone()))
            .map_err(|e| TtsError::OrtError(e.to_string()))?;
        let style_ttl_tensor2 = Tensor::from_array((
            [
                self.style_ttl_shape[0],
                self.style_ttl_shape[1],
                self.style_ttl_shape[2],
            ],
            style_ttl_flat.clone(),
        ))
        .map_err(|e| TtsError::OrtError(e.to_string()))?;

        let te_outputs = self
            .text_encoder
            .run(ort::inputs![
                "text_ids" => text_ids_tensor2,
                "style_ttl" => style_ttl_tensor2.view(),
                "text_mask" => text_mask_tensor2.view()
            ])
            .map_err(|e| TtsError::OrtError(format!("Text encoder failed: {}", e)))?;

        // Get text embeddings
        let text_emb_output = te_outputs
            .values()
            .next()
            .ok_or_else(|| TtsError::OrtError("No text encoder output".to_string()))?;
        let (text_emb_shape, text_emb_slice) = text_emb_output
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::OrtError(e.to_string()))?;
        let text_emb: Vec<f32> = text_emb_slice.to_vec();
        let text_emb_shape: Vec<i64> = text_emb_shape.to_vec();

        log::debug!("Text embeddings shape: {:?}", text_emb_shape);

        // Step 4: Calculate latent dimensions from duration
        // latent_len = duration * sample_rate / latent_size
        let latent_len =
            ((adjusted_duration * SAMPLE_RATE as f32 / LATENT_SIZE as f32).ceil() as usize).max(1);
        log::debug!("Latent length: {}", latent_len);

        // Step 5: Initialize noisy latent with Gaussian noise
        // Shape: [batch, LATENT_CHANNELS (144), latent_len]
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();
        let mut latent: Vec<f32> = (0..LATENT_CHANNELS * latent_len)
            .map(|_| normal.sample(&mut rng))
            .collect();

        // Create latent mask: [batch, 1, latent_len]
        let latent_mask: Vec<f32> = vec![1.0; latent_len];

        // Step 6: Iterative denoising with flow-matching
        // The model internally handles the denoising schedule via current_step/total_step
        let num_steps = self.config.num_inference_steps;

        for step in 0..num_steps {
            // Prepare tensors for this iteration
            let latent_tensor =
                Tensor::from_array(([1, LATENT_CHANNELS, latent_len], latent.clone()))
                    .map_err(|e| TtsError::OrtError(e.to_string()))?;

            let text_emb_tensor = Tensor::from_array((
                [
                    text_emb_shape[0] as usize,
                    text_emb_shape[1] as usize,
                    text_emb_shape[2] as usize,
                ],
                text_emb.clone(),
            ))
            .map_err(|e| TtsError::OrtError(e.to_string()))?;

            let style_ttl_tensor3 = Tensor::from_array((
                [
                    self.style_ttl_shape[0],
                    self.style_ttl_shape[1],
                    self.style_ttl_shape[2],
                ],
                style_ttl_flat.clone(),
            ))
            .map_err(|e| TtsError::OrtError(e.to_string()))?;

            let latent_mask_tensor = Tensor::from_array(([1, 1, latent_len], latent_mask.clone()))
                .map_err(|e| TtsError::OrtError(e.to_string()))?;

            let text_mask_tensor3 = Tensor::from_array(([1, 1, seq_len], text_mask_3d.clone()))
                .map_err(|e| TtsError::OrtError(e.to_string()))?;

            let current_step_tensor = Tensor::from_array(([1], vec![step as f32]))
                .map_err(|e| TtsError::OrtError(e.to_string()))?;

            let total_step_tensor = Tensor::from_array(([1], vec![num_steps as f32]))
                .map_err(|e| TtsError::OrtError(e.to_string()))?;

            // Run vector estimator
            let ve_outputs = self
                .vector_estimator
                .run(ort::inputs![
                    "noisy_latent" => latent_tensor,
                    "text_emb" => text_emb_tensor,
                    "style_ttl" => style_ttl_tensor3,
                    "latent_mask" => latent_mask_tensor,
                    "text_mask" => text_mask_tensor3,
                    "current_step" => current_step_tensor,
                    "total_step" => total_step_tensor
                ])
                .map_err(|e| {
                    TtsError::OrtError(format!("Vector estimator step {} failed: {}", step, e))
                })?;

            // Get updated latent from vector estimator
            // NOTE: The model returns the updated latent directly, NOT a velocity field
            // (unlike traditional flow-matching which requires Euler integration)
            let updated_latent_output = ve_outputs
                .values()
                .next()
                .ok_or_else(|| TtsError::OrtError("No vector estimator output".to_string()))?;
            let (_, updated_latent_slice) = updated_latent_output
                .try_extract_tensor::<f32>()
                .map_err(|e| TtsError::OrtError(e.to_string()))?;

            // Directly replace latent with model output (no Euler integration)
            latent = updated_latent_slice.to_vec();

            log::debug!("Denoising step {}/{} complete", step + 1, num_steps);
        }

        // Step 7: Run vocoder to generate audio
        // Input: latent [batch, LATENT_CHANNELS=144, latent_len]
        let latent_tensor = Tensor::from_array(([1, LATENT_CHANNELS, latent_len], latent))
            .map_err(|e| TtsError::OrtError(e.to_string()))?;

        let vocoder_outputs = self
            .vocoder
            .run(ort::inputs!["latent" => latent_tensor])
            .map_err(|e| TtsError::OrtError(format!("Vocoder failed: {}", e)))?;

        let audio_output = vocoder_outputs
            .values()
            .next()
            .ok_or_else(|| TtsError::OrtError("No vocoder output".to_string()))?;
        let (_, audio_slice) = audio_output
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::OrtError(e.to_string()))?;
        let raw_samples: Vec<f32> = audio_slice.to_vec();

        // Truncate vocoder output to predicted duration (matching official implementation).
        // The vocoder produces latent_len * LATENT_SIZE samples, but only
        // duration * sample_rate samples are actual audio — the rest is padding noise
        // that causes "doubling" artifacts if included.
        let wav_len = (adjusted_duration * SAMPLE_RATE as f32) as usize;
        let samples: Vec<f32> = raw_samples[..wav_len.min(raw_samples.len())].to_vec();

        log::info!(
            "TTS: Synthesized {} samples ({:.2}s) from {} raw, truncated to predicted duration {:.2}s, {} steps",
            samples.len(),
            samples.len() as f32 / SAMPLE_RATE as f32,
            raw_samples.len(),
            adjusted_duration,
            num_steps
        );

        Ok(AudioOutput {
            samples,
            sample_rate: SAMPLE_RATE,
        })
    }

    /// Get the output sample rate
    pub fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Get available voice styles
    pub fn available_voices() -> Vec<VoiceStyle> {
        vec![
            VoiceStyle {
                id: "F1".to_string(),
                name: "Female 1 (Default)".to_string(),
            },
            VoiceStyle {
                id: "F2".to_string(),
                name: "Female 2".to_string(),
            },
            VoiceStyle {
                id: "M1".to_string(),
                name: "Male 1".to_string(),
            },
            VoiceStyle {
                id: "M2".to_string(),
                name: "Male 2".to_string(),
            },
        ]
    }
}

/// TTS Error types
#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("ONNX Runtime error: {0}")]
    OrtError(String),

    #[error("IO error: {0}")]
    IoError(String),

    #[error("Shape error: {0}")]
    ShapeError(String),

    #[error("No speaker embeddings set - call set_voice_style() first")]
    NoSpeakerEmbeddings,

    #[error("Phonemization error: {0}")]
    PhonemizationError(String),
}

impl From<ort::Error> for TtsError {
    fn from(e: ort::Error) -> Self {
        TtsError::OrtError(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_available_voices() {
        let voices = TextToSpeech::available_voices();
        assert!(!voices.is_empty());
        assert!(voices.iter().any(|v| v.id == "F1"));
    }

    #[test]
    fn test_config_defaults() {
        let config = TtsConfig::default();
        assert_eq!(config.num_inference_steps, 4);
        assert_eq!(config.speed, 1.05);
    }

    /// Helper: generate a sine wave segment
    fn sine_wave(sample_rate: u32, duration_ms: u32, amplitude: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_ms as f32 / 1000.0) as usize;
        (0..n)
            .map(|i| {
                amplitude
                    * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin()
            })
            .collect()
    }

    /// Helper: generate silence
    fn silence(sample_rate: u32, duration_ms: u32) -> Vec<f32> {
        vec![0.0; (sample_rate as f32 * duration_ms as f32 / 1000.0) as usize]
    }

    #[test]
    fn gap_detection_clean_audio_passes() {
        // Continuous speech-like audio: 100ms onset silence + 1s of signal
        let sr = 44100;
        let mut samples = silence(sr, 100);
        samples.extend(sine_wave(sr, 1000, 0.3));
        samples.extend(silence(sr, 50)); // trailing silence

        assert!(detect_mid_sentence_gap(&samples, sr).is_none());
    }

    #[test]
    fn gap_detection_finds_mid_sentence_gap() {
        // Speech with a 150ms silent gap in the middle (3 x 50ms windows)
        let sr = 44100;
        let mut samples = silence(sr, 50); // onset
        samples.extend(sine_wave(sr, 300, 0.3)); // "What"
        samples.extend(silence(sr, 150)); // GAP where "would" should be
        samples.extend(sine_wave(sr, 500, 0.3)); // "you like to hear"
        samples.extend(silence(sr, 50)); // trailing

        let gap = detect_mid_sentence_gap(&samples, sr);
        assert!(gap.is_some(), "Should detect the 150ms silent gap");
        let gap = gap.unwrap();
        // Gap should be roughly in the 350-500ms range (after onset + "What")
        assert!(
            gap.start_ms >= 200,
            "Gap start should be after onset, got {}ms",
            gap.start_ms
        );
        assert!(
            gap.end_ms <= 600,
            "Gap end should be before tail, got {}ms",
            gap.end_ms
        );
    }

    #[test]
    fn gap_detection_ignores_onset_silence() {
        // 200ms onset silence followed by clean audio — should NOT trigger
        let sr = 44100;
        let mut samples = silence(sr, 200); // long onset
        samples.extend(sine_wave(sr, 800, 0.3));

        assert!(detect_mid_sentence_gap(&samples, sr).is_none());
    }

    #[test]
    fn gap_detection_ignores_trailing_silence() {
        // Clean audio with long trailing silence — should NOT trigger
        let sr = 44100;
        let mut samples = sine_wave(sr, 800, 0.3);
        samples.extend(silence(sr, 300)); // long tail

        assert!(detect_mid_sentence_gap(&samples, sr).is_none());
    }

    #[test]
    fn gap_detection_skips_short_audio() {
        // Audio shorter than GAP_MIN_AUDIO_MS — should NOT scan
        let sr = 44100;
        let samples = silence(sr, 150); // only 150ms

        assert!(detect_mid_sentence_gap(&samples, sr).is_none());
    }

    #[test]
    fn gap_detection_single_silent_window_ok() {
        // A single 50ms dip is normal prosody, not a dropped word
        let sr = 44100;
        let mut samples = silence(sr, 50);
        samples.extend(sine_wave(sr, 300, 0.3));
        samples.extend(silence(sr, 50)); // single dip
        samples.extend(sine_wave(sr, 500, 0.3));

        assert!(detect_mid_sentence_gap(&samples, sr).is_none());
    }
}
