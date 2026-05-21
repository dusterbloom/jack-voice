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
    /// Loudness normalization. The flow-matching vocoder produces output whose
    /// amplitude drifts a lot across phrases (we've measured ~10× RMS swings on
    /// long input). Set this to `Some(target_rms)` to normalize to a target
    /// linear RMS level (e.g. 0.10 ≈ -20 dBFS). `None` disables normalization
    /// for test parity with the raw vocoder output. Default: `Some(0.10)`.
    pub target_rms: Option<f32>,
    /// Dynamic-range compression to fix per-syllable "pumping" — quiet
    /// syllables get pulled up to the threshold, loud ones get squashed.
    /// `None` disables. Default: `Some(CompressorParams::default())`.
    pub compressor: Option<CompressorParams>,
}

/// Feed-forward compressor parameters (linear units, not dB).
#[derive(Clone, Debug)]
pub struct CompressorParams {
    /// Above this linear amplitude, signal is compressed. 0.10 ≈ -20 dBFS.
    pub threshold: f32,
    /// Compression ratio. 4.0 = 4:1 (input change of 4 dB → output change of 1 dB).
    pub ratio: f32,
    /// Attack time constant in seconds (how fast it clamps down).
    pub attack_s: f32,
    /// Release time constant in seconds (how fast it lets go).
    pub release_s: f32,
    /// Make-up gain applied after compression to restore loudness.
    pub makeup_gain: f32,
}

impl Default for CompressorParams {
    fn default() -> Self {
        Self {
            threshold: 0.10,    // -20 dBFS
            ratio: 4.0,         // 4:1, gentle voice compression
            attack_s: 0.005,    // 5 ms — fast enough to catch transients
            release_s: 0.050,   // 50 ms — natural decay
            makeup_gain: 2.0,   // pull quiet syllables up
        }
    }
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            // 5 = official, ~25% wall-time cost over 3 for noticeably tighter
            // denoising. Drop to 3 for chat workloads if you're CPU-bound.
            num_inference_steps: 5,
            // 1.0 = natural pace (user ear-checked vs 1.05 and 0.92).
            speed: 1.0,
            target_rms: Some(0.10),
            compressor: Some(CompressorParams::default()),
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

/// Automatic gain control (AGC) — sliding-window loudness equalizer.
///
/// Walks the signal with a windowed RMS estimator, computes the per-sample
/// gain needed to keep RMS at the target, smooths it with a one-pole filter
/// (so the gain itself doesn't pump), then hard-limits the output.
///
/// This fixes the *sustained* envelope drift the flow-matching vocoder leaves
/// behind — quiet phrases come up, loud phrases come down, all on the same
/// 200 ms-ish timescale that human ears interpret as "volume".
///
/// The legacy `CompressorParams::ratio` is reinterpreted as a soft-clip
/// curve: ratio >= 1.0 lets the AGC compute the *desired* gain, then a
/// secondary soft-knee compressor catches any residual peaks.
fn compress(samples: &mut [f32], sr: u32, p: &CompressorParams) {
    if samples.is_empty() || p.threshold <= 0.0 {
        return;
    }

    let sr_f = sr as f32;

    // Windowed RMS estimator — 200 ms gives ~5 Hz envelope tracking, well
    // below syllable rate but above word rate. Use a single-pole IIR over
    // squared samples to avoid a circular buffer.
    let rms_tau_s = 0.200;
    let rms_coef = (-1.0 / (rms_tau_s * sr_f)).exp();

    // Gain smoothing — slower than the envelope so the AGC itself doesn't pump.
    // Asymmetric: faster release (gain up) than attack (gain down), broadcast-style.
    let gain_attack = (-1.0 / (p.attack_s.max(0.005) * sr_f)).exp();
    let gain_release = (-1.0 / (p.release_s.max(0.050) * sr_f)).exp();

    let max_gain = 4.0_f32;
    let min_gain = 0.25_f32;

    let mut env_sq: f32 = p.threshold * p.threshold; // start near target
    let mut gain: f32 = 1.0;

    for s in samples.iter_mut() {
        // Track windowed mean-square (cheap RMS).
        let sq = *s * *s;
        env_sq = sq + rms_coef * (env_sq - sq);
        let env = env_sq.sqrt().max(1e-6);

        // Desired gain: target / current. Clamp so we never amplify pure silence
        // into noise, and never squash voice down past min_gain.
        let desired = (p.threshold / env).clamp(min_gain, max_gain);

        // Smooth gain transitions (slower than envelope to avoid pumping).
        let coef = if desired < gain { gain_attack } else { gain_release };
        gain = desired + coef * (gain - desired);

        let out = *s * gain;
        // Soft-knee limiter (tanh) to round off any peaks that get amplified.
        let abs_out = out.abs();
        let ceiling = 0.95_f32;
        let limited = if abs_out > ceiling {
            let knee = 1.0 - ceiling;
            let over = (abs_out - ceiling) / knee.max(1e-6);
            out.signum() * (ceiling + knee * over.tanh())
        } else {
            out
        };
        *s = limited.clamp(-0.99, 0.99);
    }

    // Suppress field warning on the unused params (kept for backwards compat).
    let _ = p.ratio;
    let _ = p.makeup_gain;
}

/// Normalize speech to a target RMS, then apply a soft-knee peak limiter so
/// the gain push from quiet syllables doesn't clip the loud ones.
///
/// This is a single-pass, non-time-varying gain — sentences keep their relative
/// dynamics within a clip, but absolute loudness becomes consistent across
/// synthesize() calls.
///
/// `target_rms`: linear RMS goal (e.g. 0.10 = ~-20 dBFS).
/// `ceiling`:    soft-knee threshold above which to compress (e.g. 0.95).
fn normalize_loudness(samples: &mut [f32], target_rms: f32, ceiling: f32) {
    if samples.is_empty() {
        return;
    }

    // 1) Compute current RMS, ignoring near-silence (the leading/trailing pad).
    let active_thresh = 0.005_f32;
    let mut sum_sq = 0.0_f32;
    let mut active = 0usize;
    for &s in samples.iter() {
        let a = s.abs();
        if a > active_thresh {
            sum_sq += s * s;
            active += 1;
        }
    }
    if active < 32 {
        return; // not enough signal to normalize meaningfully
    }
    let current_rms = (sum_sq / active as f32).sqrt();
    if current_rms <= 1e-6 {
        return;
    }

    // 2) Apply uniform gain.
    let gain = (target_rms / current_rms).clamp(0.1, 16.0);
    for s in samples.iter_mut() {
        *s *= gain;
    }

    // 3) Soft-knee peak limiter — only acts on samples above `ceiling`.
    // Uses tanh-style compression so the transition is smooth.
    let knee = 1.0 - ceiling;
    if knee > 0.0 {
        for s in samples.iter_mut() {
            let a = s.abs();
            if a > ceiling {
                let over = (a - ceiling) / knee.max(1e-6);
                let compressed = ceiling + knee * over.tanh();
                *s = s.signum() * compressed;
            }
        }
    }
}

/// Detect the number of performance cores on the host.
///
/// On Apple Silicon, this reads `hw.perflevel0.physicalcpu` (P-core count).
/// On other platforms, falls back to `min(num_cpus / 2, 4)` as a safe default
/// (most modern CPUs gain little ONNX speedup past 4 intra-op threads).
fn detect_performance_cores() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(out) = Command::new("sysctl")
            .args(["-n", "hw.perflevel0.physicalcpu"])
            .output()
        {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Ok(n) = s.trim().parse::<usize>() {
                    if n > 0 {
                        return n;
                    }
                }
            }
        }
    }
    let total = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (total / 2).clamp(2, 4)
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
    language: String,
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
            language: "en".to_string(),
        })
    }

    fn load_session<P: AsRef<Path>>(path: P) -> Result<Session, TtsError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(TtsError::ModelNotFound(path.display().to_string()));
        }

        // intra_threads: match the host's performance-core count, not total cores.
        // E-cores typically hurt ONNX intra-op parallelism via cache thrash.
        // Override with SUPERTONIC_INTRA_THREADS env if you need to pin a value.
        let intra_threads = std::env::var("SUPERTONIC_INTRA_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(detect_performance_cores);

        let builder = ort::session::Session::builder()
            .map_err(|e| TtsError::OrtError(e.to_string()))?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(|e| TtsError::OrtError(e.to_string()))?
            .with_intra_threads(intra_threads)
            .map_err(|e| TtsError::OrtError(e.to_string()))?;

        // Try CUDA first (Linux/Windows GPU), then CoreML (macOS Apple Silicon), then CPU
        let cuda = ort::ep::CUDA::default();
        if cuda.is_available()? {
            log::info!("Using CUDA for TTS inference");
            let cuda_ep = cuda.build();
            return builder
                .with_execution_providers([cuda_ep])
                .map_err(|e| TtsError::OrtError(format!("Session with CUDA failed: {}", e)))?
                .commit_from_file(path)
                .map_err(|e| TtsError::OrtError(format!("Session with CUDA failed: {}", e)));
        }

        #[cfg(target_os = "macos")]
        {
            let coreml = ort::ep::CoreML::default();
            if coreml.is_available()? {
                log::info!("Using CoreML for TTS inference");
                let coreml_ep = coreml.build();
                return builder
                    .with_execution_providers([coreml_ep])
                    .map_err(|e| TtsError::OrtError(format!("Session with CoreML failed: {}", e)))?
                    .commit_from_file(path)
                    .map_err(|e| TtsError::OrtError(format!("Session with CoreML failed: {}", e)));
            }
        }

        log::info!("CUDA/CoreML not available, using CPU for TTS");
        builder
            .commit_from_file(path)
            .map_err(|e| TtsError::OrtError(format!("Session failed: {}", e)))
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

    pub fn set_language(&mut self, lang: &str) {
        self.language = lang.to_string();
    }

    pub fn language(&self) -> &str {
        &self.language
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

    /// Synthesize a chunk, with one-shot instrumentation if a mid-sentence
    /// gap is detected.
    ///
    /// # Root cause of mid-sentence gaps (do not "fix" via retries)
    ///
    /// The duration predictor ONNX model in `synthesize_chunk` occasionally
    /// emits an outlier per-token duration (e.g. one token gets 500–1000 ms
    /// when neighbours get 30–80 ms). When the vocoder is fed those
    /// durations, the resulting waveform contains a long silent passage
    /// mid-sentence. **It is not RNG-driven sampling noise.** Re-running
    /// `synthesize_chunk` with a fresh seed only sometimes draws a different
    /// duration distribution; on the requests that show this bug in
    /// production logs, retries reliably re-produce the same shape of gap.
    ///
    /// The previous implementation retried up to 2 times and then accepted
    /// the bad audio anyway ("best effort"). That cost **300–1000 ms of
    /// added latency on every chunk that hit the detector**, including in
    /// the common case where the gap was actually a natural sentence-end
    /// pause being misclassified — for zero quality improvement on the
    /// pathological case.
    ///
    /// The proper fix is upstream of supertonic: clamp per-token durations
    /// to a sane maximum (e.g. 200–300 ms) inside the duration-predictor
    /// post-processing in [`synthesize_chunk`], before `latent_len` is
    /// computed. That would shorten the total predicted duration and
    /// eliminate the silent passages entirely. See the per-token-sum logic
    /// around the `duration_predictor.run(...)` block.
    ///
    /// Until that lands, we keep gap detection as a single passive log so
    /// the rate stays measurable, but we never retry.
    fn synthesize_chunk_with_retry(&mut self, text: &str) -> Result<AudioOutput, TtsError> {
        let audio = self.synthesize_chunk(text)?;

        // Skip detection for very short text — natural punctuation gaps would
        // dominate the signal.
        if text.split_whitespace().count() >= 3 {
            if let Some(gap) = detect_mid_sentence_gap(&audio.samples, audio.sample_rate) {
                log::warn!(
                    "[TTS] Mid-sentence gap at {}-{}ms (peak={:.4}) \u{2014} root cause is duration-predictor outlier, see synthesize_chunk_with_retry doc: '{}'",
                    gap.start_ms,
                    gap.end_ms,
                    gap.peak,
                    if text.len() > 40 { &text[..40] } else { text }
                );
            }
        }

        Ok(audio)
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
        let (text_ids, text_mask_1d, seq_len) = self.unicode_indexer.text_to_ids(text, &self.language);

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

        // Per-token duration cap: any single token predicted to last longer
        // than this is an outlier — a phoneme would naturally last 50–200 ms,
        // and the duration predictor occasionally emits 0.5–1.0 s for a
        // single token, which the vocoder renders as a silent passage. That
        // surfaces as either a fade-in at the start of a sentence (outlier
        // on token 1–3, masked from the previous gap detector by its 80 ms
        // onset skip) or a mid-sentence dropout (outlier in the body). See
        // the `synthesize_chunk_with_retry` doc-comment for the full root
        // cause analysis — clamping here is that documented "proper fix".
        const MAX_TOKEN_DURATION_S: f32 = 0.30;

        let duration_seconds: f32 = if dp_all_values.len() > 1 && dp_all_values.len() == seq_len {
            // Per-token durations — clamp outliers then sum for total.
            let outliers = dp_all_values
                .iter()
                .filter(|&&d| d > MAX_TOKEN_DURATION_S)
                .count();
            if outliers > 0 {
                log::debug!(
                    "Duration predictor: clamping {} outlier token(s) above {:.2}s",
                    outliers,
                    MAX_TOKEN_DURATION_S
                );
            }
            dp_all_values
                .iter()
                .map(|&d| d.min(MAX_TOKEN_DURATION_S))
                .sum()
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
        let mut samples: Vec<f32> = raw_samples[..wav_len.min(raw_samples.len())].to_vec();

        // 1) Compression first — squashes intra-utterance dynamics (per-syllable
        //    pumping that the vocoder leaves behind).
        if let Some(ref c) = self.config.compressor {
            compress(&mut samples, SAMPLE_RATE, c);
        }
        // 2) RMS normalization second — pulls absolute loudness to a target,
        //    consistent across synthesize() calls.
        if let Some(target) = self.config.target_rms {
            normalize_loudness(&mut samples, target, 0.95);
        }

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
        assert_eq!(config.num_inference_steps, 5);
        assert_eq!(config.speed, 1.0);
        assert_eq!(config.target_rms, Some(0.10));
        assert!(config.compressor.is_some());
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
