use log;
use ort::value::Tensor;
use std::time::Instant;

use crate::models;
use crate::models::ModelError;
use crate::vad::VadError;

const SAMPLE_RATE: u32 = 16000;
/// SmartTurn analyzes up to 8 seconds of recent audio (per Pipecat docs)
const MAX_ANALYSIS_SECS: usize = 8;
const MAX_ANALYSIS_SAMPLES: usize = MAX_ANALYSIS_SECS * SAMPLE_RATE as usize;
/// Maximum turn duration before force-completing (5 minutes)
const MAX_TURN_DURATION_SECS: u64 = 300;
const MAX_TURN_SAMPLES: usize = MAX_TURN_DURATION_SECS as usize * SAMPLE_RATE as usize;
/// Silence timeout before checking SmartTurn (Pipecat recommends VAD stop_secs=0.2,
/// but SmartTurn needs a bit more silence to be useful)
const SILENCE_TIMEOUT_MS: u64 = 1200;
const SMART_TURN_THRESHOLD: f32 = 0.70;
/// Lookback buffer: 300ms of audio kept before speech detection (like Moonshine's LOOKBACK_CHUNKS)
/// This captures the speech onset that VAD misses
const LOOKBACK_MS: usize = 300;
const LOOKBACK_SAMPLES: usize = LOOKBACK_MS * SAMPLE_RATE as usize / 1000; // 4800 samples

pub struct TurnDetector {
    smart_turn: Option<SmartTurnModel>,
    /// Audio accumulated during the current turn (includes lookback)
    turn_buffer: Vec<f32>,
    /// Rolling lookback buffer — always keeps the last 300ms of audio
    /// regardless of speech state. When speech starts, this is seeded
    /// into turn_buffer so the beginning of speech isn't clipped.
    lookback_buffer: Vec<f32>,
    is_speaking: bool,
    speech_start_time: Option<Instant>,
    silence_timer: Option<Instant>,
    /// Track last buffer size evaluated by SmartTurn to prevent redundant evaluations
    last_evaluated_size: usize,
}

struct SmartTurnModel {
    session: ort::session::Session,
    #[allow(dead_code)]
    threshold: f32,
    output_name: String,
}

impl SmartTurnModel {
    pub fn load() -> Result<Self, SmartTurnError> {
        let model_path = models::get_model_path("smart-turn-v3.2-cpu.onnx")?;

        if !model_path.exists() {
            return Err(SmartTurnError::ModelNotFound(
                model_path.display().to_string(),
            ));
        }

        log::info!("Loading Smart Turn model from: {:?}", model_path);

        let builder = ort::session::Session::builder()
            .map_err(|e| SmartTurnError::InitError(e.to_string()))?
            .with_intra_threads(2)
            .map_err(|e| SmartTurnError::InitError(e.to_string()))?;

        log::info!("Using CPU for Smart Turn (lightweight model)");
        let session = builder
            .commit_from_file(&model_path)
            .map_err(|e| SmartTurnError::InitError(format!("Failed to load model: {}", e)))?;

        let output_name = session
            .outputs()
            .first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "output".to_string());

        log::info!(
            "Smart Turn model loaded successfully (output: {})",
            output_name
        );

        Ok(Self {
            session,
            threshold: SMART_TURN_THRESHOLD,
            output_name,
        })
    }

    pub fn predict(&mut self, audio: &[f32]) -> Result<f32, SmartTurnError> {
        // SmartTurn analyzes up to 8 seconds of recent audio
        let analysis_audio = if audio.len() > MAX_ANALYSIS_SAMPLES {
            &audio[audio.len() - MAX_ANALYSIS_SAMPLES..]
        } else {
            audio
        };

        let prepared = prepare_mel_spectrogram(analysis_audio)?;

        let input_tensor = Tensor::from_array(([1, 80, 800], prepared)).map_err(|e| {
            SmartTurnError::InferenceError(format!("Tensor creation failed: {}", e))
        })?;

        let outputs = self
            .session
            .run(ort::inputs!["input_features" => input_tensor])
            .map_err(|e| SmartTurnError::InferenceError(format!("Inference failed: {}", e)))?;

        let output = outputs.get(&self.output_name).ok_or_else(|| {
            SmartTurnError::InferenceError(format!("No output '{}' from model", self.output_name))
        })?;

        let (_, data) = output.try_extract_tensor::<f32>().map_err(|e| {
            SmartTurnError::InferenceError(format!("Output extraction failed: {}", e))
        })?;

        let logit = *data
            .first()
            .ok_or_else(|| SmartTurnError::InferenceError("Empty output tensor".to_string()))?;

        let probability = 1.0 / (1.0 + (-logit).exp());
        Ok(probability)
    }
}

// ============================================
// Mel Spectrogram (optimized with Radix-2 FFT)
// ============================================

const N_FFT: usize = 512; // Next power of 2 from 400, required for radix-2 FFT
const HOP_LENGTH: usize = 160;
const N_MELS: usize = 80;
const N_FRAMES: usize = 800;
const MEL_FLOOR: f32 = 1e-10;

fn prepare_mel_spectrogram(audio: &[f32]) -> Result<Vec<f32>, SmartTurnError> {
    if audio.is_empty() {
        return Err(SmartTurnError::InferenceError(
            "Empty audio input".to_string(),
        ));
    }

    // Right-align audio in a buffer sized for exactly N_FRAMES of mel output
    // N_FRAMES * HOP_LENGTH = 800 * 160 = 128,000 samples = 8 seconds
    let required_samples = N_FRAMES * HOP_LENGTH;
    let mut padded = vec![0.0f32; required_samples];

    if audio.len() >= required_samples {
        padded.copy_from_slice(&audio[audio.len() - required_samples..]);
    } else {
        let start = required_samples - audio.len();
        padded[start..].copy_from_slice(audio);
    }

    let mel = audio_to_mel_spectrogram(&padded);
    Ok(mel)
}

fn audio_to_mel_spectrogram(audio: &[f32]) -> Vec<f32> {
    #[allow(dead_code)]
    const PI: f32 = std::f32::consts::PI;

    let window = hann_window(N_FFT);
    let filterbank = create_mel_filterbank();
    let n_freqs = N_FFT / 2 + 1;
    let mut mel_spec = vec![0.0f32; N_MELS * N_FRAMES];

    // Pre-compute twiddle factors for FFT
    let (tw_re, tw_im) = precompute_twiddle_factors(N_FFT);

    let mut frame_buf = vec![0.0f32; N_FFT];

    for frame_idx in 0..N_FRAMES {
        let start = frame_idx * HOP_LENGTH;
        let end = (start + N_FFT).min(audio.len());

        // Zero-fill frame buffer
        for v in frame_buf.iter_mut() {
            *v = 0.0;
        }

        let copy_len = end.saturating_sub(start);
        if start < audio.len() && copy_len > 0 {
            // Apply Hann window while copying
            for i in 0..copy_len {
                frame_buf[i] = audio[start + i] * window[i];
            }
        }

        // Compute FFT magnitude
        let stft_mag = fft_magnitude(&frame_buf, &tw_re, &tw_im);

        // Apply mel filterbank
        for mel_idx in 0..N_MELS {
            let mut mel_energy = 0.0f32;
            for freq_idx in 0..n_freqs {
                mel_energy += filterbank[mel_idx][freq_idx] * stft_mag[freq_idx];
            }
            mel_spec[mel_idx * N_FRAMES + frame_idx] = (mel_energy.max(MEL_FLOOR)).ln();
        }
    }

    // Normalize
    for val in mel_spec.iter_mut() {
        *val = ((*val + 4.0) / 4.0).clamp(-1.0, 1.0);
    }

    mel_spec
}

fn hann_window(size: usize) -> Vec<f32> {
    const PI: f32 = std::f32::consts::PI;
    (0..size)
        .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / size as f32).cos()))
        .collect()
}

fn create_mel_filterbank() -> Vec<Vec<f32>> {
    let n_freqs = N_FFT / 2 + 1;
    let fmax = SAMPLE_RATE as f32 / 2.0;

    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).ln() / std::f32::consts::LN_10
    }

    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
    }

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(fmax);

    let mel_points: Vec<f32> = (0..=N_MELS + 1)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&hz| (N_FFT as f32 + 1.0) * hz / SAMPLE_RATE as f32)
        .collect();

    let mut filterbank = vec![vec![0.0f32; n_freqs]; N_MELS];
    for m in 0..N_MELS {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];
        for k in 0..n_freqs {
            let k_f = k as f32;
            if k_f >= f_left && k_f < f_center {
                filterbank[m][k] = (k_f - f_left) / (f_center - f_left);
            } else if k_f >= f_center && k_f <= f_right {
                filterbank[m][k] = (f_right - k_f) / (f_right - f_center);
            }
        }
    }
    filterbank
}

/// Pre-compute twiddle factors for Radix-2 FFT
fn precompute_twiddle_factors(n: usize) -> (Vec<f32>, Vec<f32>) {
    const PI: f32 = std::f32::consts::PI;
    let half = n / 2;
    let mut tw_re = Vec::with_capacity(half);
    let mut tw_im = Vec::with_capacity(half);
    for k in 0..half {
        let angle = -2.0 * PI * k as f32 / n as f32;
        tw_re.push(angle.cos());
        tw_im.push(angle.sin());
    }
    (tw_re, tw_im)
}

/// Radix-2 Cooley-Tukey FFT — O(N log N) instead of naive O(N²)
/// Returns magnitude spectrum (first N/2+1 bins)
fn fft_magnitude(input: &[f32], tw_re: &[f32], tw_im: &[f32]) -> Vec<f32> {
    let n = input.len();
    debug_assert!(n.is_power_of_two(), "FFT size must be power of 2");

    let mut real = input.to_vec();
    let mut imag = vec![0.0f32; n];

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            real.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Cooley-Tukey butterfly
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = n / len;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let tw_idx = k * step;
                let i = start + k;
                let j_idx = start + k + half;

                let tr = real[j_idx] * tw_re[tw_idx] - imag[j_idx] * tw_im[tw_idx];
                let ti = real[j_idx] * tw_im[tw_idx] + imag[j_idx] * tw_re[tw_idx];

                real[j_idx] = real[i] - tr;
                imag[j_idx] = imag[i] - ti;
                real[i] += tr;
                imag[i] += ti;
            }
        }
        len <<= 1;
    }

    // Compute magnitudes for first N/2+1 bins
    let n_freqs = n / 2 + 1;
    let mut magnitude = Vec::with_capacity(n_freqs);
    for k in 0..n_freqs {
        magnitude.push((real[k] * real[k] + imag[k] * imag[k]).sqrt());
    }
    magnitude
}

// ============================================
// SmartTurn Errors
// ============================================

#[derive(Debug, thiserror::Error)]
pub enum SmartTurnError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Initialization error: {0}")]
    InitError(String),
    #[error("Inference error: {0}")]
    InferenceError(String),
}

impl From<VadError> for SmartTurnError {
    fn from(e: VadError) -> Self {
        SmartTurnError::InitError(e.to_string())
    }
}

impl From<ModelError> for SmartTurnError {
    fn from(e: ModelError) -> Self {
        SmartTurnError::InitError(e.to_string())
    }
}

impl From<ort::Error> for SmartTurnError {
    fn from(e: ort::Error) -> Self {
        SmartTurnError::InitError(e.to_string())
    }
}

// ============================================
// TurnDetector with rolling lookback buffer
// ============================================

impl TurnDetector {
    pub fn new() -> Result<Self, SmartTurnError> {
        let smart_turn = SmartTurnModel::load().ok();

        if smart_turn.is_some() {
            log::info!("Smart Turn model loaded successfully");
        } else {
            log::warn!("Smart Turn not available, VAD-only mode");
        }

        Ok(Self {
            smart_turn,
            turn_buffer: Vec::with_capacity(MAX_TURN_SAMPLES),
            lookback_buffer: Vec::with_capacity(LOOKBACK_SAMPLES),
            is_speaking: false,
            speech_start_time: None,
            silence_timer: None,
            last_evaluated_size: 0,
        })
    }

    pub fn is_available(&self) -> bool {
        self.smart_turn.is_some()
    }

    pub fn is_speaking(&self) -> bool {
        self.is_speaking
    }

    pub fn audio_len(&self) -> usize {
        self.turn_buffer.len()
    }

    pub fn get_current_audio(&self) -> &[f32] {
        &self.turn_buffer
    }

    /// Feed audio continuously — this maintains the lookback buffer
    /// Call this on EVERY audio frame, regardless of speech state
    pub fn feed_audio(&mut self, samples: &[f32]) {
        if self.is_speaking {
            // During speech: accumulate in turn buffer
            self.turn_buffer.extend_from_slice(samples);
            // Cap turn buffer at MAX_TURN_SAMPLES
            if self.turn_buffer.len() > MAX_TURN_SAMPLES {
                let excess = self.turn_buffer.len() - MAX_TURN_SAMPLES;
                self.turn_buffer.drain(0..excess);
            }
        } else {
            // Not speaking: maintain rolling lookback buffer
            self.lookback_buffer.extend_from_slice(samples);
            if self.lookback_buffer.len() > LOOKBACK_SAMPLES {
                let excess = self.lookback_buffer.len() - LOOKBACK_SAMPLES;
                self.lookback_buffer.drain(0..excess);
            }
        }
    }

    /// VAD detected speech start — seed turn buffer with lookback
    pub fn on_speech_start(&mut self) {
        if !self.is_speaking {
            self.is_speaking = true;
            self.speech_start_time = Some(Instant::now());
            self.silence_timer = None;
            self.last_evaluated_size = 0; // Reset for new turn

            // Seed turn buffer with lookback audio (captures speech onset)
            self.turn_buffer.clear();
            self.turn_buffer.extend_from_slice(&self.lookback_buffer);
            self.lookback_buffer.clear();

            log::debug!(
                "[TurnDetector] Speech start - seeded with {}ms lookback ({} samples)",
                self.turn_buffer.len() * 1000 / SAMPLE_RATE as usize,
                self.turn_buffer.len()
            );
        }
    }

    /// Legacy API — use feed_audio() instead for continuous feeding
    pub fn add_audio(&mut self, samples: &[f32]) {
        self.feed_audio(samples);
    }

    pub fn on_silence(&mut self) -> TurnDecision {
        if !self.is_speaking {
            return TurnDecision::Continue;
        }

        if self.turn_buffer.is_empty() {
            log::debug!("[TurnDetector] Silence but no audio accumulated");
            return TurnDecision::Continue;
        }

        let duration = self
            .speech_start_time
            .map(|t| t.elapsed().as_secs_f32())
            .unwrap_or(0.0);

        let elapsed_ms = match self.silence_timer {
            Some(timer) => timer.elapsed().as_millis() as u64,
            None => {
                self.silence_timer = Some(Instant::now());
                log::debug!(
                    "[TurnDetector] Silence detected, starting {}ms timer",
                    SILENCE_TIMEOUT_MS
                );
                return TurnDecision::Continue;
            }
        };

        if elapsed_ms < SILENCE_TIMEOUT_MS {
            // Still counting down — don't log every frame
            return TurnDecision::Continue;
        }

        log::debug!(
            "[TurnDetector] Silence timeout expired ({}ms), checking Smart Turn",
            elapsed_ms
        );

        let current_size = self.turn_buffer.len();

        // Skip SmartTurn if buffer hasn't grown significantly since last evaluation
        // Requires >0.5s of new audio OR >10% growth to re-evaluate
        const MIN_NEW_AUDIO_SAMPLES: usize = (SAMPLE_RATE as usize) / 2; // 0.5 seconds
        let growth = current_size.saturating_sub(self.last_evaluated_size);
        let growth_percent = if self.last_evaluated_size > 0 {
            (growth as f32 / self.last_evaluated_size as f32) * 100.0
        } else {
            100.0 // First evaluation
        };

        if self.last_evaluated_size > 0 && growth < MIN_NEW_AUDIO_SAMPLES && growth_percent < 10.0 {
            log::info!(
                "[TurnDetector] No new audio since last evaluation ({} samples / {:.1}% growth) — completing turn",
                growth,
                growth_percent
            );
            // User stopped speaking and no new audio arrived since last SmartTurn eval.
            // Returning Continue here would loop forever since nothing will restart the
            // silence timer. Treat as end-of-turn.
            let audio = std::mem::take(&mut self.turn_buffer);
            let duration = self
                .speech_start_time
                .map(|t| t.elapsed().as_secs_f32())
                .unwrap_or(0.0);
            log::info!(
                "[TurnDetector] Turn complete via skip-logic (duration: {:.1}s, samples: {})",
                duration,
                audio.len()
            );
            self.is_speaking = false;
            self.silence_timer = None;
            self.last_evaluated_size = 0;
            return TurnDecision::Complete(audio);
        }

        let is_complete = match self.smart_turn.as_mut() {
            Some(st) => {
                let predict_start = Instant::now();
                match st.predict(&self.turn_buffer) {
                    Ok(prob) => {
                        let predict_ms = predict_start.elapsed().as_millis();
                        if predict_ms > 3000 {
                            log::warn!(
                                "[TurnDetector] SmartTurn took {}ms (>3s) — treating as complete to avoid blocking",
                                predict_ms
                            );
                            true // Force complete on slow inference
                        } else {
                            log::info!(
                                "[TurnDetector] Smart Turn: {:.1}% confident (threshold: {}%) in {}ms",
                                prob * 100.0,
                                SMART_TURN_THRESHOLD * 100.0,
                                predict_ms
                            );
                            self.last_evaluated_size = current_size;
                            prob >= SMART_TURN_THRESHOLD
                        }
                    }
                    Err(e) => {
                        log::error!("[TurnDetector] Smart Turn error: {}", e);
                        let fallback_complete = elapsed_ms > 1000;
                        log::info!(
                            "[TurnDetector] Fallback to silence-based: {}ms > 1000ms = {}",
                            elapsed_ms,
                            fallback_complete
                        );
                        fallback_complete
                    }
                }
            }
            None => true,
        };

        if is_complete {
            let audio = std::mem::take(&mut self.turn_buffer);
            log::info!(
                "[TurnDetector] Turn complete (duration: {:.1}s, samples: {})",
                duration,
                audio.len()
            );
            self.is_speaking = false;
            self.silence_timer = None;
            self.last_evaluated_size = 0;
            TurnDecision::Complete(audio)
        } else {
            log::debug!("[TurnDetector] Smart Turn says user still speaking");
            self.silence_timer = None;
            TurnDecision::Incomplete
        }
    }

    pub fn clear(&mut self) {
        self.turn_buffer.clear();
        self.is_speaking = false;
        self.silence_timer = None;
        self.last_evaluated_size = 0;
        // Don't clear lookback — it should always be maintained
    }
}

pub enum TurnDecision {
    Continue,
    Incomplete,
    Complete(Vec<f32>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_magnitude_basic() {
        // Test with a simple sine wave
        let n = 512;
        let (tw_re, tw_im) = precompute_twiddle_factors(n);
        let mut input = vec![0.0f32; n];
        for i in 0..n {
            input[i] = (2.0 * std::f32::consts::PI * 10.0 * i as f32 / n as f32).sin();
        }
        let mag = fft_magnitude(&input, &tw_re, &tw_im);
        assert_eq!(mag.len(), n / 2 + 1);
        // Bin 10 should have the highest magnitude
        let max_bin = mag
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_bin, 10);
    }

    #[test]
    fn test_lookback_buffer() {
        // Simulate the lookback behavior
        let mut lookback: Vec<f32> = Vec::with_capacity(LOOKBACK_SAMPLES);
        let chunk = vec![0.5f32; 160]; // 10ms chunk

        // Fill lookback
        for _ in 0..50 {
            // 500ms of audio
            lookback.extend_from_slice(&chunk);
            if lookback.len() > LOOKBACK_SAMPLES {
                let excess = lookback.len() - LOOKBACK_SAMPLES;
                lookback.drain(0..excess);
            }
        }

        // Should be capped at LOOKBACK_SAMPLES
        assert_eq!(lookback.len(), LOOKBACK_SAMPLES);
    }

    #[test]
    fn test_prepare_mel_spectrogram_short() {
        let short_audio = vec![0.5f32; 8000];
        let prepared = prepare_mel_spectrogram(&short_audio).unwrap();
        assert_eq!(prepared.len(), N_MELS * N_FRAMES);
    }
}
