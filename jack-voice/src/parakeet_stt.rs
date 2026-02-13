// Jack Desktop - Parakeet STT Backend
// Uses parakeet-rs for multilingual speech-to-text:
// - ParakeetEOU: streaming transcription (160ms chunks, cache-aware)
// - ParakeetTDT: offline final transcription (25 European languages, auto-detect)

use crate::stt::{SttError, TranscriptionResult};
use parakeet_rs::{ExecutionConfig, ParakeetEOU, ParakeetTDT, Transcriber};

/// Post-STT validation: Check if transcription meets quality thresholds
/// Prevents partial/low-quality transcriptions from being sent to LLM
fn should_send_transcription(text: &str) -> bool {
    let trimmed = text.trim();

    // Reject empty/whitespace
    if trimmed.is_empty() {
        return false;
    }

    // Reject if < 5 characters (too short to be meaningful)
    if trimmed.len() < 5 {
        return false;
    }

    true
}

/// Streaming STT using ParakeetEOU (160ms chunks, real-time)
pub struct ParakeetStreamingStt {
    recognizer: ParakeetEOU,
}

impl ParakeetStreamingStt {
    pub fn new(model_dir: &str) -> Result<Self, SttError> {
        let config = create_execution_config();
        let recognizer = ParakeetEOU::from_pretrained(model_dir, Some(config))
            .map_err(|e| SttError::InitError(format!("ParakeetEOU init failed: {}", e)))?;
        log::info!(
            "[STT] ParakeetEOU streaming model loaded from {}",
            model_dir
        );
        Ok(Self { recognizer })
    }

    /// Feed a chunk of audio (ideally 160ms = 2560 samples at 16kHz)
    /// Returns partial transcription if available
    pub fn feed_chunk(
        &mut self,
        samples: &[f32],
        _sample_rate: u32,
    ) -> Result<Option<TranscriptionResult>, SttError> {
        let text = self.recognizer.transcribe(samples, false).map_err(|e| {
            SttError::ProcessingError(format!("ParakeetEOU transcribe failed: {}", e))
        })?;

        let text = text.trim().to_string();
        if text.is_empty() {
            return Ok(None);
        }

        Ok(Some(TranscriptionResult {
            text,
            is_final: false,
            is_partial: true,
            latency_ms: 0,
        }))
    }

    /// Reset model state at turn boundaries
    pub fn reset(&mut self) -> Result<(), SttError> {
        // Clear internal state by feeding empty buffer with reset=true
        let _ = self
            .recognizer
            .transcribe(&[], true)
            .map_err(|e| SttError::ProcessingError(format!("ParakeetEOU reset failed: {}", e)))?;
        Ok(())
    }

    pub fn is_ready(&self) -> bool {
        true
    }
}

/// Offline STT using ParakeetTDT (25 European languages, high accuracy)
pub struct ParakeetOfflineStt {
    recognizer: ParakeetTDT,
}

impl ParakeetOfflineStt {
    pub fn new(model_dir: &str) -> Result<Self, SttError> {
        let config = create_execution_config();
        let recognizer = ParakeetTDT::from_pretrained(model_dir, Some(config))
            .map_err(|e| SttError::InitError(format!("ParakeetTDT init failed: {}", e)))?;
        log::info!("[STT] ParakeetTDT offline model loaded from {}", model_dir);
        Ok(Self { recognizer })
    }

    /// Transcribe a complete audio segment (called after turn-complete)
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<TranscriptionResult, SttError> {
        if samples.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                is_final: true,
                is_partial: false,
                latency_ms: 0,
            });
        }

        // Energy gating: Reject low-energy audio before STT processing
        // This prevents hallucinations on silence/noise
        const STT_MIN_RMS: f32 = 0.01;
        const STT_MIN_AMP: f32 = 0.03;

        // Calculate RMS energy
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

        // Calculate max amplitude
        let max_amp = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);

        // Reject if energy is too low
        if rms < STT_MIN_RMS || max_amp < STT_MIN_AMP {
            log::info!(
                "[STT] ParakeetTDT: Rejecting low-energy audio (rms={:.4}, max={:.4})",
                rms,
                max_amp
            );
            return Ok(TranscriptionResult {
                text: String::new(),
                is_final: true,
                is_partial: false,
                latency_ms: 0,
            });
        }

        let start = std::time::Instant::now();
        let result = self
            .recognizer
            .transcribe_samples(
                samples.to_vec(),
                sample_rate,
                1,    // mono channels
                None, // no timestamp mode
            )
            .map_err(|e| {
                SttError::ProcessingError(format!("ParakeetTDT transcribe failed: {}", e))
            })?;

        let latency_ms = start.elapsed().as_millis() as u64;
        let text = result.text.trim().to_string();

        // Post-STT validation: Check transcription quality
        if !should_send_transcription(&text) {
            log::info!(
                "[STT] ParakeetTDT: Rejecting low-quality transcription: '{}' (len={}, words={})",
                text,
                text.len(),
                text.split_whitespace().count()
            );
            return Ok(TranscriptionResult {
                text: String::new(),
                is_final: true,
                is_partial: false,
                latency_ms,
            });
        }

        log::info!(
            "[STT] ParakeetTDT: '{}' in {}ms ({} samples)",
            text,
            latency_ms,
            samples.len()
        );

        Ok(TranscriptionResult {
            text,
            is_final: true,
            is_partial: false,
            latency_ms,
        })
    }

    pub fn is_ready(&self) -> bool {
        true
    }
}

/// Create execution config with best available provider (GPU fallback to CPU)
fn create_execution_config() -> ExecutionConfig {
    let config = ExecutionConfig::default();

    #[cfg(feature = "cuda")]
    {
        log::info!("[STT] Attempting CUDA provider for Parakeet");
        return config.with_execution_provider(parakeet_rs::ExecutionProvider::Cuda);
    }

    #[cfg(feature = "directml")]
    {
        log::info!("[STT] Attempting DirectML provider for Parakeet");
        return config.with_execution_provider(parakeet_rs::ExecutionProvider::DirectML);
    }

    #[cfg(not(any(feature = "cuda", feature = "directml")))]
    {
        log::info!("[STT] Using CPU provider for Parakeet");
        config
    }
}
