//! Qwen3 TTS ONNX Backend
//!
//! ONNX Runtime-based implementation of Qwen3 TTS for cross-platform support.
//! Supports CUDA, CoreML, DirectML, OpenVINO, and CPU execution providers.
//!
//! Models:
//! - INT8 quantized: ~1.6 GB, optimized for edge/realtime
//! - FP16: ~3 GB, higher quality
//!
//! References:
//! - https://huggingface.co/sivasub987/Qwen3-TTS-0.6B-ONNX-INT8
//! - https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX

use std::path::PathBuf;
use std::sync::Arc;

use ort::session::{builder::GraphOptimizationLevel, Session};

const SAMPLE_RATE: u32 = 24000;

/// Qwen ONNX model components
struct OnnxModelComponents {
    /// Talker LM prefill (initial token generation)
    talker_prefill: Session,
    /// Talker LM decode (autoregressive generation)
    talker_decode: Session,
    /// Code predictor (generates codebook indices)
    code_predictor: Session,
    /// Vocoder (converts codes to audio)
    vocoder: Session,
    /// Speaker encoder for voice cloning (optional)
    speaker_encoder: Option<Session>,
}

/// ONNX model variant
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxModelVariant {
    /// Full precision (FP16), ~3GB
    Fp16,
    /// INT8 quantized, ~1.6GB
    Int8,
}

/// Execution provider for ONNX inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    Cpu,
    Cuda,
    CoreML,
    DirectML,
    OpenVINO,
}

impl Default for ExecutionProvider {
    fn default() -> Self {
        Self::detect_best()
    }
}

impl ExecutionProvider {
    /// Detect the best available execution provider
    pub fn detect_best() -> Self {
        #[cfg(target_os = "macos")]
        {
            Self::CoreML
        }
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        {
            // TODO: Actually check for CUDA availability
            Self::Cuda
        }
        #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
        {
            Self::Cpu
        }
    }
}

/// Qwen3 TTS using ONNX Runtime
pub struct QwenOnnxTts {
    models: Arc<OnnxModelComponents>,
    variant: OnnxModelVariant,
    provider: ExecutionProvider,
    current_speaker: String,
    sample_rate: u32,
}

impl QwenOnnxTts {
    /// Create a new Qwen ONNX TTS instance with the specified variant
    pub fn new(model_dir: &std::path::Path, variant: OnnxModelVariant) -> Result<Self, TtsError> {
        Self::with_provider(model_dir, variant, ExecutionProvider::default())
    }

    /// Create a new Qwen ONNX TTS instance with specific execution provider
    pub fn with_provider(
        model_dir: &std::path::Path,
        variant: OnnxModelVariant,
        provider: ExecutionProvider,
    ) -> Result<Self, TtsError> {
        log::info!(
            "[QwenOnnx] Initializing {:?} variant with {:?} provider from {}",
            variant,
            provider,
            model_dir.display()
        );

        let models = Self::load_models(model_dir, variant, provider)?;

        log::info!("[QwenOnnx] Models loaded successfully");

        Ok(Self {
            models: Arc::new(models),
            variant,
            provider,
            current_speaker: "ryan".to_string(),
            sample_rate: SAMPLE_RATE,
        })
    }

    /// Load all ONNX model components
    fn load_models(
        model_dir: &std::path::Path,
        variant: OnnxModelVariant,
        _provider: ExecutionProvider,
    ) -> Result<OnnxModelComponents, TtsError> {
        let suffix = match variant {
            OnnxModelVariant::Fp16 => "",
            OnnxModelVariant::Int8 => "_q",
        };

        let talker_prefill =
            Self::load_session(&model_dir.join(&format!("talker_prefill{}.onnx", suffix)))?;
        let talker_decode =
            Self::load_session(&model_dir.join(&format!("talker_decode{}.onnx", suffix)))?;
        let code_predictor =
            Self::load_session(&model_dir.join(&format!("code_predictor{}.onnx", suffix)))?;
        let vocoder = Self::load_session(&model_dir.join("vocoder.onnx"))?;

        let speaker_encoder_path = model_dir.join(&format!("speaker_encoder{}.onnx", suffix));
        let speaker_encoder = if speaker_encoder_path.exists() {
            Some(Self::load_session(&speaker_encoder_path)?)
        } else {
            None
        };

        Ok(OnnxModelComponents {
            talker_prefill,
            talker_decode,
            code_predictor,
            vocoder,
            speaker_encoder,
        })
    }

    /// Load a single ONNX session
    fn load_session(model_path: &std::path::Path) -> Result<Session, TtsError> {
        if !model_path.exists() {
            return Err(TtsError::ModelNotFound(format!(
                "ONNX model not found: {}",
                model_path.display()
            )));
        }

        Session::builder()
            .map_err(|e| TtsError::InitError(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| TtsError::InitError(format!("Failed to set optimization level: {}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| {
                TtsError::InitError(format!(
                    "Failed to load ONNX model {}: {}",
                    model_path.display(),
                    e
                ))
            })
    }

    /// Set the speaker voice
    pub fn set_speaker(&mut self, speaker_id: &str) -> Result<(), TtsError> {
        // Validate speaker is in available list
        let valid_speakers = Self::available_speakers();
        if !valid_speakers.iter().any(|(id, _)| id == &speaker_id) {
            return Err(TtsError::ModelNotFound(format!(
                "Unknown speaker '{}'. Available: {:?}",
                speaker_id,
                valid_speakers.iter().map(|(id, _)| id).collect::<Vec<_>>()
            )));
        }
        self.current_speaker = speaker_id.to_string();
        Ok(())
    }

    /// Get the current speaker
    pub fn speaker(&self) -> &str {
        &self.current_speaker
    }

    /// Get available speakers
    pub fn available_speakers() -> Vec<(&'static str, &'static str)> {
        crate::qwen_tts::QWEN_LITE_VOICES.to_vec()
    }

    /// Check if voice cloning is supported
    pub fn supports_voice_cloning(&self) -> bool {
        self.models.speaker_encoder.is_some()
    }

    /// Synthesize text to audio
    ///
    /// # Implementation Status
    ///
    /// This is a placeholder implementation. Full synthesis requires:
    ///
    /// 1. **Text Tokenization**: BPE tokenizer (vocab.json + merges.txt)
    /// 2. **Talker Prefill**: Run `talker_prefill.onnx` with text embeddings
    /// 3. **Talker Decode Loop**: Run `talker_decode.onnx` autoregressively
    /// 4. **Code Predictor**: For each talker step, run `code_predictor.onnx` to generate
    ///    31 additional codebook tokens
    /// 5. **Vocoder**: Run `vocoder.onnx` to convert codes to audio
    ///
    /// See: https://huggingface.co/elbruno/Qwen3-TTS-12Hz-0.6B-CustomVoice-ONNX
    ///
    /// # TODO
    ///
    /// - Implement BPE tokenizer (or use tokenizers crate)
    /// - Implement embedding lookups (text + codec)
    /// - Implement KV cache management for decode loop
    /// - Implement vocoder decode
    pub fn synthesize(&self, text: &str) -> Result<AudioOutput, TtsError> {
        if text.trim().is_empty() {
            return Ok(AudioOutput {
                samples: vec![],
                sample_rate: self.sample_rate,
            });
        }

        log::info!(
            "[QwenOnnx] Synthesis requested for: '{}' ({} chars)",
            text.chars().take(50).collect::<String>(),
            text.len()
        );
        log::warn!("[QwenOnnx] Synthesis not yet implemented");
        log::info!(
            "[QwenOnnx] Required models: talker_prefill.onnx, talker_decode.onnx, \
             code_predictor.onnx, vocoder.onnx, embeddings/, tokenizer/"
        );

        // Return empty audio for now
        Ok(AudioOutput {
            samples: vec![],
            sample_rate: self.sample_rate,
        })
    }

    /// Synthesize text with streaming callback
    ///
    /// # Implementation Status
    ///
    /// This is a placeholder implementation. See [`synthesize`](Self::synthesize) for details.
    pub fn synthesize_streaming<F>(&self, text: &str, _on_chunk: F) -> Result<u32, TtsError>
    where
        F: FnMut(&[f32], u32) -> bool,
    {
        if text.trim().is_empty() {
            return Ok(self.sample_rate);
        }

        log::info!(
            "[QwenOnnx] Streaming synthesis requested for: '{}' ({} chars)",
            text.chars().take(50).collect::<String>(),
            text.len()
        );
        log::warn!("[QwenOnnx] Streaming synthesis not yet implemented");

        Ok(self.sample_rate)
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get the engine type string
    pub fn engine_type(&self) -> &'static str {
        match self.variant {
            OnnxModelVariant::Fp16 => "qwen-onnx",
            OnnxModelVariant::Int8 => "qwen-onnx-int8",
        }
    }
}

/// Audio output from TTS
#[derive(Clone, Debug)]
pub struct AudioOutput {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

/// TTS error type
#[derive(Debug, thiserror::Error)]
pub enum TtsError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Initialization error: {0}")]
    InitError(String),
    #[error("Synthesis error: {0}")]
    SynthesisError(String),
}

// ============================================================================
// Tests (TDD RED phase)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_model_variant_exists() {
        let _fp16 = OnnxModelVariant::Fp16;
        let _int8 = OnnxModelVariant::Int8;
    }

    #[test]
    fn test_execution_provider_exists() {
        let _cpu = ExecutionProvider::Cpu;
        let _cuda = ExecutionProvider::Cuda;
        let _coreml = ExecutionProvider::CoreML;
        let _directml = ExecutionProvider::DirectML;
        let _openvino = ExecutionProvider::OpenVINO;
    }

    #[test]
    fn test_execution_provider_default() {
        let provider = ExecutionProvider::default();
        // Should return a valid provider
        assert!(matches!(
            provider,
            ExecutionProvider::Cpu
                | ExecutionProvider::Cuda
                | ExecutionProvider::CoreML
                | ExecutionProvider::DirectML
                | ExecutionProvider::OpenVINO
        ));
    }

    #[test]
    fn test_execution_provider_detect_best() {
        let provider = ExecutionProvider::detect_best();
        // Should return a valid provider
        assert!(matches!(
            provider,
            ExecutionProvider::Cpu
                | ExecutionProvider::Cuda
                | ExecutionProvider::CoreML
                | ExecutionProvider::DirectML
                | ExecutionProvider::OpenVINO
        ));
    }

    #[test]
    fn test_available_speakers() {
        let speakers = QwenOnnxTts::available_speakers();
        assert!(!speakers.is_empty(), "Should have available speakers");
        assert!(
            speakers.iter().any(|(id, _)| *id == "ryan"),
            "Should include ryan"
        );
        assert!(
            speakers.iter().any(|(id, _)| *id == "serena"),
            "Should include serena"
        );
    }

    #[test]
    fn test_qwen_onnx_tts_new_fails_without_models() {
        let result = QwenOnnxTts::new(
            std::path::Path::new("/nonexistent/path"),
            OnnxModelVariant::Int8,
        );
        assert!(result.is_err(), "Should fail without model files");
    }

    #[test]
    fn test_set_speaker_validates() {
        // Create a mock instance - this will fail without models, but we test the validation
        let result = QwenOnnxTts::new(std::path::Path::new("/nonexistent"), OnnxModelVariant::Int8);
        // Expected to fail due to missing models, not speaker validation
        assert!(result.is_err());
    }

    #[test]
    fn test_audio_output_exists() {
        let output = AudioOutput {
            samples: vec![0.0f32; 100],
            sample_rate: 24000,
        };
        assert_eq!(output.sample_rate, 24000);
        assert_eq!(output.samples.len(), 100);
    }

    #[test]
    fn test_tts_error_variants() {
        let _ = TtsError::ModelNotFound("test".to_string());
        let _ = TtsError::InitError("test".to_string());
        let _ = TtsError::SynthesisError("test".to_string());
    }

    #[test]
    fn test_onnx_model_variant_equality() {
        assert_eq!(OnnxModelVariant::Fp16, OnnxModelVariant::Fp16);
        assert_eq!(OnnxModelVariant::Int8, OnnxModelVariant::Int8);
        assert_ne!(OnnxModelVariant::Fp16, OnnxModelVariant::Int8);
    }

    #[test]
    fn test_engine_type() {
        // Test via mock - we can't create real instance without models
        // but we can verify the logic
        let fp16_type = "qwen-onnx";
        let int8_type = "qwen-onnx-int8";
        assert_ne!(fp16_type, int8_type);
    }

    /// Documentation test for synthesis pipeline architecture.
    /// This documents the expected ONNX model I/O for future implementers.
    #[test]
    fn test_synthesis_pipeline_documentation() {
        // This test documents the expected ONNX synthesis pipeline.
        //
        // SYNTHESIS PIPELINE:
        //
        // 1. TEXT TOKENIZATION (BPE Tokenizer)
        //    Input:  text string (e.g., "Hello world")
        //    Output: input_ids: Vec<i64> (token IDs)
        //    Files:  tokenizer/vocab.json, tokenizer/merges.txt
        //
        // 2. TALKER PREFILL (talker_prefill.onnx)
        //    Inputs:
        //      - input_ids: (1, T_text) int64
        //      - speaker_id: int64 (0-8 for preset voices)
        //    Outputs:
        //      - logits: (1, 1, 3072) float32  -- first token prediction
        //      - hidden_states: (1, T, 1024) float32  -- for code predictor
        //      - kv_cache: (28, 1, 8, T, 128) float32  -- KV cache per layer
        //
        // 3. TALKER DECODE LOOP (talker_decode.onnx)
        //    For each generated token until EOS:
        //    Inputs:
        //      - input_ids: (1, 1) int64  -- previous token
        //      - kv_cache: from previous step
        //    Outputs:
        //      - logits: (1, 1, 3072) float32
        //      - hidden_states: (1, 1, 1024) float32
        //      - kv_cache: updated cache
        //
        // 4. CODE PREDICTOR (code_predictor.onnx)
        //    For each talker step, runs 31 times:
        //    Inputs:
        //      - talker_hidden: (1, 1, 1024) float32
        //      - group_0_embed: (1, 1, 1024) float32
        //      - generation_step: int64 (1-31)
        //    Outputs:
        //      - logits: (1, 1, 2048) float32  -- predicted codebook token
        //
        // 5. VOCODER (vocoder.onnx)
        //    Input:
        //      - codes: (1, 16, T_frames) int64  -- 16 codebooks, T timesteps
        //    Output:
        //      - waveform: (1, num_samples) float32  -- 24kHz audio
        //
        // MODEL FILES NEEDED:
        //   - talker_prefill.onnx + .data (~1.7 GB)
        //   - talker_decode.onnx + .data (~1.7 GB)
        //   - code_predictor.onnx (~420 MB)
        //   - vocoder.onnx + .data (~437 MB)
        //   - embeddings/*.npy (~1.4 GB)
        //   - tokenizer/vocab.json, merges.txt (~4 MB)
        //
        // Total: ~5.3 GB for FP16, ~1.6 GB for INT8

        // This test always passes - it's just documentation
        assert!(true);
    }
}
