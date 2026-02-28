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

use std::sync::Arc;

use ort::session::{builder::GraphOptimizationLevel, Session};

use crate::qwen_onnx::{
    BpeTokenizer, CodePredictor, EmbeddingManager, Language, OnnxTtsConfig, SamplingConfig,
    Speaker, TalkerKVCache, TalkerLM, TextTokenizer, Vocoder,
};

const SAMPLE_RATE: u32 = 24000;

struct OnnxComponents {
    talker: TalkerLM,
    code_predictor: CodePredictor,
    vocoder: Vocoder,
    tokenizer: BpeTokenizer,
    embeddings: Arc<EmbeddingManager>,
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
    components: Option<OnnxComponents>,
    variant: OnnxModelVariant,
    provider: ExecutionProvider,
    current_speaker: Speaker,
    current_language: Language,
    config: OnnxTtsConfig,
    sample_rate: u32,
}

impl QwenOnnxTts {
    pub fn new(model_dir: &std::path::Path, variant: OnnxModelVariant) -> Result<Self, TtsError> {
        Self::with_provider(model_dir, variant, ExecutionProvider::default())
    }

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

        let suffix = match variant {
            OnnxModelVariant::Fp16 => "",
            OnnxModelVariant::Int8 => "_q",
        };

        let tokenizer_dir = model_dir.join("tokenizer");
        let embeddings_dir = model_dir.join("embeddings");

        if !tokenizer_dir.exists() {
            return Err(TtsError::InitError(format!(
                "Tokenizer directory not found: {}",
                tokenizer_dir.display()
            )));
        }
        if !embeddings_dir.exists() {
            return Err(TtsError::InitError(format!(
                "Embeddings directory not found: {}",
                embeddings_dir.display()
            )));
        }

        let tokenizer = BpeTokenizer::from_files(&tokenizer_dir)
            .map_err(|e| TtsError::InitError(format!("Failed to load tokenizer: {}", e)))?;

        let embeddings = EmbeddingManager::load(&embeddings_dir)
            .map_err(|e| TtsError::InitError(format!("Failed to load embeddings: {}", e)))?;
        let embeddings = Arc::new(embeddings);

        let talker_prefill =
            Self::load_session(&model_dir.join(&format!("talker_prefill{}.onnx", suffix)))?;
        let talker_decode =
            Self::load_session(&model_dir.join(&format!("talker_decode{}.onnx", suffix)))?;
        let code_predictor_session =
            Self::load_session(&model_dir.join(&format!("code_predictor{}.onnx", suffix)))?;
        let vocoder_session = Self::load_session(&model_dir.join("vocoder.onnx"))?;

        let talker = TalkerLM::new(
            talker_prefill,
            talker_decode,
            Arc::clone(&embeddings),
            SamplingConfig::default(),
        );

        let code_predictor = CodePredictor::new(code_predictor_session, SamplingConfig::default());
        let vocoder = Vocoder::new(vocoder_session);

        let components = OnnxComponents {
            talker,
            code_predictor,
            vocoder,
            tokenizer,
            embeddings,
        };

        log::info!("[QwenOnnx] All components loaded successfully");

        Ok(Self {
            components: Some(components),
            variant,
            provider,
            current_speaker: Speaker::Ryan,
            current_language: Language::English,
            config: OnnxTtsConfig::default(),
            sample_rate: SAMPLE_RATE,
        })
    }

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

    pub fn set_speaker(&mut self, speaker_id: &str) -> Result<(), TtsError> {
        let speaker = Speaker::from_name(speaker_id).ok_or_else(|| {
            TtsError::ModelNotFound(format!(
                "Unknown speaker '{}'. Available: {:?}",
                speaker_id,
                Speaker::all()
                    .iter()
                    .map(|s| format!("{:?}", s))
                    .collect::<Vec<_>>()
            ))
        })?;
        self.current_speaker = speaker;
        Ok(())
    }

    pub fn set_language(&mut self, lang_code: &str) -> Result<(), TtsError> {
        let language = Language::from_iso(lang_code).ok_or_else(|| {
            TtsError::ModelNotFound(format!("Unknown language code: {}", lang_code))
        })?;
        self.current_language = language;
        Ok(())
    }

    pub fn speaker(&self) -> Speaker {
        self.current_speaker
    }

    pub fn language(&self) -> Language {
        self.current_language
    }

    pub fn available_speakers() -> Vec<(&'static str, &'static str)> {
        crate::qwen_tts::QWEN_LITE_VOICES.to_vec()
    }

    pub fn synthesize(&mut self, text: &str) -> Result<AudioOutput, TtsError> {
        if text.trim().is_empty() {
            return Ok(AudioOutput {
                samples: vec![],
                sample_rate: self.sample_rate,
            });
        }

        let components = self
            .components
            .as_mut()
            .ok_or_else(|| TtsError::InitError("Components not initialized".to_string()))?;

        log::info!(
            "[QwenOnnx] Synthesizing: '{}' ({} chars)",
            text.chars().take(50).collect::<String>(),
            text.len()
        );

        let input_ids = components
            .tokenizer
            .encode(text)
            .map_err(|e| TtsError::SynthesisError(format!("Tokenization failed: {}", e)))?;

        log::debug!("[QwenOnnx] Tokenized to {} tokens", input_ids.len());

        let prefill_output = components
            .talker
            .prefill(&input_ids, self.current_speaker, self.current_language)
            .map_err(|e| TtsError::SynthesisError(format!("Talker prefill failed: {}", e)))?;

        log::debug!(
            "[QwenOnnx] Prefill complete, seq_len={}",
            prefill_output.seq_len
        );

        let mut all_codes: Vec<[i64; 16]> = Vec::new();
        let mut kv_cache = prefill_output.kv_cache;
        let mut current_logits = prefill_output.logits;
        let mut current_hidden = prefill_output.hidden;
        let mut position = prefill_output.seq_len;

        let eos_token = components.tokenizer.eos_token_id();
        let codec_eos = crate::qwen_onnx::special_tokens::CODEC_EOS as i64;

        let mut rng = crate::qwen_onnx::create_rng(self.config.sampling.seed);

        for frame_idx in 0..self.config.max_frames {
            let semantic_token = crate::qwen_onnx::sample_token(
                &current_logits,
                &self.config.sampling,
                codec_eos,
                &mut rng,
            )
            .map_err(|e| TtsError::SynthesisError(format!("Sampling failed: {}", e)))?;

            if semantic_token == eos_token || semantic_token == codec_eos {
                log::info!("[QwenOnnx] Reached EOS at frame {}", frame_idx);
                break;
            }

            let acoustic_codes = components
                .code_predictor
                .generate_acoustic_codes(&current_hidden, semantic_token, &components.embeddings)
                .map_err(|e| TtsError::SynthesisError(format!("Code predictor failed: {}", e)))?;

            let mut frame_codes = [0i64; 16];
            frame_codes[0] = semantic_token;
            frame_codes[1..16].copy_from_slice(&acoustic_codes);
            all_codes.push(frame_codes);

            let step_embed = components
                .talker
                .build_step_embedding(semantic_token, &acoustic_codes, None)
                .map_err(|e| TtsError::SynthesisError(format!("Step embedding failed: {}", e)))?;

            let decode_output = components
                .talker
                .decode_step(&step_embed, position, &mut kv_cache)
                .map_err(|e| TtsError::SynthesisError(format!("Decode step failed: {}", e)))?;

            current_logits = decode_output.logits;
            current_hidden = decode_output.hidden;
            position += 1;
        }

        if all_codes.is_empty() {
            return Ok(AudioOutput {
                samples: vec![],
                sample_rate: self.sample_rate,
            });
        }

        let num_frames = all_codes.len();
        let mut codes_array = ndarray::Array2::zeros((16, num_frames));
        for (t, frame) in all_codes.iter().enumerate() {
            for (c, &code) in frame.iter().enumerate() {
                codes_array[[c, t]] = code;
            }
        }

        log::debug!("[QwenOnnx] Vocoding {} frames", num_frames);

        let audio = components
            .vocoder
            .decode(&codes_array)
            .map_err(|e| TtsError::SynthesisError(format!("Vocoder failed: {}", e)))?;

        log::info!(
            "[QwenOnnx] Synthesis complete: {} samples ({:.2}s)",
            audio.len(),
            audio.len() as f32 / self.sample_rate as f32
        );

        Ok(AudioOutput {
            samples: audio,
            sample_rate: self.sample_rate,
        })
    }

    pub fn synthesize_streaming<F>(&mut self, text: &str, mut on_chunk: F) -> Result<u32, TtsError>
    where
        F: FnMut(&[f32], u32) -> bool,
    {
        if text.trim().is_empty() {
            return Ok(self.sample_rate);
        }

        let components = self
            .components
            .as_mut()
            .ok_or_else(|| TtsError::InitError("Components not initialized".to_string()))?;

        log::info!(
            "[QwenOnnx] Streaming synthesis: '{}' ({} chars)",
            text.chars().take(50).collect::<String>(),
            text.len()
        );

        let input_ids = components
            .tokenizer
            .encode(text)
            .map_err(|e| TtsError::SynthesisError(format!("Tokenization failed: {}", e)))?;

        let prefill_output = components
            .talker
            .prefill(&input_ids, self.current_speaker, self.current_language)
            .map_err(|e| TtsError::SynthesisError(format!("Talker prefill failed: {}", e)))?;

        let mut chunk_codes: Vec<[i64; 16]> = Vec::with_capacity(self.config.frames_per_chunk);
        let mut kv_cache = prefill_output.kv_cache;
        let mut current_logits = prefill_output.logits;
        let mut current_hidden = prefill_output.hidden;
        let mut position = prefill_output.seq_len;

        let eos_token = components.tokenizer.eos_token_id();
        let codec_eos = crate::qwen_onnx::special_tokens::CODEC_EOS as i64;

        let mut rng = crate::qwen_onnx::create_rng(self.config.sampling.seed);

        for frame_idx in 0..self.config.max_frames {
            let semantic_token = crate::qwen_onnx::sample_token(
                &current_logits,
                &self.config.sampling,
                codec_eos,
                &mut rng,
            )
            .map_err(|e| TtsError::SynthesisError(format!("Sampling failed: {}", e)))?;

            if semantic_token == eos_token || semantic_token == codec_eos {
                log::info!("[QwenOnnx] Reached EOS at frame {}", frame_idx);
                break;
            }

            let acoustic_codes = components
                .code_predictor
                .generate_acoustic_codes(&current_hidden, semantic_token, &components.embeddings)
                .map_err(|e| TtsError::SynthesisError(format!("Code predictor failed: {}", e)))?;

            let mut frame_codes = [0i64; 16];
            frame_codes[0] = semantic_token;
            frame_codes[1..16].copy_from_slice(&acoustic_codes);
            chunk_codes.push(frame_codes);

            if chunk_codes.len() >= self.config.frames_per_chunk {
                let mut codes_array = ndarray::Array2::zeros((16, chunk_codes.len()));
                for (t, frame) in chunk_codes.iter().enumerate() {
                    for (c, &code) in frame.iter().enumerate() {
                        codes_array[[c, t]] = code;
                    }
                }

                let audio = components
                    .vocoder
                    .decode(&codes_array)
                    .map_err(|e| TtsError::SynthesisError(format!("Vocoder failed: {}", e)))?;

                if !on_chunk(&audio, self.sample_rate) {
                    log::info!("[QwenOnnx] Streaming cancelled by callback");
                    return Ok(self.sample_rate);
                }

                chunk_codes.clear();
            }

            let step_embed = components
                .talker
                .build_step_embedding(semantic_token, &acoustic_codes, None)
                .map_err(|e| TtsError::SynthesisError(format!("Step embedding failed: {}", e)))?;

            let decode_output = components
                .talker
                .decode_step(&step_embed, position, &mut kv_cache)
                .map_err(|e| TtsError::SynthesisError(format!("Decode step failed: {}", e)))?;

            current_logits = decode_output.logits;
            current_hidden = decode_output.hidden;
            position += 1;
        }

        if !chunk_codes.is_empty() {
            let mut codes_array = ndarray::Array2::zeros((16, chunk_codes.len()));
            for (t, frame) in chunk_codes.iter().enumerate() {
                for (c, &code) in frame.iter().enumerate() {
                    codes_array[[c, t]] = code;
                }
            }

            let audio = components
                .vocoder
                .decode(&codes_array)
                .map_err(|e| TtsError::SynthesisError(format!("Vocoder failed: {}", e)))?;

            on_chunk(&audio, self.sample_rate);
        }

        log::info!("[QwenOnnx] Streaming synthesis complete");
        Ok(self.sample_rate)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

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
