//! ONNX session management for Qwen3 TTS.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use ort::session::{builder::GraphOptimizationLevel, Session};

use super::super::ExecutionProvider;

/// Manages all ONNX sessions for Qwen TTS.
pub struct OnnxSessions {
    /// Talker prefill session (initial text processing).
    pub talker_prefill: Arc<Session>,

    /// Talker decode session (autoregressive generation).
    pub talker_decode: Arc<Session>,

    /// Code predictor session (acoustic code generation).
    pub code_predictor: Arc<Session>,

    /// Vocoder session (code to audio).
    pub vocoder: Arc<Session>,
}

impl OnnxSessions {
    /// Load all ONNX sessions from the model directory.
    ///
    /// # Arguments
    /// * `model_dir` - Directory containing ONNX model files
    /// * `use_int8` - Whether to use INT8 quantized models
    pub fn load(model_dir: &Path, use_int8: bool, _provider: ExecutionProvider) -> Result<Self> {
        let suffix = if use_int8 { "_q" } else { "" };

        // Load sessions
        let talker_prefill =
            Self::load_session(&model_dir.join(format!("talker_prefill{}.onnx", suffix)))?;

        let talker_decode =
            Self::load_session(&model_dir.join(format!("talker_decode{}.onnx", suffix)))?;

        let code_predictor =
            Self::load_session(&model_dir.join(format!("code_predictor{}.onnx", suffix)))?;

        let vocoder = Self::load_session(&model_dir.join("vocoder.onnx"))?;

        Ok(Self {
            talker_prefill: Arc::new(talker_prefill),
            talker_decode: Arc::new(talker_decode),
            code_predictor: Arc::new(code_predictor),
            vocoder: Arc::new(vocoder),
        })
    }

    /// Load a single ONNX session.
    fn load_session(model_path: &Path) -> Result<Session> {
        if !model_path.exists() {
            anyhow::bail!("ONNX model not found: {}", model_path.display());
        }

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {}", e))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("Failed to set optimization level: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| {
                anyhow::anyhow!("Failed to load ONNX model {}: {}", model_path.display(), e)
            })?;

        log::info!("[OnnxSessions] Loaded: {}", model_path.display());
        Ok(session)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_fails_without_files() {
        let result = OnnxSessions::load(Path::new("/nonexistent"), false, ExecutionProvider::Cpu);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_int8_fails_without_files() {
        let result = OnnxSessions::load(Path::new("/nonexistent"), true, ExecutionProvider::Cpu);
        assert!(result.is_err());
    }

    // Integration test - requires downloaded models
    #[test]
    #[ignore]
    fn test_load_real_sessions() {
        let model_dir = dirs::home_dir()
            .unwrap()
            .join(".nanobot/models/qwen/qwen-onnx");

        if !model_dir.exists() {
            return;
        }

        let sessions = OnnxSessions::load(&model_dir, false, ExecutionProvider::Cpu).unwrap();

        // Verify sessions loaded
        assert!(Arc::strong_count(&sessions.talker_prefill) >= 1);
        assert!(Arc::strong_count(&sessions.talker_decode) >= 1);
        assert!(Arc::strong_count(&sessions.code_predictor) >= 1);
        assert!(Arc::strong_count(&sessions.vocoder) >= 1);
    }
}
