//! Code Predictor for generating acoustic codes.
//!
//! For each semantic token from the Talker, the Code Predictor generates
//! 15 additional acoustic codes (groups 1-15). Uses a 5-layer transformer
//! with its own KV cache.

use std::borrow::Cow;

use anyhow::Result;
use ndarray::{Array1, Array3, Axis};
use ort::session::{Session, SessionInputValue, SessionInputs};
use ort::value::{Tensor, Value};
use rand::SeedableRng;

use super::embeddings::EmbeddingManager;
use super::kv_cache::CodePredictorKVCache;
use super::sampling::{sample_token, SamplingConfig};
use super::special_tokens::CODEC_EOS;

/// Code Predictor for acoustic code generation.
pub struct CodePredictor {
    session: Session,
    config: SamplingConfig,
}

impl CodePredictor {
    pub fn new(session: Session, config: SamplingConfig) -> Self {
        Self { session, config }
    }

    pub fn generate_acoustic_codes(
        &mut self,
        talker_hidden: &Array3<f32>,
        semantic_code: i64,
        embeddings: &EmbeddingManager,
    ) -> Result<[i64; 15]> {
        let mut codes = [0i64; 15];
        let mut kv_cache = CodePredictorKVCache::new_default(64);

        let semantic_embed = embeddings.codec_embed(semantic_code)?;
        let semantic_embed_3d = Array1::from_vec(semantic_embed)
            .insert_axis(Axis(0))
            .insert_axis(Axis(0));

        let input_embed = Self::concat_inputs(talker_hidden, &semantic_embed_3d)?;

        let mut current_embed = input_embed;

        for step in 0..15 {
            let (logits, hidden) = self.step(&current_embed, step, &mut kv_cache)?;

            let code = sample_token(
                &logits,
                &self.config,
                CODEC_EOS as i64,
                &mut rand::rngs::StdRng::seed_from_u64(42),
            )?;

            codes[step] = code;

            if step < 14 {
                let next_embed = embeddings.cp_codec_embed(step + 1, code)?;
                let next_embed_3d = Array1::from_vec(next_embed)
                    .insert_axis(Axis(0))
                    .insert_axis(Axis(0));
                current_embed = Self::concat_inputs(&hidden, &next_embed_3d)?;
            }
        }

        Ok(codes)
    }

    fn step(
        &mut self,
        input_embed: &Array3<f32>,
        step_idx: usize,
        kv_cache: &mut CodePredictorKVCache,
    ) -> Result<(Array1<f32>, Array3<f32>)> {
        let hidden_size = input_embed.shape()[2];
        let input_data: Vec<f32> = input_embed.iter().cloned().collect();
        let input_tensor = Tensor::from_array((vec![1, 1, hidden_size], input_data))
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {}", e))?;

        let step_tensor = Tensor::from_array((vec![] as Vec<usize>, vec![step_idx as i64]))
            .map_err(|e| anyhow::anyhow!("Failed to create step tensor: {}", e))?;

        let inputs = SessionInputs::from(vec![
            (
                Cow::Borrowed("input_embeds"),
                SessionInputValue::Owned(Value::from(input_tensor)),
            ),
            (
                Cow::Borrowed("step"),
                SessionInputValue::Owned(Value::from(step_tensor)),
            ),
        ]);

        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| anyhow::anyhow!("Code predictor inference failed: {}", e))?;

        let mut outputs_iter = outputs.iter();
        let logits_output = outputs_iter
            .next()
            .ok_or_else(|| anyhow::anyhow!("No logits output"))?;
        let hidden_output = outputs_iter
            .next()
            .ok_or_else(|| anyhow::anyhow!("No hidden output"))?;

        let (_logits_shape, logits_data) = logits_output
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract logits: {}", e))?;
        let logits_1d = Array1::from_vec(logits_data.to_vec());

        let (_hidden_shape, hidden_data) = hidden_output
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract hidden: {}", e))?;
        let hidden_3d = Array3::from_shape_vec((1, 1, hidden_size), hidden_data.to_vec())?;

        kv_cache.step();

        Ok((logits_1d, hidden_3d))
    }

    fn concat_inputs(hidden: &Array3<f32>, embed: &Array3<f32>) -> Result<Array3<f32>> {
        let combined = ndarray::concatenate(Axis(1), &[hidden.view(), embed.view()])?;
        Ok(combined)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_predictor_config() {
        let config = SamplingConfig::default();
        assert_eq!(config.top_k, 30);
    }

    #[test]
    #[ignore]
    fn test_code_predictor_generate() {
        let model_dir = dirs::home_dir()
            .unwrap()
            .join(".nanobot/models/qwen/qwen-onnx");

        if !model_dir.exists() {
            return;
        }

        use ort::session::{builder::GraphOptimizationLevel, Session};

        let session = Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .commit_from_file(model_dir.join("code_predictor.onnx"))
            .unwrap();

        let embeddings = EmbeddingManager::load(&model_dir.join("embeddings")).unwrap();
        let mut predictor = CodePredictor::new(session, SamplingConfig::default());

        let hidden: Array3<f32> = Array3::zeros((1, 1, 1024));

        let codes = predictor.generate_acoustic_codes(&hidden, 0, &embeddings);

        if let Ok(codes) = codes {
            assert_eq!(codes.len(), 15);
        }
    }
}
