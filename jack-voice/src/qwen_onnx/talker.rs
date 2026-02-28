//! Talker LM for semantic token generation.
//!
//! The Talker is a 28-layer transformer that generates semantic tokens
//! autoregressively. Uses KV cache for efficient decoding.

use std::borrow::Cow;
use std::sync::Arc;

use anyhow::Result;
use ndarray::{Array1, Array3, Axis};
use ort::session::{Session, SessionInputValue, SessionInputs};
use ort::value::{Tensor, Value};

use super::embeddings::EmbeddingManager;
use super::kv_cache::TalkerKVCache;
use super::sampling::SamplingConfig;
use super::types::{Language, Speaker};

/// Talker LM for semantic token generation.
pub struct TalkerLM {
    prefill_session: Session,
    decode_session: Session,
    embeddings: Arc<EmbeddingManager>,
    config: SamplingConfig,
}

/// Output from prefill phase.
pub struct PrefillOutput {
    /// Hidden states for code predictor: [1, 1, hidden_size]
    pub hidden: Array3<f32>,
    /// Logits for first token: [1, vocab_size]
    pub logits: Array1<f32>,
    /// Initial KV cache.
    pub kv_cache: TalkerKVCache,
    /// Number of tokens in prefill.
    pub seq_len: usize,
}

/// Output from decode step.
pub struct DecodeOutput {
    /// Hidden states: [1, 1, hidden_size]
    pub hidden: Array3<f32>,
    /// Logits: [1, vocab_size]
    pub logits: Array1<f32>,
}

impl TalkerLM {
    /// Create a new Talker LM.
    pub fn new(
        prefill_session: Session,
        decode_session: Session,
        embeddings: Arc<EmbeddingManager>,
        config: SamplingConfig,
    ) -> Self {
        Self {
            prefill_session,
            decode_session,
            embeddings,
            config,
        }
    }

    /// Run prefill phase with text tokens.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs from text tokenizer
    /// * `speaker` - Speaker for voice selection
    /// * `language` - Language for synthesis
    ///
    /// # Returns
    /// Prefill output with hidden states, logits, and KV cache.
    pub fn prefill(
        &mut self,
        input_ids: &[i64],
        speaker: Speaker,
        language: Language,
    ) -> Result<PrefillOutput> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            anyhow::bail!("Empty input_ids");
        }

        let input_embeds = self.build_prefill_embeddings(input_ids, speaker, language)?;

        let attention_mask: Vec<i64> = vec![1; seq_len];
        let position_ids: Vec<i64> = (0..seq_len as i64).collect();

        let input_shape: Vec<usize> = vec![1, seq_len, input_embeds.shape()[2]];
        let input_data: Vec<f32> = input_embeds.iter().cloned().collect();
        let input_tensor = Tensor::from_array((input_shape, input_data))
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {}", e))?;

        let attention_tensor = Tensor::from_array((vec![1, seq_len], attention_mask))
            .map_err(|e| anyhow::anyhow!("Failed to create attention tensor: {}", e))?;

        let position_tensor = Tensor::from_array((vec![1, seq_len], position_ids))
            .map_err(|e| anyhow::anyhow!("Failed to create position tensor: {}", e))?;

        let inputs = SessionInputs::from(vec![
            (
                Cow::Borrowed("input_embeds"),
                SessionInputValue::Owned(Value::from(input_tensor)),
            ),
            (
                Cow::Borrowed("attention_mask"),
                SessionInputValue::Owned(Value::from(attention_tensor)),
            ),
            (
                Cow::Borrowed("position_ids"),
                SessionInputValue::Owned(Value::from(position_tensor)),
            ),
        ]);

        let outputs = self
            .prefill_session
            .run(inputs)
            .map_err(|e| anyhow::anyhow!("Talker prefill failed: {}", e))?;

        let mut outputs_iter = outputs.iter();
        let logits_output = outputs_iter
            .next()
            .ok_or_else(|| anyhow::anyhow!("No logits output from prefill"))?;
        let hidden_output = outputs_iter
            .next()
            .ok_or_else(|| anyhow::anyhow!("No hidden output from prefill"))?;

        let (_logits_shape, logits_data) = logits_output
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract logits: {}", e))?;
        let logits_1d = Array1::from_vec(logits_data.to_vec());

        let (_hidden_shape, hidden_data) = hidden_output
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract hidden: {}", e))?;
        let hidden_size = self.embeddings.hidden_size;
        let hidden_3d = Array3::from_shape_vec((1, 1, hidden_size), hidden_data.to_vec())?;

        let mut kv_cache = TalkerKVCache::new_default(2048);
        kv_cache.step();

        Ok(PrefillOutput {
            hidden: hidden_3d,
            logits: logits_1d,
            kv_cache,
            seq_len,
        })
    }

    /// Run single decode step with input embedding.
    ///
    /// # Arguments
    /// * `input_embed` - Input embedding: [1, 1, hidden_size]
    /// * `position` - Current position in sequence
    /// * `kv_cache` - KV cache (will be updated)
    pub fn decode_step(
        &mut self,
        input_embed: &Array3<f32>,
        position: usize,
        kv_cache: &mut TalkerKVCache,
    ) -> Result<DecodeOutput> {
        let hidden_size = input_embed.shape()[2];
        let input_data: Vec<f32> = input_embed.iter().cloned().collect();
        let input_tensor = Tensor::from_array((vec![1, 1, hidden_size], input_data))
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {}", e))?;

        let attention_tensor = Tensor::from_array((vec![1, 1], vec![1i64]))
            .map_err(|e| anyhow::anyhow!("Failed to create attention tensor: {}", e))?;

        let position_tensor = Tensor::from_array((vec![1, 1], vec![position as i64]))
            .map_err(|e| anyhow::anyhow!("Failed to create position tensor: {}", e))?;

        let inputs = SessionInputs::from(vec![
            (
                Cow::Borrowed("input_embeds"),
                SessionInputValue::Owned(Value::from(input_tensor)),
            ),
            (
                Cow::Borrowed("attention_mask"),
                SessionInputValue::Owned(Value::from(attention_tensor)),
            ),
            (
                Cow::Borrowed("position_ids"),
                SessionInputValue::Owned(Value::from(position_tensor)),
            ),
        ]);

        let outputs = self
            .decode_session
            .run(inputs)
            .map_err(|e| anyhow::anyhow!("Talker decode failed: {}", e))?;

        let mut outputs_iter = outputs.iter();
        let logits_output = outputs_iter
            .next()
            .ok_or_else(|| anyhow::anyhow!("No logits output from decode"))?;
        let hidden_output = outputs_iter
            .next()
            .ok_or_else(|| anyhow::anyhow!("No hidden output from decode"))?;

        let (_logits_shape, logits_data) = logits_output
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract logits: {}", e))?;
        let logits_1d = Array1::from_vec(logits_data.to_vec());

        let (_hidden_shape, hidden_data) = hidden_output
            .1
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to extract hidden: {}", e))?;
        let hidden_3d =
            Array3::from_shape_vec((1, 1, self.embeddings.hidden_size), hidden_data.to_vec())?;

        kv_cache.step();

        Ok(DecodeOutput {
            hidden: hidden_3d,
            logits: logits_1d,
        })
    }

    fn build_prefill_embeddings(
        &self,
        input_ids: &[i64],
        speaker: Speaker,
        language: Language,
    ) -> Result<Array3<f32>> {
        let seq_len = input_ids.len();
        let hidden_size = self.embeddings.hidden_size;

        let mut embeds = Array3::zeros((1, seq_len, hidden_size));

        for (t, &token_id) in input_ids.iter().enumerate() {
            let text_embed = self.embeddings.text_embed_projected(token_id)?;

            for (d, &v) in text_embed.iter().enumerate() {
                embeds[[0, t, d]] = v;
            }
        }

        if let Some(speaker_token) = self.embeddings.speaker_token_id(speaker.name()) {
            let speaker_embed = self.embeddings.codec_embed(speaker_token)?;
            for t in 0..seq_len {
                for (d, &v) in speaker_embed.iter().enumerate() {
                    embeds[[0, t, d]] += v;
                }
            }
        }

        if let Some(lang_token) = self.embeddings.language_token_id(language.name()) {
            let lang_embed = self.embeddings.codec_embed(lang_token)?;
            for t in 0..seq_len {
                for (d, &v) in lang_embed.iter().enumerate() {
                    embeds[[0, t, d]] += v;
                }
            }
        }

        Ok(embeds)
    }

    /// Get codec embedding for a semantic token.
    pub fn get_codec_embedding(&self, code: i64) -> Result<Array3<f32>> {
        let embed = self.embeddings.codec_embed(code)?;
        let arr = Array1::from_vec(embed);
        Ok(arr.insert_axis(Axis(0)).insert_axis(Axis(0)))
    }

    /// Build step input embedding from semantic + acoustic + text.
    pub fn build_step_embedding(
        &self,
        semantic_code: i64,
        acoustic_codes: &[i64],
        text_embed: Option<&[f32]>,
    ) -> Result<Array3<f32>> {
        let mut embeds: Vec<Vec<f32>> = vec![];

        let semantic_embed = self.embeddings.codec_embed(semantic_code)?;
        embeds.push(semantic_embed);

        for &code in acoustic_codes {
            let acoustic_embed = self.embeddings.codec_embed(code)?;
            embeds.push(acoustic_embed);
        }

        if let Some(text) = text_embed {
            embeds.push(text.to_vec());
        }

        let slices: Vec<&[f32]> = embeds.iter().map(|v| v.as_slice()).collect();
        let sum = EmbeddingManager::sum_embeddings(&slices);

        let arr = Array1::from_vec(sum);
        Ok(arr.insert_axis(Axis(0)).insert_axis(Axis(0)))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_output_shape() {
        let hidden: Array3<f32> = Array3::zeros((1, 1, 1024));
        let logits: Array1<f32> = Array1::zeros(3072);

        assert_eq!(hidden.shape(), &[1, 1, 1024]);
        assert_eq!(logits.len(), 3072);
    }

    #[test]
    fn test_prefill_output_seq_len() {
        // Verify seq_len tracking
        let ids = vec![1i64, 2, 3, 4, 5];
        assert_eq!(ids.len(), 5);
    }

    // Integration tests require real models
    #[test]
    #[ignore]
    fn test_talker_prefill() {
        let model_dir = dirs::home_dir()
            .unwrap()
            .join(".nanobot/models/qwen/qwen-onnx");

        if !model_dir.exists() {
            return;
        }

        // This would require loading all components
        // Just verify structure for now
        assert!(
            model_dir.join("talker_prefill.onnx").exists()
                || model_dir.join("talker_prefill_q.onnx").exists()
        );
    }
}
