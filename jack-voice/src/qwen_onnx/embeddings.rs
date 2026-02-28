//! Embedding loader for Qwen3 TTS ONNX.
//!
//! Loads embeddings from `.npy` files in the model directory:
//! - `text_embedding.npy`: Text token embeddings [151936, 2048]
//! - `talker_codec_embedding.npy`: Codec token embeddings [3072, 1024]
//! - `cp_codec_embedding_*.npy`: Code predictor per-group embeddings [2048, 1024] × 15
//! - `text_projection_fc1_weight/bias.npy`: Text projection layer 1
//! - `text_projection_fc2_weight/bias.npy`: Text projection layer 2
//!
//! Speaker and language IDs are codec token IDs (not separate embeddings):
//! - Speaker IDs: ryan=3061, serena=3066, etc.
//! - Language IDs: english=2050, chinese=2055, etc.

use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;

/// Speaker token IDs in codec vocabulary.
pub const SPEAKER_IDS: &[(&str, i64)] = &[
    ("ryan", 3061),
    ("serena", 3066),
    ("vivian", 3065),
    ("uncle_fu", 3010),
    ("aiden", 2861),
    ("ono_anna", 2873),
    ("sohee", 2864),
    ("eric", 2875),
    ("dylan", 2878),
];

/// Language token IDs in codec vocabulary.
pub const LANGUAGE_IDS: &[(&str, i64)] = &[
    ("english", 2050),
    ("chinese", 2055),
    ("spanish", 2054),
    ("french", 2061),
    ("japanese", 2058),
    ("korean", 2064),
    ("german", 2053),
    ("italian", 2070),
    ("portuguese", 2071),
    ("russian", 2069),
];

/// Manages all embeddings for Qwen ONNX TTS.
pub struct EmbeddingManager {
    /// Text token embeddings: [text_vocab, text_embed_dim]
    /// [151936, 2048]
    pub text_embeddings: Array2<f32>,

    /// Text projection FC1 weight: [text_embed_dim, text_embed_dim]
    pub text_proj_fc1_weight: Array2<f32>,
    /// Text projection FC1 bias: [text_embed_dim]
    pub text_proj_fc1_bias: Array1<f32>,

    /// Text projection FC2 weight: [hidden_size, text_embed_dim]
    pub text_proj_fc2_weight: Array2<f32>,
    /// Text projection FC2 bias: [hidden_size]
    pub text_proj_fc2_bias: Array1<f32>,

    /// Talker codec embeddings: [codec_vocab, hidden_size]
    /// [3072, 1024]
    pub codec_embeddings: Array2<f32>,

    /// Code predictor per-group embeddings: 15 × [cp_vocab, hidden_size]
    /// Each is [2048, 1024]
    pub cp_codec_embeddings: Vec<Array2<f32>>,

    /// Speaker name → codec token ID
    pub speaker_ids: HashMap<String, i64>,

    /// Language name → codec token ID
    pub language_ids: HashMap<String, i64>,

    /// Hidden size for the model (1024 for 0.6B)
    pub hidden_size: usize,

    /// Text embedding dimension (2048)
    pub text_embed_dim: usize,
}

impl EmbeddingManager {
    /// Load all embeddings from the embeddings directory.
    pub fn load(embeddings_dir: &Path) -> Result<Self> {
        let text_path = embeddings_dir.join("text_embedding.npy");
        let codec_path = embeddings_dir.join("talker_codec_embedding.npy");

        if !text_path.exists() {
            anyhow::bail!("text_embedding.npy not found: {}", text_path.display());
        }
        if !codec_path.exists() {
            anyhow::bail!(
                "talker_codec_embedding.npy not found: {}",
                codec_path.display()
            );
        }

        let text_embeddings: Array2<f32> = read_npy(&text_path)
            .map_err(|e| anyhow::anyhow!("Failed to load text_embedding.npy: {}", e))?;

        let codec_embeddings: Array2<f32> = read_npy(&codec_path)
            .map_err(|e| anyhow::anyhow!("Failed to load talker_codec_embedding.npy: {}", e))?;

        let fc1_w: Array2<f32> =
            read_npy(&embeddings_dir.join("text_projection_fc1_weight.npy"))
                .map_err(|e| anyhow::anyhow!("Failed to load fc1 weight: {}", e))?;
        let fc1_b: Array1<f32> = read_npy(&embeddings_dir.join("text_projection_fc1_bias.npy"))
            .map_err(|e| anyhow::anyhow!("Failed to load fc1 bias: {}", e))?;
        let fc2_w: Array2<f32> =
            read_npy(&embeddings_dir.join("text_projection_fc2_weight.npy"))
                .map_err(|e| anyhow::anyhow!("Failed to load fc2 weight: {}", e))?;
        let fc2_b: Array1<f32> = read_npy(&embeddings_dir.join("text_projection_fc2_bias.npy"))
            .map_err(|e| anyhow::anyhow!("Failed to load fc2 bias: {}", e))?;

        let mut cp_codec_embeddings = Vec::with_capacity(15);
        for i in 0..15 {
            let cp_path = embeddings_dir.join(format!("cp_codec_embedding_{}.npy", i));
            if cp_path.exists() {
                let cp_emb: Array2<f32> = read_npy(&cp_path).map_err(|e| {
                    anyhow::anyhow!("Failed to load cp_codec_embedding_{}.npy: {}", i, e)
                })?;
                cp_codec_embeddings.push(cp_emb);
            }
        }

        let speaker_ids: HashMap<String, i64> = SPEAKER_IDS
            .iter()
            .map(|(name, id)| (name.to_string(), *id))
            .collect();

        let language_ids: HashMap<String, i64> = LANGUAGE_IDS
            .iter()
            .map(|(name, id)| (name.to_string(), *id))
            .collect();

        let hidden_size = codec_embeddings.ncols();
        let text_embed_dim = text_embeddings.ncols();

        Ok(Self {
            text_embeddings,
            text_proj_fc1_weight: fc1_w,
            text_proj_fc1_bias: fc1_b,
            text_proj_fc2_weight: fc2_w,
            text_proj_fc2_bias: fc2_b,
            codec_embeddings,
            cp_codec_embeddings,
            speaker_ids,
            language_ids,
            hidden_size,
            text_embed_dim,
        })
    }

    /// Get text embedding and project to hidden_size.
    pub fn text_embed_projected(&self, token_id: i64) -> Result<Vec<f32>> {
        let idx = token_id as usize;
        if idx >= self.text_embeddings.nrows() {
            anyhow::bail!(
                "Text token ID {} out of range (vocab size: {})",
                token_id,
                self.text_embeddings.nrows()
            );
        }

        let text_embed = self.text_embeddings.row(idx);
        let mut hidden = vec![0.0f32; self.text_embed_dim];
        for (i, &v) in text_embed.iter().enumerate() {
            hidden[i] = v;
        }

        hidden = self.apply_fc1(&hidden);
        hidden = self.apply_fc2(&hidden);

        Ok(hidden)
    }

    fn apply_fc1(&self, input: &[f32]) -> Vec<f32> {
        let mut output = self.text_proj_fc1_bias.to_vec();
        for j in 0..self.text_embed_dim {
            for (i, &x) in input.iter().enumerate() {
                output[j] += self.text_proj_fc1_weight[[j, i]] * x;
            }
        }
        output.iter().map(|&x| x.max(0.0)).collect()
    }

    fn apply_fc2(&self, input: &[f32]) -> Vec<f32> {
        let mut output = self.text_proj_fc2_bias.to_vec();
        for j in 0..self.hidden_size {
            for (i, &x) in input.iter().enumerate() {
                output[j] += self.text_proj_fc2_weight[[j, i]] * x;
            }
        }
        output
    }

    /// Get raw text embedding (no projection).
    pub fn text_embed(&self, token_id: i64) -> Result<Vec<f32>> {
        let idx = token_id as usize;
        if idx >= self.text_embeddings.nrows() {
            anyhow::bail!(
                "Text token ID {} out of range (vocab size: {})",
                token_id,
                self.text_embeddings.nrows()
            );
        }
        Ok(self.text_embeddings.row(idx).to_vec())
    }

    /// Get codec embedding for a semantic/acoustic token.
    pub fn codec_embed(&self, code: i64) -> Result<Vec<f32>> {
        let idx = code as usize;
        if idx >= self.codec_embeddings.nrows() {
            anyhow::bail!(
                "Codec token {} out of range (vocab size: {})",
                code,
                self.codec_embeddings.nrows()
            );
        }
        Ok(self.codec_embeddings.row(idx).to_vec())
    }

    /// Get code predictor codec embedding for a specific group.
    pub fn cp_codec_embed(&self, group: usize, code: i64) -> Result<Vec<f32>> {
        if group >= self.cp_codec_embeddings.len() {
            anyhow::bail!("Code predictor group {} out of range", group);
        }
        let idx = code as usize;
        let embeddings = &self.cp_codec_embeddings[group];
        if idx >= embeddings.nrows() {
            anyhow::bail!(
                "CP codec token {} out of range for group {} (vocab size: {})",
                code,
                group,
                embeddings.nrows()
            );
        }
        Ok(embeddings.row(idx).to_vec())
    }

    /// Get speaker codec token ID.
    pub fn speaker_token_id(&self, speaker: &str) -> Option<i64> {
        self.speaker_ids.get(speaker).copied()
    }

    /// Get language codec token ID.
    pub fn language_token_id(&self, language: &str) -> Option<i64> {
        self.language_ids.get(language).copied()
    }

    /// Get a zero embedding of hidden_size.
    pub fn zero_embed(&self) -> Vec<f32> {
        vec![0.0f32; self.hidden_size]
    }

    /// Sum multiple embeddings element-wise.
    pub fn sum_embeddings(embeds: &[&[f32]]) -> Vec<f32> {
        if embeds.is_empty() {
            return vec![];
        }

        let len = embeds[0].len();
        let mut sum = vec![0.0f32; len];

        for embed in embeds {
            for (i, &v) in embed.iter().enumerate() {
                sum[i] += v;
            }
        }

        sum
    }

    /// Get text vocabulary size.
    pub fn text_vocab_size(&self) -> usize {
        self.text_embeddings.nrows()
    }

    /// Get codec vocabulary size.
    pub fn codec_vocab_size(&self) -> usize {
        self.codec_embeddings.nrows()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_embeddings_empty() {
        let sum = EmbeddingManager::sum_embeddings(&[]);
        assert!(sum.is_empty());
    }

    #[test]
    fn test_sum_embeddings_single() {
        let embed = &[1.0, 2.0, 3.0];
        let sum = EmbeddingManager::sum_embeddings(&[embed]);
        assert_eq!(sum, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sum_embeddings_multiple() {
        let a = &[1.0, 2.0, 3.0];
        let b = &[4.0, 5.0, 6.0];
        let c = &[7.0, 8.0, 9.0];
        let sum = EmbeddingManager::sum_embeddings(&[a, b, c]);
        assert_eq!(sum, vec![12.0, 15.0, 18.0]);
    }

    #[test]
    fn test_speaker_ids() {
        assert_eq!(SPEAKER_IDS.len(), 9);
        assert!(SPEAKER_IDS.iter().any(|(n, _)| *n == "ryan"));
    }

    #[test]
    fn test_language_ids() {
        assert!(LANGUAGE_IDS.len() >= 6);
        assert!(LANGUAGE_IDS.iter().any(|(n, _)| *n == "english"));
    }

    #[test]
    fn test_load_fails_without_files() {
        let result = EmbeddingManager::load(Path::new("/nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_load_real_embeddings() {
        let embeddings_dir = dirs::home_dir()
            .unwrap()
            .join(".nanobot/models/qwen/qwen-onnx/embeddings");

        if !embeddings_dir.exists() {
            return;
        }

        let manager = EmbeddingManager::load(&embeddings_dir).unwrap();

        assert!(manager.text_vocab_size() > 100000);
        assert_eq!(manager.codec_vocab_size(), 3072);
        assert_eq!(manager.hidden_size, 1024);
        assert_eq!(manager.text_embed_dim, 2048);
        assert_eq!(manager.cp_codec_embeddings.len(), 15);
    }

    #[test]
    #[ignore]
    fn test_text_embed_projected() {
        let embeddings_dir = dirs::home_dir()
            .unwrap()
            .join(".nanobot/models/qwen/qwen-onnx/embeddings");

        if !embeddings_dir.exists() {
            return;
        }

        let manager = EmbeddingManager::load(&embeddings_dir).unwrap();
        let embed = manager.text_embed_projected(0).unwrap();
        assert_eq!(embed.len(), manager.hidden_size);
    }

    #[test]
    #[ignore]
    fn test_codec_embed_lookup() {
        let embeddings_dir = dirs::home_dir()
            .unwrap()
            .join(".nanobot/models/qwen/qwen-onnx/embeddings");

        if !embeddings_dir.exists() {
            return;
        }

        let manager = EmbeddingManager::load(&embeddings_dir).unwrap();
        let embed = manager.codec_embed(0).unwrap();
        assert_eq!(embed.len(), manager.hidden_size);
    }

    #[test]
    #[ignore]
    fn test_speaker_token_id() {
        let embeddings_dir = dirs::home_dir()
            .unwrap()
            .join(".nanobot/models/qwen/qwen-onnx/embeddings");

        if !embeddings_dir.exists() {
            return;
        }

        let manager = EmbeddingManager::load(&embeddings_dir).unwrap();
        assert_eq!(manager.speaker_token_id("ryan"), Some(3061));
        assert_eq!(manager.speaker_token_id("serena"), Some(3066));
    }

    #[test]
    #[ignore]
    fn test_language_token_id() {
        let embeddings_dir = dirs::home_dir()
            .unwrap()
            .join(".nanobot/models/qwen/qwen-onnx/embeddings");

        if !embeddings_dir.exists() {
            return;
        }

        let manager = EmbeddingManager::load(&embeddings_dir).unwrap();
        assert_eq!(manager.language_token_id("english"), Some(2050));
        assert_eq!(manager.language_token_id("chinese"), Some(2055));
    }
}
