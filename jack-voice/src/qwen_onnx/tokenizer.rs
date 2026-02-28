//! BPE Tokenizer for Qwen3 TTS ONNX.
//!
//! Uses HuggingFace `tokenizers` crate for text tokenization.
//! Loads from `tokenizer.json` (preferred) or `vocab.json` + `merges.txt`.

use std::path::Path;

use anyhow::Result;
use tokenizers::Tokenizer;

/// Special token IDs for Qwen3 TTS.
pub mod special_tokens {
    pub const BOS_TOKEN: u32 = 151643; // <|im_start|>
    pub const EOS_TOKEN: u32 = 151645; // <|im_end|>
    pub const ASSISTANT_TOKEN: u32 = 151644;

    pub const CODEC_PAD: u32 = 2148;
    pub const CODEC_BOS: u32 = 2149;
    pub const CODEC_EOS: u32 = 2150;
    pub const CODEC_VOCAB_SIZE: usize = 3072;

    pub const LANG_ENGLISH: u32 = 2050;
    pub const LANG_CHINESE: u32 = 2055;
}

/// BPE tokenizer for Qwen3 TTS text processing.
pub struct BpeTokenizer {
    tokenizer: Tokenizer,
}

impl BpeTokenizer {
    /// Load tokenizer from tokenizer.json file.
    pub fn from_files(tokenizer_dir: &Path) -> Result<Self> {
        let tokenizer_path = tokenizer_dir.join("tokenizer.json");

        if tokenizer_path.exists() {
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer.json: {}", e))?;
            return Ok(Self { tokenizer });
        }

        let vocab_path = tokenizer_dir.join("vocab.json");
        let merges_path = tokenizer_dir.join("merges.txt");

        if !vocab_path.exists() {
            anyhow::bail!(
                "No tokenizer.json or vocab.json found in {}",
                tokenizer_dir.display()
            );
        }
        if !merges_path.exists() {
            anyhow::bail!("merges.txt not found: {}", merges_path.display());
        }

        anyhow::bail!(
            "tokenizer.json not found. Please download it from HuggingFace (e.g., Qwen/Qwen2.5-0.5B) and place in {}",
            tokenizer_dir.display()
        );
    }

    /// Encode text for TTS synthesis.
    pub fn encode(&self, text: &str) -> Result<Vec<i64>> {
        let formatted = format!("<|im_start|>assistant\n{}<|im_end|>", text);

        let encoding = self
            .tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        Ok(ids)
    }

    /// Encode text without chat template (raw tokenization).
    pub fn encode_raw(&self, text: &str) -> Result<Vec<i64>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        Ok(ids)
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true) as usize
    }

    /// Get BOS token ID.
    pub fn bos_token_id(&self) -> i64 {
        special_tokens::BOS_TOKEN as i64
    }

    /// Get EOS token ID.
    pub fn eos_token_id(&self) -> i64 {
        special_tokens::EOS_TOKEN as i64
    }
}

/// Trait alias for text tokenization.
pub trait TextTokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<i64>>;
    fn bos_token_id(&self) -> i64;
    fn eos_token_id(&self) -> i64;
}

impl TextTokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i64>> {
        Self::encode(self, text)
    }

    fn bos_token_id(&self) -> i64 {
        Self::bos_token_id(self)
    }

    fn eos_token_id(&self) -> i64 {
        Self::eos_token_id(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens_constants() {
        assert_eq!(special_tokens::BOS_TOKEN, 151643);
        assert_eq!(special_tokens::EOS_TOKEN, 151645);
        assert_eq!(special_tokens::CODEC_EOS, 2150);
        assert_eq!(special_tokens::CODEC_VOCAB_SIZE, 3072);
    }

    #[test]
    fn test_bpe_tokenizer_new_fails_without_files() {
        let result = BpeTokenizer::from_files(Path::new("/nonexistent"));
        assert!(result.is_err());
    }

    #[test]
    fn test_bos_eos_token_ids() {
        assert_eq!(special_tokens::BOS_TOKEN, 151643);
        assert_eq!(special_tokens::EOS_TOKEN, 151645);
    }

    #[test]
    #[ignore]
    fn test_encode_hello_world() {
        let tokenizer_dir = dirs::home_dir()
            .unwrap()
            .join(".nanobot/models/qwen/qwen-onnx/tokenizer");

        if !tokenizer_dir.exists() {
            return;
        }

        let tokenizer = BpeTokenizer::from_files(&tokenizer_dir).unwrap();
        let ids = tokenizer.encode("Hello world").unwrap();

        assert!(!ids.is_empty());
    }

    #[test]
    #[ignore]
    fn test_vocab_size() {
        let tokenizer_dir = dirs::home_dir()
            .unwrap()
            .join(".nanobot/models/qwen/qwen-onnx/tokenizer");

        if !tokenizer_dir.exists() {
            return;
        }

        let tokenizer = BpeTokenizer::from_files(&tokenizer_dir).unwrap();
        let vocab_size = tokenizer.vocab_size();

        assert!(vocab_size > 100000);
    }
}
