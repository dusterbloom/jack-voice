// Qwen3-TTS BPE Tokenizer
// Pure Rust implementation of BPE tokenization for Qwen3-ONNX

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenTextType {
    Ref,
    Instruct,
    Assistant,
}

pub struct QwenBpeTokenizer {
    vocab: HashMap<String, i64>,
    merges: Vec<(String, String)>,
    pat: regex::Regex,
    bos_id: i64,
    eos_id: i64,
    pad_id: i64,
    bos_str: String,
    eos_str: String,
    pad_str: String,
}

#[derive(Deserialize)]
struct TokenizerConfig {
    #[serde(rename = "bos_token")]
    bos_token: Option<TokenOrString>,
    #[serde(rename = "eos_token")]
    eos_token: Option<TokenOrString>,
    #[serde(rename = "pad_token")]
    pad_token: Option<TokenOrString>,
    #[serde(rename = "unk_token")]
    unk_token: Option<TokenOrString>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum TokenOrString {
    Info(TokenInfo),
    Str(String),
}

impl TokenOrString {
    fn content(&self) -> String {
        match self {
            TokenOrString::Info(t) => t.content.clone(),
            TokenOrString::Str(s) => s.clone(),
        }
    }

    fn id(&self) -> Option<i64> {
        match self {
            TokenOrString::Info(t) => t.id,
            TokenOrString::Str(_) => None,
        }
    }
}

#[derive(Deserialize)]
struct TokenInfo {
    #[serde(rename = "content")]
    content: String,
    #[serde(rename = "id")]
    id: Option<i64>,
}

impl QwenBpeTokenizer {
    pub fn load(vocab_path: &Path, merges_path: &Path, config_path: &Path) -> Result<Self, String> {
        let vocab: HashMap<String, i64> =
            serde_json::from_str(&fs::read_to_string(vocab_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;

        let merges_text = fs::read_to_string(merges_path).map_err(|e| e.to_string())?;
        let mut merges = Vec::new();
        for line in merges_text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                merges.push((parts[0].to_string(), parts[1].to_string()));
            }
        }

        let config: TokenizerConfig =
            serde_json::from_str(&fs::read_to_string(config_path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;

        let bos_id = config.bos_token.as_ref().and_then(|t| t.id()).unwrap_or(1);
        let eos_id = config.eos_token.as_ref().and_then(|t| t.id()).unwrap_or(2);
        let pad_id = config.pad_token.as_ref().and_then(|t| t.id()).unwrap_or(0);

        let bos_str = config
            .bos_token
            .as_ref()
            .map(|t| t.content())
            .unwrap_or_else(|| "<|im_start|>".to_string());
        let eos_str = config
            .eos_token
            .as_ref()
            .map(|t| t.content())
            .unwrap_or_else(|| "<|im_end|>".to_string());
        let pad_str = config
            .pad_token
            .as_ref()
            .map(|t| t.content())
            .unwrap_or_else(|| "<|pad|>".to_string());

        let pat = regex::Regex::new(
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|[\p{N}]++| ?[^\s\p{L}\p{N}]++|[\r\n/]+|\s++",
        )
        .map_err(|e| e.to_string())?;

        Ok(Self {
            vocab,
            merges,
            pat,
            bos_id,
            eos_id,
            pad_id,
            bos_str,
            eos_str,
            pad_str,
        })
    }

    fn get_word_pieces(&self, text: &str) -> Vec<String> {
        let mut pieces = Vec::new();
        for mat in self.pat.find_iter(text) {
            let piece = mat.as_str().to_string();
            let mut chars: Vec<char> = piece.chars().collect();
            while !chars.is_empty() {
                let mut end = chars.len();
                let mut found = false;
                while end > 0 {
                    let subpiece: String = chars[..end].iter().collect();
                    if self.vocab.contains_key(&subpiece) {
                        pieces.push(subpiece);
                        chars = chars[end..].to_vec();
                        found = true;
                        break;
                    }
                    end -= 1;
                }
                if !found {
                    pieces.push(chars[0].to_string());
                    chars = chars[1..].to_vec();
                }
            }
        }
        pieces
    }

    fn bpe_merge(&self, word_pieces: &[String]) -> Vec<String> {
        let mut pieces: Vec<String> = word_pieces.to_vec();

        while pieces.len() > 1 {
            let mut best_pair_pos: Option<usize> = None;
            let mut best_pair_idx = 0;

            for i in 0..pieces.len() - 1 {
                let pair = (&pieces[i], &pieces[i + 1]);
                if let Some(pos) = self
                    .merges
                    .iter()
                    .position(|(a, b)| a == pair.0 && b == pair.1)
                {
                    if best_pair_pos.is_none() || pos < best_pair_pos.unwrap() {
                        best_pair_pos = Some(pos);
                        best_pair_idx = i;
                    }
                }
            }

            if let Some(_) = best_pair_pos {
                let first = pieces[best_pair_idx].clone();
                let second = pieces[best_pair_idx + 1].clone();
                let merged = format!("{}{}", first, second);

                let mut new_pieces = Vec::new();
                new_pieces.extend(pieces[..best_pair_idx].iter().cloned());
                new_pieces.push(merged);
                new_pieces.extend(pieces[best_pair_idx + 2..].iter().cloned());
                pieces = new_pieces;
            } else {
                break;
            }
        }

        pieces
    }

    pub fn encode(&self, text: &str) -> Vec<i64> {
        let word_pieces = self.get_word_pieces(text);
        let merged = self.bpe_merge(&word_pieces);
        merged
            .iter()
            .filter_map(|p| self.vocab.get(p).copied())
            .collect()
    }

    pub fn build_text(&self, text: &str, text_type: QwenTextType) -> String {
        match text_type {
            QwenTextType::Ref => {
                format!("{}system\n<|ref|>\n{}{}", self.bos_str, text, self.eos_str)
            }
            QwenTextType::Instruct => format!(
                "{}system\n<|instruct|>\n{}{}",
                self.bos_str, text, self.eos_str
            ),
            QwenTextType::Assistant => {
                format!("{}assistant\n{} {}", self.bos_str, text, self.eos_str)
            }
        }
    }

    pub fn bos_id(&self) -> i64 {
        self.bos_id
    }

    pub fn eos_id(&self) -> i64 {
        self.eos_id
    }

    pub fn pad_id(&self) -> i64 {
        self.pad_id
    }
}
