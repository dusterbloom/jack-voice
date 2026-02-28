//! Qwen3 ONNX TTS synthesis components.
//!
//! This module provides the individual components for ONNX-based TTS synthesis:
//!
//! - `tokenizer`: BPE text tokenization using HuggingFace tokenizers
//! - `embeddings`: Embedding loader from .npy files
//! - `sessions`: ONNX session management
//! - `kv_cache`: KV cache for autoregressive generation
//! - `talker`: Talker LM (prefill + decode)
//! - `code_predictor`: Acoustic code generation
//! - `vocoder`: Code-to-audio decoder
//! - `sampling`: Token sampling strategies

mod code_predictor;
mod embeddings;
mod kv_cache;
mod sampling;
mod sessions;
mod talker;
mod tokenizer;
mod types;
mod vocoder;

pub use code_predictor::CodePredictor;
pub use embeddings::EmbeddingManager;
pub use kv_cache::{CodePredictorKVCache, LayerKVCache, TalkerKVCache};
pub use sampling::{create_rng, sample_token, SamplingConfig};
pub use sessions::OnnxSessions;
pub use talker::{DecodeOutput, PrefillOutput, TalkerLM};
pub use tokenizer::special_tokens;
pub use tokenizer::{BpeTokenizer, TextTokenizer};
pub use types::{Language, OnnxTtsConfig, Speaker};
pub use vocoder::Vocoder;
