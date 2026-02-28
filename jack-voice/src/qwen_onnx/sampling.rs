//! Token sampling strategies for ONNX TTS generation.

use std::cmp::Ordering;

use anyhow::Result;
use ndarray::Array1;
use rand::distributions::{Distribution, WeightedIndex};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Configuration for token sampling.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature for softmax (1.0 = normal, <1.0 = more deterministic).
    pub temperature: f32,
    /// Top-k filtering (0 = disabled).
    pub top_k: usize,
    /// Random seed (None = random).
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 30,
            seed: None,
        }
    }
}

/// Sample a token from logits.
///
/// # Arguments
/// * `logits` - Raw logits from the model [vocab_size]
/// * `config` - Sampling configuration
/// * `eos_token_id` - EOS token ID to never suppress
/// * `rng` - Random number generator
///
/// # Returns
/// Sampled token ID.
pub fn sample_token(
    logits: &Array1<f32>,
    config: &SamplingConfig,
    eos_token_id: i64,
    rng: &mut StdRng,
) -> Result<i64> {
    // Apply temperature
    let scaled = if config.temperature != 1.0 {
        logits.mapv(|v| v / config.temperature)
    } else {
        logits.clone()
    };

    // Apply top-k filtering
    let filtered = if config.top_k > 0 && config.top_k < scaled.len() {
        top_k_filter(&scaled, config.top_k, eos_token_id)
    } else {
        scaled
    };

    // Convert to probabilities via softmax
    let probs = softmax(&filtered);

    // Sample from distribution
    let token_id = sample_from_probs(&probs, rng)?;

    Ok(token_id as i64)
}

/// Sample a token from 2D logits [1, vocab_size].
pub fn sample_token_2d(
    logits: &ndarray::Array2<f32>,
    config: &SamplingConfig,
    eos_token_id: i64,
    rng: &mut StdRng,
) -> Result<i64> {
    // Flatten to 1D
    let flat = logits.row(0).to_owned();
    sample_token(&flat, config, eos_token_id, rng)
}

/// Apply top-k filtering: keep only top k logits, set rest to -inf.
fn top_k_filter(logits: &Array1<f32>, k: usize, keep_token: i64) -> Array1<f32> {
    let mut filtered = logits.clone();

    // Get indices of top-k values
    let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();

    // Sort by value descending
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // Keep top-k indices (plus EOS token)
    let mut keep_indices = std::collections::HashSet::new();
    for (idx, _) in indexed.iter().take(k) {
        keep_indices.insert(*idx);
    }
    keep_indices.insert(keep_token as usize);

    // Set non-kept to -inf
    for (i, v) in filtered.iter_mut().enumerate() {
        if !keep_indices.contains(&i) {
            *v = f32::NEG_INFINITY;
        }
    }

    filtered
}

/// Apply softmax to logits.
fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    // Find max for numerical stability
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max)
    let exp_sum: f32 = logits.iter().map(|&v| (v - max).exp()).sum();

    if exp_sum == 0.0 || exp_sum.is_nan() || exp_sum.is_infinite() {
        // Fallback: uniform distribution over non-inf tokens
        let valid_count = logits.iter().filter(|&&v| !v.is_infinite()).count();
        return logits.mapv(|v| {
            if v.is_infinite() {
                0.0
            } else {
                1.0 / valid_count as f32
            }
        });
    }

    logits.mapv(|v| (v - max).exp() / exp_sum)
}

/// Sample from probability distribution.
fn sample_from_probs(probs: &Array1<f32>, rng: &mut StdRng) -> Result<usize> {
    // Filter out zero/near-zero probabilities
    let weights: Vec<f32> = probs
        .iter()
        .map(|&p| if p > 1e-10 { p } else { 0.0 })
        .collect();

    let dist = WeightedIndex::new(&weights)
        .map_err(|e| anyhow::anyhow!("Failed to create weighted distribution: {}", e))?;

    Ok(dist.sample(rng))
}

/// Create RNG from optional seed.
pub fn create_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    }
}

/// Build token suppression mask.
///
/// Suppresses tokens in range [vocab_size - 1024, vocab_size) except EOS.
pub fn build_suppression_mask(vocab_size: usize, eos_token_id: i64) -> Vec<bool> {
    let suppress_start = vocab_size.saturating_sub(1024);
    let mut mask = vec![false; vocab_size];

    for i in suppress_start..vocab_size {
        if i as i64 != eos_token_id {
            mask[i] = true; // true = suppress
        }
    }

    mask
}

/// Apply suppression mask to logits (set suppressed tokens to -inf).
pub fn apply_suppression(logits: &mut Array1<f32>, mask: &[bool]) {
    for (i, &suppress) in mask.iter().enumerate() {
        if suppress && i < logits.len() {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_config_default() {
        let config = SamplingConfig::default();
        assert!((config.temperature - 1.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 30);
        assert!(config.seed.is_none());
    }

    #[test]
    fn test_softmax_basic() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits);

        // Check sums to 1
        let sum: f32 = probs.sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check ordering preserved
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_with_neg_inf() {
        let logits = Array1::from_vec(vec![f32::NEG_INFINITY, 2.0, f32::NEG_INFINITY]);
        let probs = softmax(&logits);

        // All probability on the non-inf token
        assert_eq!(probs[1], 1.0);
        assert_eq!(probs[0], 0.0);
    }

    #[test]
    fn test_top_k_filter() {
        let logits = Array1::from_vec(vec![1.0, 5.0, 3.0, 4.0, 2.0]);
        let filtered = top_k_filter(&logits, 2, -1); // k=2, no keep_token

        // Top 2 are indices 1 (5.0) and 3 (4.0)
        assert!(!filtered[1].is_infinite());
        assert!(!filtered[3].is_infinite());
        assert!(filtered[0].is_infinite());
        assert!(filtered[2].is_infinite());
        assert!(filtered[4].is_infinite());
    }

    #[test]
    fn test_top_k_filter_preserves_eos() {
        let logits = Array1::from_vec(vec![1.0, 5.0, 3.0, 4.0, 2.0]);
        let filtered = top_k_filter(&logits, 2, 0); // keep token 0

        // Token 0 should be preserved even though it's not top-k
        assert!(!filtered[0].is_infinite());
        assert!(!filtered[1].is_infinite());
        assert!(!filtered[3].is_infinite());
    }

    #[test]
    fn test_sample_token_deterministic_with_seed() {
        let logits = Array1::from_vec(vec![0.0, 5.0, 0.0]); // Strong preference for index 1
        let config = SamplingConfig {
            temperature: 0.1, // Low temperature = more deterministic
            top_k: 0,
            seed: Some(42),
        };
        let mut rng = create_rng(config.seed);

        // Should always sample index 1 with low temperature
        for _ in 0..10 {
            let token = sample_token(&logits, &config, -1, &mut rng).unwrap();
            assert_eq!(token, 1);
        }
    }

    #[test]
    fn test_build_suppression_mask() {
        let mask = build_suppression_mask(3072, 2150);

        // Check size
        assert_eq!(mask.len(), 3072);

        // Check suppression range
        let suppress_start = 3072 - 1024;
        assert!(!mask[0]); // Not suppressed
        assert!(!mask[suppress_start - 1]); // Not suppressed
        assert!(mask[suppress_start]); // Suppressed
        assert!(mask[2149]); // Suppressed
        assert!(!mask[2150]); // EOS not suppressed
        assert!(mask[2151]); // Suppressed
        assert!(mask[3071]); // Suppressed
    }

    #[test]
    fn test_apply_suppression() {
        let mut logits = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let mask = vec![false, true, false, true, false];

        apply_suppression(&mut logits, &mask);

        assert!(!logits[0].is_infinite());
        assert!(logits[1].is_infinite());
        assert!(!logits[2].is_infinite());
        assert!(logits[3].is_infinite());
        assert!(!logits[4].is_infinite());
    }

    #[test]
    fn test_create_rng_with_seed() {
        let mut rng1 = create_rng(Some(42));
        let mut rng2 = create_rng(Some(42));

        // Same seed = same sequence
        let v1: u32 = rand::Rng::gen(&mut rng1);
        let v2: u32 = rand::Rng::gen(&mut rng2);
        assert_eq!(v1, v2);
    }
}
