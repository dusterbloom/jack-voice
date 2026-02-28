//! KV Cache management for autoregressive ONNX TTS generation.
//!
//! ONNX requires explicit KV cache inputs/outputs for autoregressive models.
//! This module provides cache management for:
//! - Talker LM (28 layers)
//! - Code Predictor (5 layers)

use ndarray::{Array4, Axis};

/// KV cache for a single transformer layer.
#[derive(Debug, Clone)]
pub struct LayerKVCache {
    /// Key cache: [batch, num_kv_heads, max_seq, head_dim]
    pub k: Array4<f32>,
    /// Value cache: [batch, num_kv_heads, max_seq, head_dim]
    pub v: Array4<f32>,
    /// Current sequence length.
    pub current_len: usize,
}

impl LayerKVCache {
    /// Create a new layer KV cache.
    pub fn new(batch: usize, num_heads: usize, head_dim: usize, max_seq: usize) -> Self {
        Self {
            k: Array4::zeros((batch, num_heads, max_seq, head_dim)),
            v: Array4::zeros((batch, num_heads, max_seq, head_dim)),
            current_len: 0,
        }
    }

    /// Get the current cache slice for K.
    pub fn k_slice(&self) -> ndarray::ArrayView4<f32> {
        self.k.slice(ndarray::s![.., .., ..self.current_len, ..])
    }

    /// Get the current cache slice for V.
    pub fn v_slice(&self) -> ndarray::ArrayView4<f32> {
        self.v.slice(ndarray::s![.., .., ..self.current_len, ..])
    }

    /// Update cache at the current position.
    pub fn update(&mut self, k_new: &ndarray::ArrayView3<f32>, v_new: &ndarray::ArrayView3<f32>) {
        let pos = self.current_len;

        // Copy new values into cache
        for h in 0..k_new.len_of(Axis(0)) {
            for d in 0..k_new.len_of(Axis(2)) {
                self.k[[0, h, pos, d]] = k_new[[h, 0, d]];
                self.v[[0, h, pos, d]] = v_new[[h, 0, d]];
            }
        }
    }

    /// Increment sequence length.
    pub fn step(&mut self) {
        self.current_len += 1;
    }

    /// Reset cache for new sequence.
    pub fn reset(&mut self) {
        self.current_len = 0;
        self.k.fill(0.0);
        self.v.fill(0.0);
    }

    /// Get total capacity.
    pub fn capacity(&self) -> usize {
        self.k.shape()[2]
    }
}

/// KV cache for Talker LM (28 layers).
#[derive(Debug, Clone)]
pub struct TalkerKVCache {
    /// Per-layer caches.
    pub layers: Vec<LayerKVCache>,
    /// Number of layers.
    pub num_layers: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl TalkerKVCache {
    /// Create a new Talker KV cache.
    ///
    /// # Arguments
    /// * `batch` - Batch size (typically 1)
    /// * `num_layers` - Number of transformer layers (28)
    /// * `num_kv_heads` - Number of KV heads (8 for GQA)
    /// * `head_dim` - Head dimension (128)
    /// * `max_seq` - Maximum sequence length
    pub fn new(
        batch: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKVCache::new(batch, num_kv_heads, head_dim, max_seq))
            .collect();

        Self {
            layers,
            num_layers,
            num_kv_heads,
            head_dim,
        }
    }

    /// Create with default Talker configuration.
    pub fn new_default(max_seq: usize) -> Self {
        Self::new(1, 28, 8, 128, max_seq)
    }

    /// Get current sequence length.
    pub fn current_len(&self) -> usize {
        self.layers.first().map(|l| l.current_len).unwrap_or(0)
    }

    /// Increment all layers.
    pub fn step(&mut self) {
        for layer in &mut self.layers {
            layer.step();
        }
    }

    /// Reset all layers.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Convert to ONNX input format (flat vector for all layers).
    pub fn to_onnx_input(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.total_elements());

        for layer in &self.layers {
            // Flatten K then V for this layer
            for h in 0..self.num_kv_heads {
                for pos in 0..layer.current_len {
                    for d in 0..self.head_dim {
                        flat.push(layer.k[[0, h, pos, d]]);
                    }
                }
            }
            for h in 0..self.num_kv_heads {
                for pos in 0..layer.current_len {
                    for d in 0..self.head_dim {
                        flat.push(layer.v[[0, h, pos, d]]);
                    }
                }
            }
        }

        flat
    }

    /// Get total number of elements.
    fn total_elements(&self) -> usize {
        let len = self.current_len();
        self.num_layers * 2 * self.num_kv_heads * len * self.head_dim
    }
}

/// KV cache for Code Predictor (5 layers).
#[derive(Debug, Clone)]
pub struct CodePredictorKVCache {
    /// Per-layer caches.
    pub layers: Vec<LayerKVCache>,
    /// Number of layers.
    pub num_layers: usize,
    /// Number of KV heads.
    pub num_kv_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl CodePredictorKVCache {
    /// Create a new Code Predictor KV cache.
    pub fn new(
        batch: usize,
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
    ) -> Self {
        let layers = (0..num_layers)
            .map(|_| LayerKVCache::new(batch, num_kv_heads, head_dim, max_seq))
            .collect();

        Self {
            layers,
            num_layers,
            num_kv_heads,
            head_dim,
        }
    }

    /// Create with default Code Predictor configuration.
    pub fn new_default(max_seq: usize) -> Self {
        Self::new(1, 5, 8, 128, max_seq)
    }

    /// Get current sequence length.
    pub fn current_len(&self) -> usize {
        self.layers.first().map(|l| l.current_len).unwrap_or(0)
    }

    /// Increment all layers.
    pub fn step(&mut self) {
        for layer in &mut self.layers {
            layer.step();
        }
    }

    /// Reset all layers (call before each new semantic token).
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
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
    fn test_layer_kv_cache_new() {
        let cache = LayerKVCache::new(1, 8, 128, 1024);

        assert_eq!(cache.k.shape(), &[1, 8, 1024, 128]);
        assert_eq!(cache.v.shape(), &[1, 8, 1024, 128]);
        assert_eq!(cache.current_len, 0);
    }

    #[test]
    fn test_layer_kv_cache_step() {
        let mut cache = LayerKVCache::new(1, 8, 128, 1024);
        assert_eq!(cache.current_len, 0);

        cache.step();
        assert_eq!(cache.current_len, 1);

        cache.step();
        assert_eq!(cache.current_len, 2);
    }

    #[test]
    fn test_layer_kv_cache_reset() {
        let mut cache = LayerKVCache::new(1, 8, 128, 1024);
        cache.step();
        cache.step();
        assert_eq!(cache.current_len, 2);

        cache.reset();
        assert_eq!(cache.current_len, 0);
    }

    #[test]
    fn test_layer_kv_cache_capacity() {
        let cache = LayerKVCache::new(1, 8, 128, 1024);
        assert_eq!(cache.capacity(), 1024);
    }

    #[test]
    fn test_talker_kv_cache_new() {
        let cache = TalkerKVCache::new(1, 28, 8, 128, 1024);

        assert_eq!(cache.layers.len(), 28);
        assert_eq!(cache.num_layers, 28);
        assert_eq!(cache.num_kv_heads, 8);
        assert_eq!(cache.head_dim, 128);
    }

    #[test]
    fn test_talker_kv_cache_new_default() {
        let cache = TalkerKVCache::new_default(2048);

        assert_eq!(cache.layers.len(), 28);
        assert_eq!(cache.current_len(), 0);
    }

    #[test]
    fn test_talker_kv_cache_step() {
        let mut cache = TalkerKVCache::new_default(1024);
        assert_eq!(cache.current_len(), 0);

        cache.step();
        assert_eq!(cache.current_len(), 1);

        // All layers should be incremented
        for layer in &cache.layers {
            assert_eq!(layer.current_len, 1);
        }
    }

    #[test]
    fn test_talker_kv_cache_reset() {
        let mut cache = TalkerKVCache::new_default(1024);
        cache.step();
        cache.step();
        cache.step();

        cache.reset();
        assert_eq!(cache.current_len(), 0);

        for layer in &cache.layers {
            assert_eq!(layer.current_len, 0);
        }
    }

    #[test]
    fn test_code_predictor_kv_cache_new() {
        let cache = CodePredictorKVCache::new(1, 5, 8, 128, 64);

        assert_eq!(cache.layers.len(), 5);
        assert_eq!(cache.num_layers, 5);
    }

    #[test]
    fn test_code_predictor_kv_cache_new_default() {
        let cache = CodePredictorKVCache::new_default(64);

        assert_eq!(cache.layers.len(), 5);
        assert_eq!(cache.current_len(), 0);
    }

    #[test]
    fn test_code_predictor_kv_cache_reset() {
        let mut cache = CodePredictorKVCache::new_default(64);
        cache.step();
        cache.step();

        cache.reset();
        assert_eq!(cache.current_len(), 0);
    }
}
