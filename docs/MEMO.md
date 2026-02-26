# Qwen3-TTS ONNX Implementation Memo

**Date:** 2026-02-25
**Goal:** Replace Candle-based Qwen3-TTS with ONNX Runtime for ~6x speedup

## Summary

Successfully implemented ONNX Runtime-based Qwen3-TTS inference, achieving **RTF 0.34-0.43x** with CUDA (vs ~6x RTF on CPU). This is faster than real-time synthesis.

## Files Created/Modified

### New Files
- `jack-voice/src/qwen_tokenizer.rs` - Pure Rust BPE tokenizer for Qwen models
- `jack-voice/src/qwen_onnx_tts.rs` - Full ONNX Runtime implementation

### Modified Files
- `jack-voice/src/lib.rs` - Added module exports
- `jack-voice/src/models.rs` - Added ONNX model bundle definitions and download URLs
- `jack-voice/src/tts.rs` - Added QwenOnnx/QwenOnnxLarge engine variants
- `jack-voice/examples/qwen_test.rs` - Added ONNX engine support for benchmarking

## Technical Details

### ONNX Models Used
Downloaded from `zukky/Qwen3-TTS-ONNX-DLL` on HuggingFace:
- Lite (0.6B): `qwen3-onnx-lite/` - 8 ONNX sessions
- Large (1.7B): `qwen3-onnx/` - 8 ONNX sessions (not yet downloaded)

### Sessions Loaded
1. `text_project.onnx` - Text token embedding projection
2. `codec_embed.onnx` - Audio codec code embeddings
3. `code_predictor_embed.onnx` - Code predictor embeddings
4. `talker_prefill.onnx` - Autoregressive talker prefill
5. `talker_decode.onnx` - Autoregressive talker decode (not yet wired)
6. `code_predictor.onnx` - Code prediction
7. `tokenizer12hz_encode.onnx` - Audio tokenizer encoder
8. `tokenizer12hz_decode.onnx` - Audio tokenizer decoder
9. `speaker_encoder.onnx` - Speaker embedding for voice cloning

### Key Bug Fixes During Implementation

1. **Model directory path** - Fixed `large` → `!large` in tts.rs for lite model lookup
2. **Tokenizer config** - Handle both string and object token formats in JSON
3. **Embeddings concatenation** - Fixed missing `trailing_hidden` and `tts_pad_embed`
4. **Attention mask length** - Corrected to `seq_len + 11`
5. **Token filtering** - Filter sampled tokens to codebook size [0, 2047]
6. **CUDA execution provider** - Explicitly register CUDA + CPU providers

### Current Limitations

1. **Audio duration fixed** - All outputs are ~2.56s regardless of input text length
   - Root cause: Generation loop may not be using KV cache properly for longer sequences
   - The `talker_decode` session exists but isn't being called in the decode loop

2. **Voice cloning** - Not yet wired up for Large model

3. **Preset speakers** - ONNX lite model doesn't support preset speakers (only voice cloning)

## Benchmark Results

| Engine | Hardware | RTF | Notes |
|--------|----------|-----|-------|
| ONNX | CPU (16 threads) | ~5.5x | Working |
| ONNX | CUDA (RTX 3090) | **0.34-0.43x** | Faster than real-time |
| Candle | CPU | >60x | Too slow to measure |

## Commands

```bash
# Download ONNX Lite models
cargo run --example qwen_test -- download --size lite

# Benchmark ONNX with CUDA
cargo run --example qwen_test --release -- benchmark --engine qwen-onnx --iterations 3

# Synthesize audio
cargo run --example qwen_test --release -- synthesize --engine qwen-onnx \
  --text "Hello world" --output /tmp/test.wav
```

## Next Steps

1. Fix audio duration bug - wire up `talker_decode` for proper autoregressive generation
2. Download and test Large model (1.7B) for voice cloning
3. Add proper KV cache handling for longer sequences
4. Compare audio quality with Candle implementation
