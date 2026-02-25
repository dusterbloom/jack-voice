// Qwen3-TTS ONNX Runtime implementation
// Pure ONNX Runtime-based Qwen3-TTS inference (no Candle)

use std::borrow::Cow;
use std::path::PathBuf;

use ndarray::{Array, Array2, Array3, Axis};
use ort::{
    ep,
    session::{builder::GraphOptimizationLevel, Session, SessionInputValue, SessionInputs},
    value::Tensor,
};

use crate::models;
use crate::qwen_tokenizer::{QwenBpeTokenizer, QwenTextType};
use crate::tts::{AudioOutput, TtsError};

const SAMPLE_RATE: u32 = 24000;
const NUM_CODEBOOKS: usize = 16;
const ENCODE_DOWNSAMPLE_RATE: i64 = 1920;
const DECODE_UPSAMPLE_RATE: i64 = 1920;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenOnnxModelSize {
    Lite,
    Large,
}

pub struct QwenOnnxTts {
    model_dir: PathBuf,
    size: QwenOnnxModelSize,
    tokenizer: QwenBpeTokenizer,
    config: QwenOnnxConfig,
    text_project_session: Session,
    codec_embed_session: Session,
    code_predictor_embed_session: Session,
    talker_prefill_session: Session,
    talker_decode_session: Session,
    code_predictor_session: Session,
    tokenizer_encode_session: Session,
    tokenizer_decode_session: Session,
    speaker_encoder_session: Session,
}

#[derive(Debug, Clone)]
pub struct QwenOnnxConfig {
    pub num_code_groups: i64,
    pub vocab_size: i64,
    pub codec_bos_id: i64,
    pub codec_eos_id: i64,
    pub codec_pad_id: i64,
    pub codec_nothink_id: i64,
    pub codec_think_id: i64,
    pub codec_think_bos_id: i64,
    pub codec_think_eos_id: i64,
    pub tts_bos_id: i64,
    pub tts_eos_id: i64,
    pub tts_pad_id: i64,
    pub sample_rate: i64,
    pub n_fft: i64,
    pub hop_size: i64,
    pub win_size: i64,
    pub num_mels: i64,
    pub fmin: f64,
    pub fmax: f64,
}

fn rand_simple() -> f32 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as f32) / (u32::MAX as f32)
}

impl QwenOnnxTts {
    pub fn new(model_dir: &std::path::Path, size: QwenOnnxModelSize) -> Result<Self, TtsError> {
        let model_dir = model_dir.to_path_buf();
        let onnx_base_dir = if size == QwenOnnxModelSize::Lite {
            model_dir.join("models").join("Qwen3-TTS-12Hz-0.6B-Base")
        } else {
            model_dir.join("models").join("Qwen3-TTS-12Hz-1.7B-Base")
        };

        let vocab_path = onnx_base_dir.join("vocab.json");
        let merges_path = onnx_base_dir.join("merges.txt");
        let config_path = onnx_base_dir.join("tokenizer_config.json");

        let tokenizer = QwenBpeTokenizer::load(&vocab_path, &merges_path, &config_path)
            .map_err(|e| TtsError::InitError(format!("Failed to load tokenizer: {}", e)))?;

        let config = Self::load_config(&onnx_base_dir)?;

        let onnx_dir = if size == QwenOnnxModelSize::Lite {
            model_dir
                .parent()
                .map(|p| p.join("qwen3-onnx-lite"))
                .unwrap_or_else(|| model_dir.clone())
        } else {
            model_dir.clone()
        };

        log::info!(
            "[QwenOnnx] Loading ONNX models from: {}",
            onnx_dir.display()
        );

        let text_project_session = Self::load_session(onnx_dir.join("text_project.onnx"))?;
        let codec_embed_session = Self::load_session(onnx_dir.join("codec_embed.onnx"))?;
        let code_predictor_embed_session =
            Self::load_session(onnx_dir.join("code_predictor_embed.onnx"))?;
        let talker_prefill_session = Self::load_session(onnx_dir.join("talker_prefill.onnx"))?;
        let talker_decode_session = Self::load_session(onnx_dir.join("talker_decode.onnx"))?;
        let code_predictor_session = Self::load_session(onnx_dir.join("code_predictor.onnx"))?;
        let tokenizer_encode_session =
            Self::load_session(onnx_dir.join("tokenizer12hz_encode.onnx"))?;
        let tokenizer_decode_session =
            Self::load_session(onnx_dir.join("tokenizer12hz_decode.onnx"))?;
        let speaker_encoder_session = Self::load_session(onnx_dir.join("speaker_encoder.onnx"))?;

        log::info!(
            "[QwenOnnx] Initialized {:?} model from {}",
            size,
            model_dir.display()
        );

        Ok(Self {
            model_dir,
            size,
            tokenizer,
            config,
            text_project_session,
            codec_embed_session,
            code_predictor_embed_session,
            talker_prefill_session,
            talker_decode_session,
            code_predictor_session,
            tokenizer_encode_session,
            tokenizer_decode_session,
            speaker_encoder_session,
        })
    }

    fn load_session(path: std::path::PathBuf) -> Result<Session, TtsError> {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let builder = Session::builder()
            .map_err(|e| TtsError::InitError(format!("Failed to create session builder: {}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| TtsError::InitError(format!("Failed to set optimization level: {}", e)))?
            .with_intra_threads(num_threads)
            .map_err(|e| TtsError::InitError(format!("Failed to set intra threads: {}", e)))?
            .with_execution_providers([ep::CUDA::default().build(), ep::CPU::default().build()])
            .map_err(|e| {
                TtsError::InitError(format!("Failed to set execution providers: {}", e))
            })?;

        let session = builder
            .commit_from_file(&path)
            .map_err(|e| TtsError::InitError(format!("Failed to load ONNX session: {}", e)))?;

        Ok(session)
    }

    fn load_config(model_dir: &std::path::Path) -> Result<QwenOnnxConfig, TtsError> {
        let config_path = model_dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| TtsError::InitError(format!("Failed to read config: {}", e)))?;

        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| TtsError::InitError(format!("Failed to parse config: {}", e)))?;

        let talker_config = config_json
            .get("talker_config")
            .ok_or_else(|| TtsError::InitError("Missing talker_config".to_string()))?;

        let speaker_config = config_json
            .get("speaker_encoder_config")
            .ok_or_else(|| TtsError::InitError("Missing speaker_encoder_config".to_string()))?;

        Ok(QwenOnnxConfig {
            num_code_groups: talker_config
                .get("num_code_groups")
                .and_then(|v| v.as_i64())
                .unwrap_or(16),
            vocab_size: talker_config
                .get("vocab_size")
                .and_then(|v| v.as_i64())
                .unwrap_or(151936),
            codec_bos_id: talker_config
                .get("codec_bos_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(151643),
            codec_eos_id: talker_config
                .get("codec_eos_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(151645),
            codec_pad_id: talker_config
                .get("codec_pad_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(151643),
            codec_nothink_id: talker_config
                .get("codec_nothink_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(151655),
            codec_think_id: talker_config
                .get("codec_think_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(151656),
            codec_think_bos_id: talker_config
                .get("codec_think_bos_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(151657),
            codec_think_eos_id: talker_config
                .get("codec_think_eos_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(151658),
            tts_bos_id: config_json
                .get("tts_bos_token_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(151642),
            tts_eos_id: config_json
                .get("tts_eos_token_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(151643),
            tts_pad_id: config_json
                .get("tts_pad_token_id")
                .and_then(|v| v.as_i64())
                .unwrap_or(151643),
            sample_rate: speaker_config
                .get("sample_rate")
                .and_then(|v| v.as_i64())
                .unwrap_or(24000),
            n_fft: speaker_config
                .get("n_fft")
                .and_then(|v| v.as_i64())
                .unwrap_or(1024),
            hop_size: speaker_config
                .get("hop_size")
                .and_then(|v| v.as_i64())
                .unwrap_or(256),
            win_size: speaker_config
                .get("win_size")
                .and_then(|v| v.as_i64())
                .unwrap_or(1024),
            num_mels: speaker_config
                .get("num_mels")
                .and_then(|v| v.as_i64())
                .unwrap_or(128),
            fmin: speaker_config
                .get("fmin")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0),
            fmax: speaker_config
                .get("fmax")
                .and_then(|v| v.as_f64())
                .unwrap_or(12000.0),
        })
    }

    fn compute_mel_spectrogram(&self, audio: &[f32]) -> Result<Array2<f32>, TtsError> {
        let n_fft = self.config.n_fft as usize;
        let hop_size = self.config.hop_size as usize;
        let win_size = self.config.win_size as usize;
        let num_mels = self.config.num_mels as usize;

        let n_bins = n_fft / 2 + 1;
        let frames = ((audio.len().saturating_sub(win_size)) / hop_size).max(1);

        let mut mel_spectrogram = Array2::<f32>::zeros((num_mels, frames));

        for frame_idx in 0..frames {
            let start = frame_idx * hop_size;
            let end = (start + win_size).min(audio.len());
            if end <= start {
                break;
            }
            let mut window = vec![0.0f32; win_size];
            for (i, j) in (start..end).enumerate() {
                let hann = 0.5
                    * (1.0
                        - (2.0 * std::f32::consts::PI * i as f32 / (win_size as f32 - 1.0)).cos());
                window[i] = audio[j] * hann;
            }

            let magnitudes = Self::compute_fft_magnitude(&window, n_fft);
            for m in 0..num_mels {
                let f_mel_low = 2595.0 * ((self.config.fmin as f32 / 700.0) + 1.0).log10();
                let f_mel_high = 2595.0 * ((self.config.fmax as f32 / 700.0) + 1.0).log10();
                let f_mel =
                    f_mel_low + (f_mel_high - f_mel_low) * (m as f32 / (num_mels - 1) as f32);
                let f_hz = 700.0 * 10f32.powf(f_mel / 2595.0) - 700.0;
                let bin = (f_hz * n_fft as f32 / self.config.sample_rate as f32).round() as usize;
                if bin < magnitudes.len() {
                    mel_spectrogram[[m, frame_idx]] = (magnitudes[bin] + 1e-8).log10().max(-10.0);
                }
            }
        }

        Ok(mel_spectrogram)
    }

    fn compute_fft_magnitude(input: &[f32], n: usize) -> Vec<f32> {
        let half_n = n / 2;
        let mut result = vec![0.0f32; half_n];

        for k in 0..half_n {
            let mut sum_r = 0.0f32;
            let mut sum_i = 0.0f32;
            for t in 0..n.min(input.len()) {
                let angle = -2.0 * std::f32::consts::PI * (k * t) as f32 / n as f32;
                sum_r += input[t] * angle.cos();
                sum_i += input[t] * angle.sin();
            }
            result[k] = (sum_r * sum_r + sum_i * sum_i).sqrt();
        }
        result
    }

    fn resample_audio(&self, audio: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
        if from_sr == to_sr {
            return audio.to_vec();
        }
        let ratio = to_sr as f64 / from_sr as f64;
        let new_len = (audio.len() as f64 * ratio) as usize;
        let mut output = vec![0.0f32; new_len];
        for i in 0..new_len {
            let src_idx = i as f64 / ratio;
            let src_idx_floor = src_idx.floor() as usize;
            let frac = (src_idx - src_idx_floor as f64) as f32;
            if src_idx_floor < audio.len() - 1 {
                output[i] = audio[src_idx_floor] * (1.0 - frac) + audio[src_idx_floor + 1] * frac;
            } else if src_idx_floor < audio.len() {
                output[i] = audio[src_idx_floor];
            }
        }
        output
    }

    fn text_project(&mut self, input_ids: &[i64]) -> Result<Array3<f32>, TtsError> {
        let input_ids_shape = [1_usize, input_ids.len()];
        let input_tensor = Tensor::from_array((input_ids_shape, input_ids.to_vec()))
            .map_err(|e| TtsError::SynthesisError(format!("Failed to create tensor: {}", e)))?;

        let inputs = SessionInputs::from(std::collections::HashMap::from([(
            Cow::Borrowed("input_ids"),
            SessionInputValue::from(input_tensor),
        )]));

        let outputs = self
            .text_project_session
            .run(inputs)
            .map_err(|e| TtsError::SynthesisError(format!("text_project failed: {}", e)))?;

        let embed = &outputs[0];
        let shape = embed.shape();
        let (_shape, tensor_data) = embed.try_extract_tensor::<f32>().map_err(|e| {
            TtsError::SynthesisError(format!("Failed to extract embed tensor: {}", e))
        })?;
        let data: Vec<f32> = tensor_data.to_vec();

        let arr = Array3::from_shape_vec(
            (shape[0] as usize, shape[1] as usize, shape[2] as usize),
            data,
        )
        .map_err(|e| TtsError::SynthesisError(format!("Failed to create array: {}", e)))?;

        Ok(arr)
    }

    fn codec_embed(&mut self, code_ids: &[i64]) -> Result<Array3<f32>, TtsError> {
        let input_ids_shape = [1_usize, code_ids.len()];
        let input_tensor = Tensor::from_array((input_ids_shape, code_ids.to_vec()))
            .map_err(|e| TtsError::SynthesisError(format!("Failed to create tensor: {}", e)))?;

        let inputs = SessionInputs::from(std::collections::HashMap::from([(
            Cow::Borrowed("input_ids"),
            SessionInputValue::from(input_tensor),
        )]));

        let outputs = self
            .codec_embed_session
            .run(inputs)
            .map_err(|e| TtsError::SynthesisError(format!("codec_embed failed: {}", e)))?;

        let embed = &outputs[0];
        let shape = embed.shape();
        let (_shape, tensor_data) = embed.try_extract_tensor::<f32>().map_err(|e| {
            TtsError::SynthesisError(format!("Failed to extract embed tensor: {}", e))
        })?;
        let data: Vec<f32> = tensor_data.to_vec();

        let arr = Array3::from_shape_vec(
            (shape[0] as usize, shape[1] as usize, shape[2] as usize),
            data,
        )
        .map_err(|e| TtsError::SynthesisError(format!("Failed to create array: {}", e)))?;

        Ok(arr)
    }

    fn speaker_encode(&mut self, audio: &[f32], sample_rate: u32) -> Result<Vec<f32>, TtsError> {
        let audio = self.resample_audio(audio, sample_rate, self.config.sample_rate as u32);
        let mel = self.compute_mel_spectrogram(&audio)?;
        let mel_t = mel.t();

        let input_shape = [1_usize, mel_t.nrows(), mel_t.ncols()];
        let mel_data: Vec<f32> = mel_t.iter().copied().collect();
        let mel_tensor = Tensor::from_array((input_shape, mel_data))
            .map_err(|e| TtsError::SynthesisError(format!("Failed to create mel tensor: {}", e)))?;

        let inputs = SessionInputs::from(std::collections::HashMap::from([(
            Cow::Borrowed("mels"),
            SessionInputValue::from(mel_tensor),
        )]));

        let outputs = self
            .speaker_encoder_session
            .run(inputs)
            .map_err(|e| TtsError::SynthesisError(format!("Speaker encoder failed: {}", e)))?;

        let embedding = &outputs[0];
        let (_shape, tensor_data) = embedding
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::SynthesisError(format!("Failed to extract embedding: {}", e)))?;
        let data: Vec<f32> = tensor_data.to_vec();

        Ok(data)
    }

    fn generate_codes(
        &mut self,
        input_embeds: Array3<f32>,
        attention_mask: Array2<i64>,
        trailing_text_hidden: Array3<f32>,
        tts_pad_embed: Array3<f32>,
        max_new_tokens: usize,
    ) -> Result<Vec<Vec<i64>>, TtsError> {
        let batch_size = input_embeds.shape()[0];
        let num_code_groups = self.config.num_code_groups as usize;
        let hidden_dim = input_embeds.shape()[2];

        let mut full_embeds = input_embeds.clone();
        let mut trailing = trailing_text_hidden.clone();
        full_embeds
            .append(Axis(1), trailing.view())
            .map_err(|e| TtsError::SynthesisError(format!("Failed to append trailing: {}", e)))?;
        let mut pad = tts_pad_embed.clone();
        full_embeds
            .append(Axis(1), pad.view())
            .map_err(|e| TtsError::SynthesisError(format!("Failed to append pad: {}", e)))?;

        let inputs_embeds_data: Vec<f32> = full_embeds.iter().copied().collect();
        let inputs_embeds_tensor = Tensor::from_array((
            [
                batch_size,
                inputs_embeds_data.len() / batch_size / hidden_dim,
                hidden_dim,
            ],
            inputs_embeds_data,
        ))
        .map_err(|e| TtsError::SynthesisError(format!("Failed to create inputs_embeds: {}", e)))?;

        let mask_data: Vec<i64> = attention_mask.iter().copied().collect();
        let attention_mask_tensor =
            Tensor::from_array(([batch_size, mask_data.len() / batch_size], mask_data)).map_err(
                |e| TtsError::SynthesisError(format!("Failed to create attention_mask: {}", e)),
            )?;

        let prefill_inputs = SessionInputs::from(std::collections::HashMap::from([
            (
                Cow::Borrowed("inputs_embeds"),
                SessionInputValue::from(inputs_embeds_tensor),
            ),
            (
                Cow::Borrowed("attention_mask"),
                SessionInputValue::from(attention_mask_tensor),
            ),
        ]));

        let prefill_outputs = self
            .talker_prefill_session
            .run(prefill_inputs)
            .map_err(|e| TtsError::SynthesisError(format!("talker_prefill failed: {}", e)))?;

        let logits = &prefill_outputs[0];
        let vocab_size = self.config.vocab_size as usize;
        let eos_token_id = self.config.codec_eos_id as i64;

        let mut generated_codes: Vec<Vec<i64>> = vec![Vec::new(); batch_size];
        let mut finished = vec![false; batch_size];

        let (_shape, tensor_data) = logits
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::SynthesisError(format!("Failed to extract logits: {}", e)))?;
        let logits_data: Vec<f32> = tensor_data.to_vec();
        let logits_array = Array::from_shape_vec(
            (
                batch_size,
                logits_data.len() / batch_size / vocab_size,
                vocab_size,
            ),
            logits_data,
        )
        .map_err(|e| TtsError::SynthesisError(format!("Failed to reshape logits: {}", e)))?;

        for step in 0..max_new_tokens {
            let last_logits_view =
                logits_array.index_axis(Axis(1), step.min(logits_array.shape()[1] - 1));
            let last_logits = last_logits_view.to_owned();

            let next_tokens = Self::sample_tokens_array(&last_logits, 50, 1.0, 0.9, 2048);

            for b in 0..batch_size {
                if finished[b] {
                    continue;
                }
                let token = next_tokens[b];
                if token == eos_token_id {
                    finished[b] = true;
                    continue;
                }
                generated_codes[b].push(token);
            }

            if finished.iter().all(|&f| f) {
                break;
            }
        }

        Ok(generated_codes)
    }

    fn sample_tokens_array(
        logits: &Array2<f32>,
        top_k: i64,
        _top_p: f32,
        temperature: f32,
        codebook_size: i64,
    ) -> Vec<i64> {
        let batch = logits.shape()[0];
        let vocab = logits.shape()[1];

        let mut scaled = logits.clone();
        for val in scaled.iter_mut() {
            *val /= temperature;
        }

        let mut result = Vec::with_capacity(batch);

        for b in 0..batch {
            let mut row = scaled.index_axis(Axis(0), b).to_vec();

            for i in 0..vocab {
                if i >= codebook_size as usize {
                    row[i] = -1e9;
                }
            }

            if top_k > 0 && top_k < vocab as i64 && top_k < codebook_size {
                let mut sorted: Vec<(usize, f32)> =
                    row.iter().enumerate().map(|(i, v)| (i, *v)).collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let threshold = sorted[top_k as usize].1;
                for val in row.iter_mut() {
                    if *val < threshold {
                        *val = -1e9;
                    }
                }
            }

            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = row.iter().map(|v| (v - max_val).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let probs: Vec<f32> = exp.iter().map(|v| v / sum).collect();

            let mut cumsum = 0.0f32;
            let mut sampled = 0usize;
            let r = rand_simple();
            for (i, p) in probs.iter().enumerate() {
                cumsum += *p;
                if r <= cumsum {
                    sampled = i;
                    break;
                }
            }

            result.push(sampled as i64);
        }

        result
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        voice_clone_audio: Option<&[f32]>,
    ) -> Result<AudioOutput, TtsError> {
        let input_text = self.tokenizer.build_text(text, QwenTextType::Assistant);
        let input_ids = self.tokenizer.encode(&input_text);

        let text_embeds = self.text_project(&input_ids)?;

        let hidden_dim = text_embeds.shape()[2];
        let seq_len = text_embeds.shape()[1];

        let mask_len = seq_len + 10 + 1;
        let attention_mask = Array2::from_elem((1, mask_len), 1i64);

        let trailing_hidden = Array3::from_elem((1, 10, hidden_dim), 0.0f32);
        let tts_pad_embed = Array3::from_elem((1, 1, hidden_dim), 0.0f32);

        let codes = self.generate_codes(
            text_embeds,
            attention_mask,
            trailing_hidden,
            tts_pad_embed,
            512,
        )?;

        let flat_codes: Vec<i64> = codes.into_iter().flat_map(|c| c).collect();

        let num_codes = flat_codes.len() / NUM_CODEBOOKS;
        let mut audio_samples = Vec::new();

        for chunk in flat_codes.chunks(1024 * NUM_CODEBOOKS) {
            if chunk.is_empty() {
                continue;
            }
            let decoded = self.decode_chunk(chunk)?;
            audio_samples.extend(decoded);
        }

        if audio_samples.is_empty() {
            audio_samples = vec![0.0f32; SAMPLE_RATE as usize];
        }

        Ok(AudioOutput {
            samples: audio_samples,
            sample_rate: SAMPLE_RATE,
        })
    }

    fn decode_chunk(&mut self, codes: &[i64]) -> Result<Vec<f32>, TtsError> {
        let num_codes = codes.len() / NUM_CODEBOOKS;
        if num_codes == 0 {
            return Ok(Vec::new());
        }

        let mut audio_codes_padded = vec![0i64; num_codes * NUM_CODEBOOKS];
        for (i, &code) in codes.iter().enumerate() {
            if i < audio_codes_padded.len() {
                audio_codes_padded[i] = code;
            }
        }

        let input_shape = [1_usize, num_codes, NUM_CODEBOOKS];
        let audio_codes_tensor =
            Tensor::from_array((input_shape, audio_codes_padded)).map_err(|e| {
                TtsError::SynthesisError(format!("Failed to create codes tensor: {}", e))
            })?;

        let inputs = SessionInputs::from(std::collections::HashMap::from([(
            Cow::Borrowed("audio_codes"),
            SessionInputValue::from(audio_codes_tensor),
        )]));

        let outputs = self
            .tokenizer_decode_session
            .run(inputs)
            .map_err(|e| TtsError::SynthesisError(format!("Tokenizer decode failed: {}", e)))?;

        let audio_values = &outputs[0];
        let (_shape, tensor_data) = audio_values
            .try_extract_tensor::<f32>()
            .map_err(|e| TtsError::SynthesisError(format!("Failed to extract audio: {}", e)))?;
        let data: Vec<f32> = tensor_data.to_vec();

        let target_length = num_codes * DECODE_UPSAMPLE_RATE as usize;
        let actual_length = data.len().min(target_length);

        Ok(data.into_iter().take(actual_length).collect())
    }

    pub fn supports_voice_cloning(&self) -> bool {
        self.size == QwenOnnxModelSize::Large
    }

    pub fn supports_preset_speakers(&self) -> bool {
        self.size == QwenOnnxModelSize::Lite
    }
}

pub fn qwen_onnx_model_ready(lite: bool) -> bool {
    if lite {
        models::qwen3_onnx_lite_model_ready()
    } else {
        models::qwen3_onnx_model_ready()
    }
}

pub fn qwen_onnx_model_dir(lite: bool) -> PathBuf {
    models::qwen3_onnx_model_dir(lite)
}
