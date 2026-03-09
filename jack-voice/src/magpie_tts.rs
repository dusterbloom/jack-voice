use std::path::Path;
use std::sync::Once;

use magpie_rs::{ExportManifest, MagpieRuntime, SynthesisRequest};
use tch::{set_num_interop_threads, set_num_threads, Device};

use crate::models;
use crate::tts::{AudioOutput, TtsError};

const DEFAULT_LANGUAGE: &str = "it";
const DEFAULT_SPEAKER: &str = "2"; // Speaker 2 is 1.8x faster than speaker 0
const STREAMING_CHUNK_FRAMES: usize = 4;

static TORCH_THREADS: Once = Once::new();

pub struct MagpieTts {
    runtime: MagpieRuntime,
    manifest: ExportManifest,
    speaker_index: usize,
    language: String,
}

impl MagpieTts {
    pub fn new() -> Result<Self, TtsError> {
        Self::new_with_voice(DEFAULT_SPEAKER, DEFAULT_LANGUAGE)
    }

    pub fn new_with_voice(voice: &str, language: &str) -> Result<Self, TtsError> {
        let manifest_path = models::get_magpie_manifest_path()?;
        Self::from_manifest(&manifest_path, voice, language)
    }

    fn from_manifest(manifest_path: &Path, voice: &str, language: &str) -> Result<Self, TtsError> {
        TORCH_THREADS.call_once(|| {
            set_num_threads(4);
            set_num_interop_threads(1);
        });

        let manifest = ExportManifest::from_path(manifest_path)
            .map_err(|e| TtsError::InitError(format!("MagPie manifest: {}", e)))?;
        let language = normalize_language(language);
        if !manifest.magpie.supported_languages.contains_key(&language) {
            return Err(TtsError::InitError(format!(
                "Unsupported MagPie language '{}'",
                language
            )));
        }
        let speaker_index = parse_speaker_index(voice, manifest.magpie.num_baked_speakers)?;

        // Use Metal/MPS for GPU acceleration on macOS, fallback to CPU
        #[cfg(target_os = "macos")]
        let device = Device::Mps;
        #[cfg(not(target_os = "macos"))]
        let device = Device::Cpu;

        let runtime = MagpieRuntime::load(manifest_path, device)
            .map_err(|e| TtsError::InitError(format!("MagPie runtime: {}", e)))?;
        Ok(Self {
            runtime,
            manifest,
            speaker_index,
            language,
        })
    }

    pub fn set_language(&mut self, language: &str) -> Result<(), TtsError> {
        let language = normalize_language(language);
        if !self
            .manifest
            .magpie
            .supported_languages
            .contains_key(&language)
        {
            return Err(TtsError::InitError(format!(
                "Unsupported MagPie language '{}'",
                language
            )));
        }
        self.language = language;
        Ok(())
    }

    pub fn set_speaker(&mut self, voice: &str) -> Result<(), TtsError> {
        self.speaker_index = parse_speaker_index(voice, self.manifest.magpie.num_baked_speakers)?;
        Ok(())
    }

    pub fn current_language(&self) -> &str {
        &self.language
    }

    pub fn sample_rate(&self) -> u32 {
        self.manifest.codec.sample_rate as u32
    }

    pub fn synthesize(&self, text: &str) -> Result<AudioOutput, TtsError> {
        let output = self
            .runtime
            .synthesize(&SynthesisRequest {
                text: text.to_string(),
                language: self.language.clone(),
                speaker_index: self.speaker_index,
                emit_stream_chunks: false,
                stream_batch_frames: STREAMING_CHUNK_FRAMES,
            })
            .map_err(|e| TtsError::SynthesisError(format!("MagPie synthesis failed: {}", e)))?;

        Ok(AudioOutput {
            samples: output.audio,
            sample_rate: output.sample_rate as u32,
        })
    }

    pub fn synthesize_streaming<F>(&self, text: &str, on_chunk: &mut F) -> Result<u32, TtsError>
    where
        F: FnMut(&[f32], u32) -> bool,
    {
        let output = self
            .runtime
            .synthesize(&SynthesisRequest {
                text: text.to_string(),
                language: self.language.clone(),
                speaker_index: self.speaker_index,
                emit_stream_chunks: true,
                stream_batch_frames: STREAMING_CHUNK_FRAMES,
            })
            .map_err(|e| TtsError::SynthesisError(format!("MagPie synthesis failed: {}", e)))?;

        let sample_rate = output.sample_rate as u32;
        for chunk in output.audio_chunks {
            if !on_chunk(&chunk.samples, sample_rate) {
                break;
            }
        }

        Ok(sample_rate)
    }
}

fn normalize_language(language: &str) -> String {
    language
        .trim()
        .split(['-', '_'])
        .next()
        .unwrap_or(language)
        .to_ascii_lowercase()
}

fn parse_speaker_index(voice: &str, num_baked_speakers: usize) -> Result<usize, TtsError> {
    let speaker_index = voice
        .trim()
        .parse::<usize>()
        .map_err(|_| TtsError::InitError(format!("Invalid MagPie speaker id '{}'", voice)))?;
    if speaker_index >= num_baked_speakers {
        return Err(TtsError::InitError(format!(
            "MagPie speaker index {} out of range ({} speakers)",
            speaker_index, num_baked_speakers
        )));
    }
    Ok(speaker_index)
}
