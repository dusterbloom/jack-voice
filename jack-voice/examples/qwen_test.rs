//! Qwen3-TTS Manual Testing CLI
//!
//! Commands:
//!   download     Download Qwen model
//!   synthesize   Synthesize text to audio file
//!   benchmark    Run performance benchmark with timing breakdown
//!   voices       List available voices
//!   info         Show model info and system status
//!
//! Examples:
//!   # Download models
//!   cargo run --example qwen_test -- download --size lite
//!   cargo run --example qwen_test -- download --size large
//!
//!   # Synthesize with preset voice (Lite)
//!   cargo run --example qwen_test -- synthesize --engine qwen \
//!     --text "Hello world" --voice ryan --output hello.wav
//!
//!   # Synthesize Italian text
//!   cargo run --example qwen_test -- synthesize --engine qwen \
//!     --text "La corazzata potioschi e' una cagata pazzesca" --output italian.wav
//!
//!   # Synthesize with voice cloning (Large only)
//!   cargo run --example qwen_test -- synthesize --engine qwen-large \
//!     --text "La corazzata potioschi e' una cagata pazzesca" \
//!     --ref-audio test_fixtures/carriera_fantozzi.wav \
//!     --output cloned.wav
//!
//!   # Benchmark performance with timing breakdown
//!   cargo run --example qwen_test --features cuda -- benchmark --engine qwen --iterations 3

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use jack_voice::{
    models, qwen_tts::QwenModelSize, qwen_tts::QwenTts, NoopProgress, TextToSpeech, TtsEngine,
};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "qwen_test", about = "Qwen3-TTS Testing CLI", version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Download Qwen model
    Download {
        /// Model size: lite (0.6B) or large (1.7B)
        #[arg(short, long, default_value = "lite")]
        size: String,
    },

    /// Synthesize text to audio file
    Synthesize {
        /// TTS engine: qwen or qwen-large
        #[arg(short, long, default_value = "qwen")]
        engine: String,

        /// Text to synthesize
        #[arg(short, long)]
        text: String,

        /// Voice ID for preset (qwen only): ryan, serena, vivian, etc.
        #[arg(short, long, default_value = "ryan")]
        voice: String,

        /// Reference audio for voice cloning (qwen-large only)
        #[arg(short, long)]
        ref_audio: Option<PathBuf>,

        /// Transcript for reference audio (optional, improves cloning)
        #[arg(long)]
        ref_transcript: Option<String>,

        /// Output WAV file
        #[arg(short, long, default_value = "output.wav")]
        output: PathBuf,
    },

    /// Run performance benchmark
    Benchmark {
        /// TTS engine: qwen or qwen-large
        #[arg(short, long, default_value = "qwen")]
        engine: String,

        /// Italian text for benchmark
        #[arg(long, default_value = "La corazzata potioschi e' una cagata pazzesca")]
        text_it: String,

        /// English text for benchmark
        #[arg(
            long,
            default_value = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."
        )]
        text_en: String,

        /// Number of iterations
        #[arg(short, long, default_value = "3")]
        iterations: usize,
    },

    /// List available voices
    Voices,

    /// Show model info and system status
    Info,
}

fn main() -> Result<()> {
    // Simple logging init - prints to stderr
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Download { size } => {
            let model_size = parse_size(&size)?;
            println!("Downloading Qwen {:?} model...", model_size);

            let start = Instant::now();
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(models::ensure_qwen_model(model_size, &NoopProgress))?;

            println!(
                "✓ Download complete in {:.1}s",
                start.elapsed().as_secs_f32()
            );
            println!("  Model dir: {:?}", models::qwen_model_dir(model_size));
        }

        Commands::Synthesize {
            engine,
            text,
            voice,
            ref_audio,
            ref_transcript,
            output,
        } => {
            let engine_type = parse_engine(&engine)?;

            let size = match engine_type {
                TtsEngine::Qwen => QwenModelSize::Lite,
                TtsEngine::QwenLarge => QwenModelSize::Large,
                _ => anyhow::bail!("Only qwen and qwen-large engines supported"),
            };

            if !models::qwen_model_ready(size) {
                println!("Model not found, downloading...");
                let rt = tokio::runtime::Runtime::new()?;
                rt.block_on(models::ensure_qwen_model(size, &NoopProgress))?;
            }

            let mut tts = TextToSpeech::with_engine(engine_type.clone())
                .context("Failed to initialize TTS engine")?;

            if let Some(ref_audio_path) = ref_audio {
                if engine_type != TtsEngine::QwenLarge {
                    anyhow::bail!("Voice cloning requires --engine qwen-large");
                }
                if !ref_audio_path.exists() {
                    anyhow::bail!("Reference audio not found: {:?}", ref_audio_path);
                }
                tts.set_voice_clone_reference(ref_audio_path.clone(), ref_transcript)
                    .context("Failed to set voice clone reference")?;
                println!("Using voice clone from: {:?}", ref_audio_path);
            } else if engine_type == TtsEngine::Qwen {
                tts.set_speaker(&voice).context("Failed to set voice")?;
                println!("Using voice: {}", voice);
            }

            println!("Synthesizing: \"{}\"", text);
            let start = Instant::now();

            let audio = tts.synthesize(&text).context("Synthesis failed")?;

            let elapsed = start.elapsed();
            let duration_secs = audio.samples.len() as f32 / audio.sample_rate as f32;
            let rtf = if duration_secs > 0.0 {
                elapsed.as_secs_f32() / duration_secs
            } else {
                0.0
            };

            save_wav(&output, &audio.samples, audio.sample_rate).context("Failed to save WAV")?;

            println!("✓ Synthesis complete");
            println!("  Output: {:?}", output);
            println!("  Audio duration: {:.2}s", duration_secs);
            println!("  Synthesis time: {:.2}s", elapsed.as_secs_f32());
            println!("  RTF (Real-Time Factor): {:.2}x", rtf);
            if rtf < 1.0 {
                println!("  Status: ✓ FASTER than real-time");
            } else {
                println!("  Status: ⚠ SLOWER than real-time");
            }
        }

        Commands::Benchmark {
            engine,
            text_it,
            text_en,
            iterations,
        } => {
            let engine_type = parse_engine(&engine)?;
            let size = match engine_type {
                TtsEngine::Qwen => QwenModelSize::Lite,
                TtsEngine::QwenLarge => QwenModelSize::Large,
                _ => anyhow::bail!("Only qwen and qwen-large engines supported"),
            };

            if !models::qwen_model_ready(size) {
                println!("Model not found, downloading...");
                let rt = tokio::runtime::Runtime::new()?;
                rt.block_on(models::ensure_qwen_model(size, &NoopProgress))?;
            }

            let model_dir = models::qwen_model_dir(size);
            let tts = QwenTts::new(&model_dir, size).context("Failed to initialize QwenTts")?;

            let texts = vec![("Italian", text_it), ("English", text_en)];

            println!(
                "Benchmarking {:?} engine with {} iterations...",
                engine_type, iterations
            );
            println!();

            for (lang, text) in &texts {
                println!("{}: \"{}\"", lang, text);

                let mut times: Vec<f64> = Vec::new();
                let mut durations: Vec<f64> = Vec::new();
                let mut prefill_times = Vec::new();
                let mut gen_times = Vec::new();
                let mut decode_times = Vec::new();

                for i in 0..iterations {
                    let (audio, timing) = tts
                        .synthesize_with_timing(text)
                        .context("Synthesis failed")?;

                    let total_ms = timing.prefill_ms + timing.generation_ms + timing.decode_ms;
                    let audio_dur = audio.samples.len() as f64 / audio.sample_rate as f64;

                    times.push(total_ms / 1000.0);
                    durations.push(audio_dur);
                    prefill_times.push(timing.prefill_ms);
                    gen_times.push(timing.generation_ms);
                    decode_times.push(timing.decode_ms);

                    let rtf = if audio_dur > 0.0 {
                        (total_ms / 1000.0) / audio_dur
                    } else {
                        0.0
                    };

                    println!(
                        "  Iter {}: {:.2}s total → {:.2}s audio (RTF: {:.2}x)",
                        i + 1,
                        total_ms / 1000.0,
                        audio_dur,
                        rtf
                    );
                    println!(
                        "         Prefill: {:.0}ms | Gen: {:.0}ms ({:.0} frames) | Decode: {:.0}ms",
                        timing.prefill_ms,
                        timing.generation_ms,
                        timing.generation_frames,
                        timing.decode_ms
                    );
                }

                let avg_time: f64 = times.iter().sum::<f64>() / times.len() as f64;
                let avg_dur: f64 = durations.iter().sum::<f64>() / durations.len() as f64;
                let avg_prefill: f64 =
                    prefill_times.iter().sum::<f64>() / prefill_times.len() as f64;
                let avg_gen: f64 = gen_times.iter().sum::<f64>() / gen_times.len() as f64;
                let avg_decode: f64 = decode_times.iter().sum::<f64>() / decode_times.len() as f64;
                let avg_rtf = if avg_dur > 0.0 {
                    avg_time / avg_dur
                } else {
                    0.0
                };

                println!(
                    "  Average: {:.2}s total → {:.2}s audio (RTF: {:.2}x)",
                    avg_time, avg_dur, avg_rtf
                );
                println!(
                    "           Prefill: {:.0}ms | Gen: {:.0}ms | Decode: {:.0}ms",
                    avg_prefill, avg_gen, avg_decode
                );
                println!();
            }
        }

        Commands::Voices => {
            println!("Available Qwen preset voices (Lite model):");
            println!();
            for (id, name) in jack_voice::QWEN_LITE_VOICES {
                println!("  {:12} {}", id, name);
            }
            println!();
            println!("Note: Qwen Large uses voice cloning, not preset voices.");
        }

        Commands::Info => {
            println!("Qwen3-TTS System Info");
            println!("=====================");
            println!();

            println!("Models:");
            println!(
                "  Lite (0.6B):  {} - {:?}",
                if models::qwen_model_ready(QwenModelSize::Lite) {
                    "✓ ready"
                } else {
                    "✗ not downloaded"
                },
                models::qwen_model_dir(QwenModelSize::Lite)
            );
            println!(
                "  Large (1.7B): {} - {:?}",
                if models::qwen_model_ready(QwenModelSize::Large) {
                    "✓ ready"
                } else {
                    "✗ not downloaded"
                },
                models::qwen_model_dir(QwenModelSize::Large)
            );
            println!();

            println!("GPU Support:");
            let can_run = TextToSpeech::can_run_qwen();
            if can_run {
                println!("  ✓ GPU available (CUDA)");
            } else {
                println!("  ✗ No GPU detected (CPU too slow for real-time)");
            }
            println!();

            println!(
                "Preset Voices: {} available",
                jack_voice::QWEN_LITE_VOICES.len()
            );
            println!();

            println!("Model Sizes:");
            println!(
                "  Lite:  ~{} MB download, ~{} MB VRAM",
                models::QWEN_LITE_SIZE_MB,
                models::QWEN_LITE_SIZE_MB / 2
            );
            println!(
                "  Large: ~{} MB download, ~{} MB VRAM",
                models::QWEN_LARGE_SIZE_MB,
                models::QWEN_LARGE_SIZE_MB
            );
        }
    }

    Ok(())
}

fn parse_size(s: &str) -> Result<QwenModelSize> {
    match s.to_lowercase().as_str() {
        "lite" | "0.6b" => Ok(QwenModelSize::Lite),
        "large" | "1.7b" => Ok(QwenModelSize::Large),
        _ => anyhow::bail!("Invalid size '{}'. Use: lite, large", s),
    }
}

fn parse_engine(s: &str) -> Result<TtsEngine> {
    match s.to_lowercase().as_str() {
        "qwen" | "qwen-lite" => Ok(TtsEngine::Qwen),
        "qwen-large" | "qwenlarge" => Ok(TtsEngine::QwenLarge),
        _ => anyhow::bail!("Invalid engine '{}'. Use: qwen, qwen-large", s),
    }
}

fn save_wav(path: &PathBuf, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;
    for sample in samples {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    Ok(())
}
