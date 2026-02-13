// Jack Desktop - Audio Capture and Playback
// Uses cpal for capture and rodio for playback

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, Stream, StreamConfig};
use rodio::buffer::SamplesBuffer;
use rodio::{OutputStream, Sink};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender, SyncSender, TrySendError};
use std::thread;
use std::time::Instant;

// Global metrics for audio drops (shared with voice pipeline)
static AUDIO_DROP_COUNT: AtomicU64 = AtomicU64::new(0);
static LAST_AUDIO_DROP_LOG_MS: AtomicU64 = AtomicU64::new(0);

/// Get current audio drop count
pub fn audio_drops() -> u64 {
    AUDIO_DROP_COUNT.load(Ordering::Relaxed)
}

pub const SAMPLE_RATE: u32 = 16000; // 16kHz for speech models
pub const CHANNELS: u16 = 1; // Mono

/// Audio capture stream that sends samples to a channel
pub struct AudioCapture {
    _stream: Stream,
    stop_tx: Sender<()>,
    pub device_sample_rate: u32,
}

impl AudioCapture {
    /// Start capturing audio with backpressure (bounded channel)
    /// Drops samples if consumer can't keep up, preventing buffer bloat
    pub fn start_with_backpressure(sample_tx: SyncSender<Vec<f32>>) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(AudioError::NoInputDevice)?;

        log::info!("Using input device: {}", device.name().unwrap_or_default());

        let config = find_suitable_config(&device)?;
        let actual_sample_rate = config.sample_rate.0;
        let channels = config.channels as usize;
        log::info!(
            "Audio config: {} channels, {}Hz (target: mono {}Hz)",
            channels,
            actual_sample_rate,
            SAMPLE_RATE
        );

        let (stop_tx, stop_rx) = channel::<()>();

        static CALLBACK_COUNT: AtomicUsize = AtomicUsize::new(0);

        let stream = device
            .build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if stop_rx.try_recv().is_ok() {
                        return;
                    }

                    let count = CALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
                    if count % 1000 == 0 {
                        // Reduced from 100 to prevent log spam
                        let max_val = data.iter().cloned().fold(0.0f32, f32::max);
                        log::debug!(
                            "Audio callback #{}: {} samples, max={:.4}",
                            count,
                            data.len(),
                            max_val
                        );
                    }

                    // Convert to mono
                    let mono_samples: Vec<f32> = if channels > 1 {
                        data.chunks(channels)
                            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                            .collect()
                    } else {
                        data.to_vec()
                    };

                    // Send raw mono samples - resampling moved to processing thread to avoid xruns
                    let samples = mono_samples;

                    // Bounded send with backpressure - drop if full
                    match sample_tx.try_send(samples) {
                        Ok(()) => {}
                        Err(TrySendError::Full(_)) => {
                            AUDIO_DROP_COUNT.fetch_add(1, Ordering::Relaxed);
                            // Rate-limit drop logs to avoid log storms when consumer stalls.
                            const DROP_LOG_INTERVAL_MS: u64 = 2000;
                            let now_ms = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_millis() as u64)
                                .unwrap_or(0);
                            let last_ms = LAST_AUDIO_DROP_LOG_MS.load(Ordering::Relaxed);
                            if now_ms.saturating_sub(last_ms) >= DROP_LOG_INTERVAL_MS {
                                LAST_AUDIO_DROP_LOG_MS.store(now_ms, Ordering::Relaxed);
                                log::warn!(
                                    "Audio buffer full, dropping samples (total drops: {})",
                                    AUDIO_DROP_COUNT.load(Ordering::Relaxed)
                                );
                            }
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            log::debug!("Audio channel disconnected");
                        }
                    }
                },
                |err| {
                    log::error!("Audio capture error: {}", err);
                },
                None,
            )
            .map_err(|e| AudioError::StreamError(e.to_string()))?;

        stream
            .play()
            .map_err(|e| AudioError::StreamError(e.to_string()))?;

        Ok(Self {
            _stream: stream,
            stop_tx,
            device_sample_rate: actual_sample_rate,
        })
    }

    /// Start capturing audio from the default input device (unbounded)
    pub fn start(sample_tx: Sender<Vec<f32>>) -> Result<Self, AudioError> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or(AudioError::NoInputDevice)?;

        log::info!("Using input device: {}", device.name().unwrap_or_default());

        let config = find_suitable_config(&device)?;
        let actual_sample_rate = config.sample_rate.0;
        let channels = config.channels as usize;
        log::info!(
            "Audio config: {} channels, {}Hz (target: mono {}Hz)",
            channels,
            actual_sample_rate,
            SAMPLE_RATE
        );

        let (stop_tx, stop_rx) = channel::<()>();

        // Counter for periodic logging
        static CALLBACK_COUNT: AtomicUsize = AtomicUsize::new(0);

        // Build the input stream
        let stream = device
            .build_input_stream(
                &config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    // Check if we should stop
                    if stop_rx.try_recv().is_ok() {
                        return;
                    }

                    // Periodic logging (reduced frequency to prevent spam)
                    let count = CALLBACK_COUNT.fetch_add(1, Ordering::Relaxed);
                    if count % 1000 == 0 {
                        // Reduced from 100 to prevent log spam
                        let max_val = data.iter().cloned().fold(0.0f32, f32::max);
                        log::debug!(
                            "Audio callback #{}: {} samples, {} channels, max={:.4}",
                            count,
                            data.len(),
                            channels,
                            max_val
                        );
                    }

                    // Convert stereo/multi-channel to mono by averaging channels
                    let mono_samples: Vec<f32> = if channels > 1 {
                        data.chunks(channels)
                            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                            .collect()
                    } else {
                        data.to_vec()
                    };

                    // Resample if needed and send
                    let samples = if actual_sample_rate != SAMPLE_RATE {
                        resample(&mono_samples, actual_sample_rate, SAMPLE_RATE)
                    } else {
                        mono_samples
                    };

                    let _ = sample_tx.send(samples);
                },
                |err| {
                    log::error!("Audio capture error: {}", err);
                },
                None,
            )
            .map_err(|e| AudioError::StreamError(e.to_string()))?;

        stream
            .play()
            .map_err(|e| AudioError::StreamError(e.to_string()))?;

        Ok(Self {
            _stream: stream,
            stop_tx,
            device_sample_rate: actual_sample_rate,
        })
    }

    /// Stop the capture stream
    pub fn stop(&self) {
        let _ = self.stop_tx.send(());
    }
}

/// Audio player using rodio for non-blocking playback
pub struct AudioPlayer {
    _stream: OutputStream,
    handle: rodio::OutputStreamHandle,
    sink: Sink,
}

impl AudioPlayer {
    /// Create a new audio player
    pub fn new() -> Result<Self, AudioError> {
        let (stream, handle) =
            OutputStream::try_default().map_err(|_e| AudioError::NoOutputDevice)?;

        let sink = Sink::try_new(&handle).map_err(|e| AudioError::StreamError(e.to_string()))?;

        Ok(Self {
            _stream: stream,
            handle,
            sink,
        })
    }

    /// Play audio samples (non-blocking)
    pub fn play(&self, samples: Vec<f32>, sample_rate: u32) {
        let source = SamplesBuffer::new(1, sample_rate, samples);
        self.sink.append(source);
    }

    /// Stop playback and clear queue.
    ///
    /// Uses clear() to empty the queue without destroying the sink, then
    /// skip_one() to stop the currently-playing source. This avoids the race
    /// condition where destroying + recreating the Sink could leave a window
    /// where play() hits a dead sink.
    pub fn stop(&mut self) {
        self.sink.clear();
        // clear() removes queued sources but the currently-playing source
        // may still be active. skip_one() drops it.
        self.sink.skip_one();
        // If we were paused (e.g. two-stage barge-in), ensure the sink is
        // unpaused so subsequent `play()` calls actually produce audio.
        self.sink.play();
    }

    /// Check if currently playing
    pub fn is_playing(&self) -> bool {
        !self.sink.empty()
    }

    /// Wait for playback to finish
    pub fn wait(&self) {
        self.sink.sleep_until_end();
    }

    /// Pause playback
    pub fn pause(&self) {
        self.sink.pause();
    }

    /// Resume playback
    pub fn resume(&self) {
        self.sink.play();
    }

    /// Set volume (0.0 to 1.0)
    pub fn set_volume(&self, volume: f32) {
        self.sink.set_volume(volume.clamp(0.0, 1.0));
    }
}

/// Legacy AudioStream for backward compatibility
pub struct AudioStream {
    capture: AudioCapture,
}

impl AudioStream {
    /// Start capturing audio with a callback
    pub fn start_capture<F>(mut callback: F) -> Result<Self, AudioError>
    where
        F: FnMut(&[f32]) + Send + 'static,
    {
        let (tx, rx) = channel::<Vec<f32>>();

        // Spawn a thread to forward samples to the callback
        thread::spawn(move || {
            while let Ok(samples) = rx.recv() {
                callback(&samples);
            }
        });

        let capture = AudioCapture::start(tx)?;
        Ok(Self { capture })
    }

    /// Stop the audio stream
    pub fn stop(&self) -> Result<(), AudioError> {
        self.capture.stop();
        Ok(())
    }
}

/// Find a suitable audio configuration for speech
fn find_suitable_config(device: &Device) -> Result<StreamConfig, AudioError> {
    let supported_configs = device
        .supported_input_configs()
        .map_err(|e| AudioError::ConfigError(e.to_string()))?;

    // Try to find a config close to 16kHz mono
    for config_range in supported_configs {
        if config_range.channels() == 1 && config_range.sample_format() == SampleFormat::F32 {
            let min_rate = config_range.min_sample_rate().0;
            let max_rate = config_range.max_sample_rate().0;

            if min_rate <= SAMPLE_RATE && SAMPLE_RATE <= max_rate {
                return Ok(StreamConfig {
                    channels: 1,
                    sample_rate: cpal::SampleRate(SAMPLE_RATE),
                    buffer_size: cpal::BufferSize::Default,
                });
            }
        }
    }

    // Fallback: use default config and resample later
    let default_config = device
        .default_input_config()
        .map_err(|e| AudioError::ConfigError(e.to_string()))?;

    Ok(default_config.into())
}

/// Resample audio from source rate to target rate (one-shot, allocates each call).
/// Prefer `PersistentResampler` on hot paths.
pub fn resample(samples: &[f32], source_rate: u32, target_rate: u32) -> Vec<f32> {
    if source_rate == target_rate || samples.is_empty() {
        return samples.to_vec();
    }

    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    match SincFixedIn::<f32>::new(
        target_rate as f64 / source_rate as f64,
        2.0,
        params,
        samples.len(),
        1,
    ) {
        Ok(mut resampler) => {
            let input = vec![samples.to_vec()];
            match resampler.process(&input, None) {
                Ok(output) => output.into_iter().next().unwrap_or_default(),
                Err(e) => {
                    log::warn!("Resampling failed: {:?}", e);
                    samples.to_vec()
                }
            }
        }
        Err(e) => {
            log::warn!("Resampler init failed: {:?}", e);
            samples.to_vec()
        }
    }
}

/// Resampler that persists across calls to avoid re-allocating sinc tables.
/// Expects fixed-size input chunks (e.g., audio callback frames).
pub struct PersistentResampler {
    resampler: rubato::SincFixedIn<f32>,
    chunk_size: usize,
    buffer: Vec<f32>,
}

impl PersistentResampler {
    /// Create a resampler for a given source→target rate and expected chunk size.
    pub fn new(source_rate: u32, target_rate: u32, chunk_size: usize) -> Result<Self, String> {
        use rubato::{
            SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
        };

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };

        let resampler = SincFixedIn::<f32>::new(
            target_rate as f64 / source_rate as f64,
            2.0,
            params,
            chunk_size,
            1,
        )
        .map_err(|e| format!("PersistentResampler init failed: {:?}", e))?;

        Ok(Self {
            resampler,
            chunk_size,
            buffer: Vec::new(),
        })
    }

    /// Resample a chunk of audio. Buffers input internally to match expected chunk size.
    pub fn process(&mut self, samples: &[f32]) -> Vec<f32> {
        use rubato::Resampler;

        self.buffer.extend_from_slice(samples);

        let mut output = Vec::new();
        while self.buffer.len() >= self.chunk_size {
            let chunk: Vec<f32> = self.buffer.drain(..self.chunk_size).collect();
            let input = vec![chunk];
            match self.resampler.process(&input, None) {
                Ok(result) => {
                    if let Some(channel) = result.into_iter().next() {
                        output.extend(channel);
                    }
                }
                Err(e) => {
                    log::warn!("PersistentResampler process error: {:?}", e);
                }
            }
        }
        output
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("No input device found")]
    NoInputDevice,
    #[error("No output device found")]
    NoOutputDevice,
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Config error: {0}")]
    ConfigError(String),
}

// ============================================
// Audio Health Monitor
// ============================================

/// Health level for audio input
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioHealthLevel {
    Good,
    Warning,
    Error,
}

impl AudioHealthLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            AudioHealthLevel::Good => "good",
            AudioHealthLevel::Warning => "warning",
            AudioHealthLevel::Error => "error",
        }
    }
}

/// Audio health status with diagnostic info
#[derive(Debug, Clone)]
pub struct AudioHealthStatus {
    pub level: AudioHealthLevel,
    pub message: String,
    pub suggestion: Option<String>,
    pub avg_rms: f32,
    pub silence_duration_ms: u64,
}

impl AudioHealthStatus {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "level": self.level.as_str(),
            "message": self.message,
            "suggestion": self.suggestion,
            "avgRms": self.avg_rms,
            "silenceDurationMs": self.silence_duration_ms,
        })
    }
}

/// Monitors audio input quality and detects issues
pub struct AudioHealthMonitor {
    rms_history: VecDeque<f32>,
    last_speech_time: Option<Instant>,
    created_at: Instant,
    sample_count: usize,
    dropout_count: usize,
    last_sample_time: Option<Instant>,

    // Thresholds
    silence_threshold_rms: f32,
    warning_silence_ms: u64,
    error_silence_ms: u64,
}

impl AudioHealthMonitor {
    pub fn new() -> Self {
        Self {
            rms_history: VecDeque::with_capacity(100),
            last_speech_time: None,
            created_at: Instant::now(),
            sample_count: 0,
            dropout_count: 0,
            last_sample_time: None,
            silence_threshold_rms: 0.01,
            warning_silence_ms: 10_000, // 10 seconds
            error_silence_ms: 30_000,   // 30 seconds
        }
    }

    /// Process an audio chunk and update health metrics
    pub fn process(&mut self, samples: &[f32]) {
        self.sample_count += 1;

        // Check for dropouts (gaps > 200ms between chunks)
        if let Some(last_time) = self.last_sample_time {
            if last_time.elapsed().as_millis() > 200 {
                self.dropout_count += 1;
            }
        }
        self.last_sample_time = Some(Instant::now());

        // Calculate RMS energy
        let rms = if !samples.is_empty() {
            (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
        } else {
            0.0
        };

        // Update history
        self.rms_history.push_back(rms);
        if self.rms_history.len() > 100 {
            self.rms_history.pop_front();
        }

        // Track speech activity
        if rms > self.silence_threshold_rms {
            self.last_speech_time = Some(Instant::now());
        }
    }

    /// Get current audio health status
    pub fn get_status(&self) -> AudioHealthStatus {
        // Calculate average RMS from history
        let avg_rms = if !self.rms_history.is_empty() {
            self.rms_history.iter().sum::<f32>() / self.rms_history.len() as f32
        } else {
            0.0
        };

        // Calculate silence duration
        let silence_duration = self
            .last_speech_time
            .map(|t| t.elapsed())
            .unwrap_or_else(|| self.created_at.elapsed());
        let silence_ms = silence_duration.as_millis() as u64;

        // Skip health checks during first 2 seconds (warm-up period)
        if self.created_at.elapsed().as_secs() < 2 {
            return AudioHealthStatus {
                level: AudioHealthLevel::Good,
                message: "Warming up...".to_string(),
                suggestion: None,
                avg_rms,
                silence_duration_ms: silence_ms,
            };
        }

        // Determine health level
        if silence_ms > self.error_silence_ms {
            return AudioHealthStatus {
                level: AudioHealthLevel::Error,
                message: format!("No audio detected for {}s", silence_ms / 1000),
                suggestion: Some("Check if microphone is connected and not muted. If using Bluetooth, check battery level.".to_string()),
                avg_rms,
                silence_duration_ms: silence_ms,
            };
        }

        if silence_ms > self.warning_silence_ms {
            return AudioHealthStatus {
                level: AudioHealthLevel::Warning,
                message: format!("Very quiet - no speech for {}s", silence_ms / 1000),
                suggestion: Some(
                    "Speak closer to the microphone or check audio input settings.".to_string(),
                ),
                avg_rms,
                silence_duration_ms: silence_ms,
            };
        }

        if avg_rms < 0.001 && self.sample_count > 50 {
            return AudioHealthStatus {
                level: AudioHealthLevel::Warning,
                message: "Extremely low audio level".to_string(),
                suggestion: Some("Microphone may be muted or disconnected.".to_string()),
                avg_rms,
                silence_duration_ms: silence_ms,
            };
        }

        if self.dropout_count > 5 {
            return AudioHealthStatus {
                level: AudioHealthLevel::Warning,
                message: format!("Audio dropouts detected ({})", self.dropout_count),
                suggestion: Some(
                    "Audio stream may be unstable. Try reconnecting your microphone.".to_string(),
                ),
                avg_rms,
                silence_duration_ms: silence_ms,
            };
        }

        AudioHealthStatus {
            level: AudioHealthLevel::Good,
            message: "Audio input OK".to_string(),
            suggestion: None,
            avg_rms,
            silence_duration_ms: silence_ms,
        }
    }

    /// Reset the monitor (call when re-starting listening)
    pub fn reset(&mut self) {
        self.rms_history.clear();
        self.last_speech_time = None;
        self.created_at = Instant::now();
        self.sample_count = 0;
        self.dropout_count = 0;
        self.last_sample_time = None;
    }

    /// Check if we should emit a health event (rate-limited)
    pub fn should_emit_event(&self) -> bool {
        // Emit every 50 samples (~3 seconds at typical chunk rate)
        self.sample_count % 50 == 0
    }
}

// ============================================
// Audio Diagnostics
// ============================================

/// Get diagnostics info about audio devices
pub fn get_audio_diagnostics() -> serde_json::Value {
    let host = cpal::default_host();

    // Get input devices
    let input_devices: Vec<String> = host
        .input_devices()
        .map(|devices| devices.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default();

    // Get default input device
    let default_input = host.default_input_device().and_then(|d| d.name().ok());

    // Get output devices
    let output_devices: Vec<String> = host
        .output_devices()
        .map(|devices| devices.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default();

    // Get default output device
    let default_output = host.default_output_device().and_then(|d| d.name().ok());

    // Try to get default input config
    let input_config = host
        .default_input_device()
        .and_then(|d| d.default_input_config().ok())
        .map(|c| {
            format!(
                "{} channels, {}Hz, {:?}",
                c.channels(),
                c.sample_rate().0,
                c.sample_format()
            )
        });

    serde_json::json!({
        "host": host.id().name(),
        "input_devices": input_devices,
        "default_input": default_input,
        "input_config": input_config,
        "output_devices": output_devices,
        "default_output": default_output,
        "has_input": default_input.is_some(),
        "has_output": default_output.is_some(),
    })
}
