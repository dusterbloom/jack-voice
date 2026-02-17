mod audio;
mod protocol;

use std::env;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use audio::{
    decode_audio_to_f32, encode_f32le_to_base64, AudioPayload, DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE_HZ,
};
use jack_voice::models::{self, ModelBundle, ModelError, ModelProgressCallback, MODEL_BUNDLES};
use jack_voice::stt::SttBackend;
use jack_voice::{
    SpeechToText, SttError, SttMode, TextToSpeech, TtsEngine, TtsError, VadError,
    VoiceActivityDetector,
};
use protocol::{
    ErrorCode, EventEnvelope, RequestEnvelope, ResponseEnvelope, RpcError, RpcMethod,
    MAX_REQUEST_BYTES, PROTOCOL_VERSION,
};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};

const UNKNOWN_REQUEST_ID: &str = "_unknown";
const MAX_TIMEOUT_MS: u64 = 5 * 60 * 1000;

#[derive(Default)]
struct BridgeState {
    vad: Option<VoiceActivityDetector>,
    stt: Option<SpeechToText>,
    stt_key: Option<SttCacheKey>,
    tts: Option<TextToSpeech>,
    tts_engine: Option<CachedTtsEngine>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SttCacheKey {
    mode: SttMode,
    language: Option<String>,
    tts_voice: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CachedTtsEngine {
    Pocket,
    Supertonic,
    Kokoro,
}

impl CachedTtsEngine {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pocket => "pocket",
            Self::Supertonic => "supertonic",
            Self::Kokoro => "kokoro",
        }
    }

    fn as_jack_voice_engine(self) -> TtsEngine {
        match self {
            Self::Pocket => TtsEngine::Pocket,
            Self::Supertonic => TtsEngine::Supertonic,
            Self::Kokoro => TtsEngine::Kokoro,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RequestedTtsEngine {
    Auto,
    Pocket,
    Supertonic,
    Kokoro,
}

struct MethodOutcome {
    result: Value,
    should_shutdown: bool,
}

struct StderrProgress;

impl ModelProgressCallback for StderrProgress {
    fn on_download_start(&self, model: &str, size_mb: u64) {
        eprintln!("[models] download_start model={model} size_mb={size_mb}");
    }

    fn on_download_progress(&self, model: &str, progress_percent: u32, downloaded_mb: u64) {
        eprintln!(
            "[models] download_progress model={model} percent={progress_percent} downloaded_mb={downloaded_mb}"
        );
    }

    fn on_download_complete(&self, model: &str) {
        eprintln!("[models] download_complete model={model}");
    }

    fn on_extracting(&self, model: &str) {
        eprintln!("[models] extracting model={model}");
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct RuntimeHelloParams {
    models_dir: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct ModelsEnsureParams {
    dry_run: bool,
}

#[derive(Debug, Deserialize)]
struct VadDetectParams {
    #[serde(flatten)]
    audio: AudioPayload,
    #[serde(default)]
    flush: bool,
}

#[derive(Debug, Deserialize)]
struct SttTranscribeParams {
    #[serde(flatten)]
    audio: AudioPayload,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    language: Option<String>,
    #[serde(default)]
    tts_voice: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TtsSynthesizeParams {
    text: String,
    #[serde(default)]
    engine: Option<String>,
    #[serde(default)]
    voice: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
    #[serde(default)]
    format: Option<String>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("[bridge] fatal error: {err}");
        std::process::exit(1);
    }
}

fn run() -> io::Result<()> {
    configure_models_dir_from_env();

    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    let mut state = BridgeState::default();

    for line_result in stdin.lock().lines() {
        let line = match line_result {
            Ok(line) => line,
            Err(err) => {
                eprintln!("[bridge] stdin read error: {err}");
                break;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let fallback_id =
            extract_request_id(&line).unwrap_or_else(|| UNKNOWN_REQUEST_ID.to_string());
        let started = Instant::now();

        let (response, should_shutdown) = if line.len() > MAX_REQUEST_BYTES {
            (
                ResponseEnvelope::err(
                    fallback_id,
                    RpcError::new(
                        ErrorCode::PayloadTooLarge,
                        format!(
                            "Request exceeds max size ({} > {})",
                            line.len(),
                            MAX_REQUEST_BYTES
                        ),
                    ),
                ),
                false,
            )
        } else {
            handle_line(&line, &mut state, &mut stdout)
        };

        write_response(&mut stdout, &response)?;

        let latency_ms = started.elapsed().as_millis();
        eprintln!(
            "[bridge] id={} ok={} latency_ms={latency_ms}",
            response.id, response.ok
        );

        if should_shutdown {
            break;
        }
    }

    Ok(())
}

fn handle_line(
    line: &str,
    state: &mut BridgeState,
    stdout: &mut dyn Write,
) -> (ResponseEnvelope, bool) {
    let json_value: Value = match serde_json::from_str(line) {
        Ok(value) => value,
        Err(err) => {
            let id = extract_request_id(line).unwrap_or_else(|| UNKNOWN_REQUEST_ID.to_string());
            return (
                ResponseEnvelope::err(
                    id,
                    RpcError::new(
                        ErrorCode::ParseError,
                        format!("Invalid JSON request: {err}"),
                    ),
                ),
                false,
            );
        }
    };

    let request_id = json_value
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.trim().is_empty())
        .unwrap_or(UNKNOWN_REQUEST_ID)
        .to_string();

    let request: RequestEnvelope = match serde_json::from_value(json_value) {
        Ok(request) => request,
        Err(err) => {
            return (
                ResponseEnvelope::err(
                    request_id,
                    RpcError::new(
                        ErrorCode::InvalidRequest,
                        format!("Invalid request envelope: {err}"),
                    ),
                ),
                false,
            );
        }
    };

    if request.message_type != "request" {
        return (
            ResponseEnvelope::err(
                request.id,
                RpcError::new(
                    ErrorCode::InvalidRequest,
                    format!(
                        "Unsupported message type '{}' (expected 'request')",
                        request.message_type
                    ),
                ),
            ),
            false,
        );
    }

    if request.id.trim().is_empty() {
        return (
            ResponseEnvelope::err(
                request.id,
                RpcError::new(ErrorCode::InvalidRequest, "Request id must not be empty"),
            ),
            false,
        );
    }

    let method = match RpcMethod::from_str(&request.method) {
        Ok(method) => method,
        Err(err) => return (ResponseEnvelope::err(request.id, err), false),
    };

    if let Some(timeout_ms) = request.timeout_ms {
        if timeout_ms == 0 {
            return (
                ResponseEnvelope::err(
                    request.id,
                    RpcError::new(
                        ErrorCode::InvalidParams,
                        "timeout_ms must be greater than 0",
                    ),
                ),
                false,
            );
        }
    }

    let started = Instant::now();
    let outcome = dispatch_request(state, method, request.params, &request.id, stdout);
    if let Some(timeout_ms) = request.timeout_ms {
        let bounded_ms = timeout_ms.min(MAX_TIMEOUT_MS);
        if started.elapsed().as_millis() > bounded_ms as u128 {
            return (
                ResponseEnvelope::err(
                    request.id,
                    RpcError::new(
                        ErrorCode::OperationTimeout,
                        format!("Request timed out after {}ms", bounded_ms),
                    ),
                ),
                false,
            );
        }
    }

    match outcome {
        Ok(outcome) => (
            ResponseEnvelope::ok(request.id, outcome.result),
            outcome.should_shutdown,
        ),
        Err(err) => (ResponseEnvelope::err(request.id, err), false),
    }
}

fn dispatch_request(
    state: &mut BridgeState,
    method: RpcMethod,
    params: Value,
    request_id: &str,
    stdout: &mut dyn Write,
) -> Result<MethodOutcome, RpcError> {
    match method {
        RpcMethod::RuntimeHello => {
            let params: RuntimeHelloParams = parse_params(params)?;
            let result = handle_runtime_hello(params)?;
            Ok(MethodOutcome {
                result,
                should_shutdown: false,
            })
        }
        RpcMethod::ModelsStatus => {
            let result = build_models_status()?;
            Ok(MethodOutcome {
                result,
                should_shutdown: false,
            })
        }
        RpcMethod::ModelsEnsure => {
            let params: ModelsEnsureParams = parse_params(params)?;
            let result = handle_models_ensure(params)?;
            Ok(MethodOutcome {
                result,
                should_shutdown: false,
            })
        }
        RpcMethod::VadDetect => {
            let params: VadDetectParams = parse_params(params)?;
            let result = handle_vad_detect(state, params)?;
            Ok(MethodOutcome {
                result,
                should_shutdown: false,
            })
        }
        RpcMethod::SttTranscribe => {
            let params: SttTranscribeParams = parse_params(params)?;
            let result = handle_stt_transcribe(state, params)?;
            Ok(MethodOutcome {
                result,
                should_shutdown: false,
            })
        }
        RpcMethod::TtsSynthesize => {
            let params: TtsSynthesizeParams = parse_params(params)?;
            let result = handle_tts_synthesize(state, params)?;
            Ok(MethodOutcome {
                result,
                should_shutdown: false,
            })
        }
        RpcMethod::TtsStream => {
            let params: TtsSynthesizeParams = parse_params(params)?;
            let result = handle_tts_stream(state, params, request_id, stdout)?;
            Ok(MethodOutcome {
                result,
                should_shutdown: false,
            })
        }
        RpcMethod::RuntimeShutdown => Ok(MethodOutcome {
            result: json!({"shutting_down": true}),
            should_shutdown: true,
        }),
    }
}

fn configure_models_dir_from_env() {
    if let Ok(path) = env::var("JACK_VOICE_MODELS_DIR") {
        let path = path.trim();
        if !path.is_empty() {
            models::set_models_dir(PathBuf::from(path));
            eprintln!("[bridge] models_dir from env: {path}");
        }
    }
}

fn handle_runtime_hello(params: RuntimeHelloParams) -> Result<Value, RpcError> {
    if let Some(models_dir) = params.models_dir {
        let models_dir = models_dir.trim();
        if models_dir.is_empty() {
            return Err(RpcError::new(
                ErrorCode::InvalidParams,
                "models_dir must not be empty",
            ));
        }

        models::set_models_dir(PathBuf::from(models_dir));
        eprintln!("[bridge] models_dir overridden via runtime.hello: {models_dir}");
    }

    let status = build_models_status()?;

    Ok(json!({
        "protocol_version": PROTOCOL_VERSION,
        "bridge": {
            "name": env!("CARGO_PKG_NAME"),
            "version": env!("CARGO_PKG_VERSION")
        },
        "methods": RpcMethod::supported(),
        "audio": {
            "default_input_format": "pcm_s16le",
            "supported_input_formats": ["pcm_s16le", "f32le"],
            "default_sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
            "default_channels": DEFAULT_CHANNELS,
            "tts_output_format": "f32le",
            "tts_streaming_events": ["tts.start", "tts.chunk", "tts.end"]
        },
        "models": status
    }))
}

fn build_models_status() -> Result<Value, RpcError> {
    let models_dir = models::get_models_dir().map_err(map_model_error)?;

    let supertonic_ready = models::get_supertonic_paths()
        .map(|paths| paths.all_exist())
        .unwrap_or(false);

    let readiness = json!({
        "silero_vad": models::model_exists("silero_vad.onnx"),
        "whisper_base": models::model_exists("sherpa-onnx-whisper-base.en"),
        "smart_turn": models::model_exists("smart-turn-v3.2-cpu.onnx"),
        "moonshine": models::moonshine_model_ready(),
        "whisper_turbo": models::whisper_turbo_model_ready(),
        "kokoro": models::kokoro_model_ready(),
        "pocket": models::pocket_model_ready(),
        "paraformer": models::paraformer_model_ready(),
        "parakeet_eou": models::parakeet_eou_ready(),
        "parakeet_tdt": models::parakeet_tdt_ready(),
        "supertonic": supertonic_ready
    });

    let mut missing = Vec::new();
    if !models::model_exists("silero_vad.onnx") {
        missing.push("silero_vad.onnx");
    }
    if !models::model_exists("sherpa-onnx-whisper-base.en") {
        missing.push("sherpa-onnx-whisper-base.en");
    }
    if !models::model_exists("smart-turn-v3.2-cpu.onnx") {
        missing.push("smart-turn-v3.2-cpu.onnx");
    }
    if !models::moonshine_model_ready() {
        missing.push("sherpa-onnx-moonshine-base-en-int8");
    }
    if !models::kokoro_model_ready() {
        missing.push("kokoro-multi-lang-v1_0");
    }
    if !models::pocket_model_ready() {
        missing.push("pocket-tts");
    }
    if !models::parakeet_eou_ready() {
        missing.push("parakeet-eou");
    }
    if !models::parakeet_tdt_ready() {
        missing.push("parakeet-tdt");
    }
    if !supertonic_ready {
        missing.push("supertonic");
    }

    let core_missing: Vec<&'static str> = models::get_missing_models()
        .iter()
        .map(|bundle| bundle.name)
        .collect();

    Ok(json!({
        "models_dir": models_dir,
        "all_ready": missing.is_empty(),
        "missing": missing,
        "core_missing": core_missing,
        "core_total_size_mb": models::total_models_size_mb(),
        "readiness": readiness
    }))
}

fn handle_models_ensure(params: ModelsEnsureParams) -> Result<Value, RpcError> {
    if params.dry_run {
        return build_models_status();
    }

    let started = Instant::now();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| {
            RpcError::new(
                ErrorCode::InternalError,
                format!("Failed to initialize async runtime: {e}"),
            )
        })?;

    runtime
        .block_on(ensure_models_without_stdout(&StderrProgress))
        .map_err(map_model_error)?;
    let status = build_models_status()?;

    Ok(json!({
        "ensured": true,
        "elapsed_ms": started.elapsed().as_millis(),
        "status": status
    }))
}

async fn ensure_models_without_stdout(
    progress: &dyn ModelProgressCallback,
) -> Result<(), ModelError> {
    // `models::ensure_models` writes status lines to stdout, which would corrupt NDJSON.
    // Keep bridge protocol on stdout and perform equivalent steps here.
    for bundle in MODEL_BUNDLES {
        ensure_bundle(bundle, progress).await?;
    }

    models::ensure_supertonic_models(progress).await?;
    models::ensure_kokoro_model(progress).await?;
    models::ensure_pocket_model(progress).await?;
    models::ensure_moonshine_model(progress).await?;
    models::ensure_parakeet_models(progress).await?;

    Ok(())
}

async fn ensure_bundle(
    bundle: &ModelBundle,
    progress: &dyn ModelProgressCallback,
) -> Result<(), ModelError> {
    let target_name = if bundle.extract_dir.is_empty() {
        bundle.name
    } else {
        bundle.extract_dir
    };

    if models::model_exists(target_name) {
        return Ok(());
    }

    progress.on_download_start(bundle.name, bundle.size_mb);
    models::download_model(bundle, progress).await?;
    progress.on_download_complete(bundle.name);
    Ok(())
}

fn handle_vad_detect(state: &mut BridgeState, params: VadDetectParams) -> Result<Value, RpcError> {
    let samples = decode_audio_to_f32(&params.audio)?;

    if state.vad.is_none() {
        state.vad = Some(VoiceActivityDetector::new().map_err(map_vad_error)?);
    }

    let vad = state
        .vad
        .as_mut()
        .ok_or_else(|| RpcError::new(ErrorCode::InternalError, "VAD state unavailable"))?;

    let mut segment = vad.process(&samples).map_err(map_vad_error)?;
    if segment.is_none() && params.flush {
        segment = vad.flush().map_err(map_vad_error)?;
    }

    let is_speech = vad.is_speech_with_energy(&samples);

    let segment_json = segment.map(|segment| {
        let duration_ms = ((segment.end_time - segment.start_time).max(0.0) * 1000.0) as u64;
        json!({
            "audio_b64": encode_f32le_to_base64(&segment.samples),
            "format": "f32le",
            "sample_rate_hz": DEFAULT_SAMPLE_RATE_HZ,
            "channels": DEFAULT_CHANNELS,
            "sample_count": segment.samples.len(),
            "start_time_s": segment.start_time,
            "end_time_s": segment.end_time,
            "duration_ms": duration_ms
        })
    });

    Ok(json!({
        "is_speech": is_speech,
        "speech_detected": segment_json.is_some(),
        "segment": segment_json
    }))
}

fn handle_stt_transcribe(
    state: &mut BridgeState,
    params: SttTranscribeParams,
) -> Result<Value, RpcError> {
    let samples = decode_audio_to_f32(&params.audio)?;
    let mode = parse_stt_mode(params.mode.as_deref())?;
    let language = normalize_optional_string(params.language);
    let tts_voice = normalize_optional_string(params.tts_voice);

    let cache_key = SttCacheKey {
        mode,
        language: language.clone(),
        tts_voice: tts_voice.clone(),
    };

    if state.stt_key.as_ref() != Some(&cache_key) {
        let stt = SpeechToText::with_language(mode, language.clone(), tts_voice.clone())
            .map_err(map_stt_error)?;
        state.stt = Some(stt);
        state.stt_key = Some(cache_key);
    }

    let stt = state
        .stt
        .as_mut()
        .ok_or_else(|| RpcError::new(ErrorCode::InternalError, "STT state unavailable"))?;

    let result = stt.transcribe(&samples).map_err(map_stt_error)?;

    Ok(json!({
        "text": result.text,
        "is_final": result.is_final,
        "is_partial": result.is_partial,
        "latency_ms": result.latency_ms,
        "mode": mode.as_str(),
        "backend": stt_backend_name(stt)
    }))
}

fn handle_tts_synthesize(
    state: &mut BridgeState,
    params: TtsSynthesizeParams,
) -> Result<Value, RpcError> {
    validate_tts_output_format(params.format.as_deref())?;

    let requested_engine = parse_tts_engine(params.engine.as_deref())?;
    let engine_used = ensure_tts_instance(state, requested_engine)?;

    let tts = state
        .tts
        .as_mut()
        .ok_or_else(|| RpcError::new(ErrorCode::InternalError, "TTS state unavailable"))?;

    if let Some(voice) = normalize_optional_string(params.voice) {
        tts.set_speaker(&voice).map_err(map_tts_error)?;
    }

    if let Some(speed) = params.speed {
        tts.set_speed(speed);
    }

    let audio = tts.synthesize(params.text.trim()).map_err(map_tts_error)?;
    let duration_ms = duration_ms_for_samples(audio.samples.len(), audio.sample_rate);

    Ok(json!({
        "audio_b64": encode_f32le_to_base64(&audio.samples),
        "format": "f32le",
        "sample_rate_hz": audio.sample_rate,
        "channels": DEFAULT_CHANNELS,
        "duration_ms": duration_ms,
        "sample_count": audio.samples.len(),
        "engine": engine_used.as_str(),
        "voice": tts.current_speaker()
    }))
}

fn handle_tts_stream(
    state: &mut BridgeState,
    params: TtsSynthesizeParams,
    request_id: &str,
    stdout: &mut dyn Write,
) -> Result<Value, RpcError> {
    validate_tts_output_format(params.format.as_deref())?;

    let requested_engine = parse_tts_engine(params.engine.as_deref())?;
    let engine_used = ensure_tts_instance(state, requested_engine)?;

    let tts = state
        .tts
        .as_mut()
        .ok_or_else(|| RpcError::new(ErrorCode::InternalError, "TTS state unavailable"))?;

    if let Some(voice) = normalize_optional_string(params.voice) {
        tts.set_speaker(&voice).map_err(map_tts_error)?;
    }

    if let Some(speed) = params.speed {
        tts.set_speed(speed);
    }

    let voice_used = tts.current_speaker().to_string();

    write_event(
        stdout,
        &EventEnvelope::new(
            request_id,
            "tts.start",
            json!({
                "engine": engine_used.as_str(),
                "voice": voice_used.as_str(),
                "format": "f32le",
                "channels": DEFAULT_CHANNELS
            }),
        ),
    )
    .map_err(|err| {
        RpcError::new(
            ErrorCode::InternalError,
            format!("Failed to emit tts.start event: {err}"),
        )
    })?;

    let mut chunk_index: u64 = 0;
    let mut sample_count: usize = 0;
    let mut observed_sample_rate: Option<u32> = None;
    let mut emit_error: Option<io::Error> = None;

    let stream_sample_rate = tts
        .synthesize_streaming(params.text.trim(), |samples, sample_rate| {
            observed_sample_rate = Some(sample_rate);
            if emit_error.is_some() {
                return false;
            }

            let event = EventEnvelope::new(
                request_id,
                "tts.chunk",
                json!({
                    "index": chunk_index,
                    "audio_b64": encode_f32le_to_base64(samples),
                    "format": "f32le",
                    "sample_rate_hz": sample_rate,
                    "channels": DEFAULT_CHANNELS,
                    "sample_count": samples.len(),
                    "duration_ms": duration_ms_for_samples(samples.len(), sample_rate),
                    "engine": engine_used.as_str(),
                    "voice": voice_used.as_str()
                }),
            );

            if let Err(err) = write_event(stdout, &event) {
                emit_error = Some(err);
                return false;
            }

            chunk_index += 1;
            sample_count += samples.len();
            true
        })
        .map_err(map_tts_error)?;

    if let Some(err) = emit_error {
        return Err(RpcError::new(
            ErrorCode::InternalError,
            format!("Failed to emit tts.chunk event: {err}"),
        ));
    }

    let sample_rate = observed_sample_rate.unwrap_or(stream_sample_rate);
    let duration_ms = duration_ms_for_samples(sample_count, sample_rate);

    write_event(
        stdout,
        &EventEnvelope::new(
            request_id,
            "tts.end",
            json!({
                "engine": engine_used.as_str(),
                "voice": voice_used.as_str(),
                "format": "f32le",
                "sample_rate_hz": sample_rate,
                "channels": DEFAULT_CHANNELS,
                "sample_count": sample_count,
                "duration_ms": duration_ms,
                "chunk_count": chunk_index
            }),
        ),
    )
    .map_err(|err| {
        RpcError::new(
            ErrorCode::InternalError,
            format!("Failed to emit tts.end event: {err}"),
        )
    })?;

    Ok(json!({
        "streamed": true,
        "event": "tts.chunk",
        "engine": engine_used.as_str(),
        "voice": voice_used.as_str(),
        "format": "f32le",
        "sample_rate_hz": sample_rate,
        "channels": DEFAULT_CHANNELS,
        "sample_count": sample_count,
        "duration_ms": duration_ms,
        "chunk_count": chunk_index,
        "native_streaming": engine_used == CachedTtsEngine::Pocket
    }))
}

fn validate_tts_output_format(format: Option<&str>) -> Result<(), RpcError> {
    let output_format = format.unwrap_or("f32le").trim().to_ascii_lowercase();
    if output_format == "f32le" {
        Ok(())
    } else {
        Err(RpcError::new(
            ErrorCode::UnsupportedAudioFormat,
            format!("Unsupported tts output format '{}'", output_format),
        ))
    }
}

fn duration_ms_for_samples(sample_count: usize, sample_rate: u32) -> u64 {
    if sample_rate == 0 {
        0
    } else {
        (sample_count as u64 * 1000) / sample_rate as u64
    }
}

fn ensure_tts_instance(
    state: &mut BridgeState,
    requested: RequestedTtsEngine,
) -> Result<CachedTtsEngine, RpcError> {
    match requested {
        RequestedTtsEngine::Auto => {
            if let Some(current) = state.tts_engine {
                return Ok(current);
            }

            match TextToSpeech::with_engine(TtsEngine::Pocket) {
                Ok(tts) => {
                    state.tts = Some(tts);
                    state.tts_engine = Some(CachedTtsEngine::Pocket);
                    Ok(CachedTtsEngine::Pocket)
                }
                Err(pocket_err) => {
                    eprintln!(
                        "[bridge] auto TTS fallback: pocket init failed ({pocket_err}), trying kokoro"
                    );

                    match TextToSpeech::with_engine(TtsEngine::Kokoro) {
                        Ok(kokoro) => {
                            state.tts = Some(kokoro);
                            state.tts_engine = Some(CachedTtsEngine::Kokoro);
                            Ok(CachedTtsEngine::Kokoro)
                        }
                        Err(kokoro_err) => {
                            eprintln!(
                                "[bridge] auto TTS fallback: kokoro init failed ({kokoro_err}), trying supertonic"
                            );

                            let supertonic = TextToSpeech::with_engine(TtsEngine::Supertonic)
                                .map_err(map_tts_error)?;
                            state.tts = Some(supertonic);
                            state.tts_engine = Some(CachedTtsEngine::Supertonic);
                            Ok(CachedTtsEngine::Supertonic)
                        }
                    }
                }
            }
        }
        RequestedTtsEngine::Pocket => set_tts_engine(state, CachedTtsEngine::Pocket),
        RequestedTtsEngine::Supertonic => set_tts_engine(state, CachedTtsEngine::Supertonic),
        RequestedTtsEngine::Kokoro => set_tts_engine(state, CachedTtsEngine::Kokoro),
    }
}

fn set_tts_engine(
    state: &mut BridgeState,
    target: CachedTtsEngine,
) -> Result<CachedTtsEngine, RpcError> {
    if state.tts_engine == Some(target) {
        return Ok(target);
    }

    let tts = TextToSpeech::with_engine(target.as_jack_voice_engine()).map_err(map_tts_error)?;
    state.tts = Some(tts);
    state.tts_engine = Some(target);
    Ok(target)
}

fn parse_stt_mode(mode: Option<&str>) -> Result<SttMode, RpcError> {
    let mode = mode.unwrap_or("auto").trim().to_ascii_lowercase();

    match mode.as_str() {
        "batch" | "auto" => Ok(SttMode::Batch),
        "streaming" => Ok(SttMode::Streaming),
        other => Err(RpcError::new(
            ErrorCode::InvalidParams,
            format!("Unsupported stt mode '{other}'"),
        )),
    }
}

fn parse_tts_engine(engine: Option<&str>) -> Result<RequestedTtsEngine, RpcError> {
    let engine = engine.unwrap_or("auto").trim().to_ascii_lowercase();

    match engine.as_str() {
        "auto" => Ok(RequestedTtsEngine::Auto),
        "pocket" => Ok(RequestedTtsEngine::Pocket),
        "supertonic" => Ok(RequestedTtsEngine::Supertonic),
        "kokoro" => Ok(RequestedTtsEngine::Kokoro),
        other => Err(RpcError::new(
            ErrorCode::InvalidParams,
            format!("Unsupported tts engine '{other}'"),
        )),
    }
}

fn stt_backend_name(stt: &SpeechToText) -> &'static str {
    match &stt.backend {
        SttBackend::Batch(_) => "whisper",
        SttBackend::Streaming(_) => "moonshine",
        SttBackend::ParakeetTdt(_) => "parakeet_tdt",
        SttBackend::ParakeetEou(_) => "parakeet_eou",
        SttBackend::WhisperTurbo(_) => "whisper_turbo",
    }
}

fn parse_params<T: DeserializeOwned>(params: Value) -> Result<T, RpcError> {
    let params = if params.is_null() { json!({}) } else { params };

    serde_json::from_value(params)
        .map_err(|err| RpcError::new(ErrorCode::InvalidParams, format!("Invalid params: {err}")))
}

fn normalize_optional_string(value: Option<String>) -> Option<String> {
    value.and_then(|v| {
        let trimmed = v.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn map_model_error(err: ModelError) -> RpcError {
    match err {
        ModelError::ModelNotFound(msg) => RpcError::new(ErrorCode::ModelMissing, msg),
        other => RpcError::new(ErrorCode::InternalError, other.to_string()),
    }
}

fn map_vad_error(err: VadError) -> RpcError {
    match err {
        VadError::ModelNotFound(msg) => RpcError::new(ErrorCode::ModelMissing, msg),
        other => RpcError::new(ErrorCode::InternalError, other.to_string()),
    }
}

fn map_stt_error(err: SttError) -> RpcError {
    match err {
        SttError::ModelNotFound(msg) => RpcError::new(ErrorCode::ModelMissing, msg),
        other => RpcError::new(ErrorCode::InternalError, other.to_string()),
    }
}

fn map_tts_error(err: TtsError) -> RpcError {
    match err {
        TtsError::ModelNotFound(msg) => RpcError::new(ErrorCode::ModelMissing, msg),
        other => RpcError::new(ErrorCode::InternalError, other.to_string()),
    }
}

fn extract_request_id(line: &str) -> Option<String> {
    let value: Value = serde_json::from_str(line).ok()?;
    value.get("id")?.as_str().map(ToString::to_string)
}

fn write_response(stdout: &mut dyn Write, response: &ResponseEnvelope) -> io::Result<()> {
    let encoded = serde_json::to_string(response)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))?;
    writeln!(stdout, "{encoded}")?;
    stdout.flush()
}

fn write_event(stdout: &mut dyn Write, event: &EventEnvelope) -> io::Result<()> {
    let encoded = serde_json::to_string(event)
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err.to_string()))?;
    writeln!(stdout, "{encoded}")?;
    stdout.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tts_engine_accepts_pocket() {
        assert!(matches!(
            parse_tts_engine(Some("pocket")),
            Ok(RequestedTtsEngine::Pocket)
        ));
        assert!(matches!(
            parse_tts_engine(Some("POCKET")),
            Ok(RequestedTtsEngine::Pocket)
        ));
    }

    #[test]
    fn parse_tts_engine_rejects_unknown() {
        let err = parse_tts_engine(Some("not-an-engine")).expect_err("expected parse error");
        assert_eq!(err.code, ErrorCode::InvalidParams);
        assert!(
            err.message.contains("Unsupported tts engine"),
            "unexpected message: {}",
            err.message
        );
    }
}
