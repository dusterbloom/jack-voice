use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::str::FromStr;

pub const PROTOCOL_VERSION: &str = "1.0.0";
pub const MAX_REQUEST_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Deserialize)]
pub struct RequestEnvelope {
    #[serde(rename = "type")]
    pub message_type: String,
    pub id: String,
    pub method: String,
    #[serde(default)]
    pub params: Value,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ResponseEnvelope {
    #[serde(rename = "type")]
    pub message_type: &'static str,
    pub id: String,
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorBody>,
}

impl ResponseEnvelope {
    pub fn ok(id: impl Into<String>, result: Value) -> Self {
        Self {
            message_type: "response",
            id: id.into(),
            ok: true,
            result: Some(result),
            error: None,
        }
    }

    pub fn err(id: impl Into<String>, err: RpcError) -> Self {
        Self {
            message_type: "response",
            id: id.into(),
            ok: false,
            result: None,
            error: Some(ErrorBody {
                code: err.code.as_str(),
                message: err.message,
                retryable: err.retryable,
            }),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ErrorBody {
    pub code: &'static str,
    pub message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone)]
pub struct RpcError {
    pub code: ErrorCode,
    pub message: String,
    pub retryable: bool,
}

impl RpcError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            retryable: code.retryable(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    ParseError,
    InvalidRequest,
    InvalidParams,
    MethodNotFound,
    PayloadTooLarge,
    UnsupportedAudioFormat,
    AudioDecodeFailed,
    ModelMissing,
    OperationTimeout,
    InternalError,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ParseError => "PARSE_ERROR",
            Self::InvalidRequest => "INVALID_REQUEST",
            Self::InvalidParams => "INVALID_PARAMS",
            Self::MethodNotFound => "METHOD_NOT_FOUND",
            Self::PayloadTooLarge => "PAYLOAD_TOO_LARGE",
            Self::UnsupportedAudioFormat => "UNSUPPORTED_AUDIO_FORMAT",
            Self::AudioDecodeFailed => "AUDIO_DECODE_FAILED",
            Self::ModelMissing => "MODEL_MISSING",
            Self::OperationTimeout => "OPERATION_TIMEOUT",
            Self::InternalError => "INTERNAL_ERROR",
        }
    }

    pub fn retryable(self) -> bool {
        matches!(self, Self::OperationTimeout | Self::InternalError)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RpcMethod {
    RuntimeHello,
    ModelsStatus,
    ModelsEnsure,
    VadDetect,
    SttTranscribe,
    TtsSynthesize,
    RuntimeShutdown,
}

impl RpcMethod {
    pub fn supported() -> &'static [&'static str] {
        &[
            "runtime.hello",
            "models.status",
            "models.ensure",
            "vad.detect",
            "stt.transcribe",
            "tts.synthesize",
            "runtime.shutdown",
        ]
    }
}

impl FromStr for RpcMethod {
    type Err = RpcError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "runtime.hello" => Ok(Self::RuntimeHello),
            "models.status" => Ok(Self::ModelsStatus),
            "models.ensure" => Ok(Self::ModelsEnsure),
            "vad.detect" => Ok(Self::VadDetect),
            "stt.transcribe" => Ok(Self::SttTranscribe),
            "tts.synthesize" => Ok(Self::TtsSynthesize),
            "runtime.shutdown" => Ok(Self::RuntimeShutdown),
            _ => Err(RpcError::new(
                ErrorCode::MethodNotFound,
                format!("Unknown method '{value}'"),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RpcMethod;
    use std::str::FromStr;

    #[test]
    fn parses_known_methods() {
        let methods = [
            "runtime.hello",
            "models.status",
            "models.ensure",
            "vad.detect",
            "stt.transcribe",
            "tts.synthesize",
            "runtime.shutdown",
        ];

        for method in methods {
            assert!(
                RpcMethod::from_str(method).is_ok(),
                "failed to parse {method}"
            );
        }
    }

    #[test]
    fn rejects_unknown_method() {
        assert!(RpcMethod::from_str("runtime.goodbye").is_err());
    }
}
