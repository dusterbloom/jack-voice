//! OpenAI Realtime API Protocol Events
//!
//! This module implements the client and server events for OpenAI-compatible
//! WebSocket streaming. See: https://platform.openai.com/docs/api-reference/realtime-client-events

use serde::{Deserialize, Serialize};

/// Protocol version
pub const PROTOCOL_VERSION: &str = "v1";

/// Client events sent from the client to the server
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientEvent {
    /// Session configuration
    #[serde(rename = "session.update")]
    SessionUpdate(SessionUpdate),

    /// Commit session configuration
    #[serde(rename = "session.commit")]
    SessionCommit,

    /// Append audio to input buffer
    #[serde(rename = "input_audio_buffer.append")]
    InputAudioBufferAppend(InputAudioBufferAppend),

    /// Commit input audio buffer (trigger transcription)
    #[serde(rename = "input_audio_buffer.commit")]
    InputAudioBufferCommit,

    /// Clear input audio buffer
    #[serde(rename = "input_audio_buffer.clear")]
    InputAudioBufferClear,

    /// Create conversation item
    #[serde(rename = "conversation.item.create")]
    ConversationItemCreate(ConversationItemCreate),

    /// Delete conversation item
    #[serde(rename = "conversation.item.delete")]
    ConversationItemDelete(ConversationItemDelete),

    /// Truncate conversation item
    #[serde(rename = "conversation.item.truncate")]
    ConversationItemTruncate(ConversationItemTruncate),

    /// Create response
    #[serde(rename = "response.create")]
    ResponseCreate(ResponseCreate),

    /// Cancel response
    #[serde(rename = "response.cancel")]
    ResponseCancel,
}

/// Session update configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionUpdate {
    #[serde(default)]
    pub session: SessionConfig,
}

/// Session configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionConfig {
    #[serde(rename = "type", default)]
    pub session_type: Option<String>,

    #[serde(default)]
    pub instructions: Option<String>,

    #[serde(default)]
    pub voice: Option<String>,

    #[serde(default)]
    pub turn_detection: Option<TurnDetection>,

    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,

    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,

    #[serde(default)]
    pub input_audio_transcription: Option<AudioTranscriptionConfig>,

    #[serde(default)]
    pub audio: Option<AudioConfig>,

    #[serde(default)]
    pub model: Option<String>,
}

/// Turn detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnDetection {
    #[serde(rename = "type", default)]
    pub detection_type: Option<String>,

    #[serde(default)]
    pub threshold: Option<f32>,

    #[serde(rename = "silence_duration_ms", default)]
    pub silence_duration_ms: Option<u32>,

    #[serde(rename = "prefix_padding_ms", default)]
    pub prefix_padding_ms: Option<u32>,
}

/// Audio transcription configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioTranscriptionConfig {
    #[serde(default)]
    pub model: Option<String>,

    #[serde(default)]
    pub language: Option<String>,

    #[serde(default)]
    pub prompt: Option<String>,
}

/// Audio format configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioConfig {
    #[serde(default)]
    pub input: Option<AudioFormat>,

    #[serde(default)]
    pub output: Option<AudioFormat>,
}

/// Audio format
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioFormat {
    #[serde(rename = "type", default)]
    pub format_type: Option<String>,

    #[serde(default)]
    pub rate: Option<u32>,

    #[serde(default)]
    pub channels: Option<u32>,
}

/// Input audio buffer append
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudioBufferAppend {
    #[serde(rename = "audio")]
    pub audio_data: String, // base64 encoded
}

/// Conversation item create
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItemCreate {
    #[serde(default)]
    pub item: Option<ConversationItem>,

    #[serde(rename = "previous_item_id", default)]
    pub previous_item_id: Option<String>,
}

/// Conversation item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItem {
    #[serde(rename = "type", default)]
    pub item_type: Option<String>,

    pub id: Option<String>,

    #[serde(default)]
    pub status: Option<String>,

    #[serde(default)]
    pub role: Option<String>,

    #[serde(default)]
    pub content: Option<Vec<ContentPart>>,
}

/// Content part
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type", default)]
    pub part_type: Option<String>,

    #[serde(default)]
    pub text: Option<String>,

    #[serde(rename = "audio", default)]
    pub audio_data: Option<String>,

    #[serde(rename = "transcript", default)]
    pub transcript: Option<String>,
}

/// Conversation item delete
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItemDelete {
    pub id: String,
}

/// Conversation item truncate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItemTruncate {
    pub id: String,

    #[serde(rename = "content_index", default)]
    pub content_index: Option<u32>,

    pub end_index: Option<u32>,
}

/// Response create
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseCreate {
    #[serde(default)]
    pub response: Option<ResponseConfig>,
}

/// Response configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponseConfig {
    #[serde(default)]
    pub modalities: Option<Vec<String>>,

    #[serde(default)]
    pub instructions: Option<String>,

    #[serde(default)]
    pub voice: Option<String>,

    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,

    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,

    #[serde(default)]
    pub temperature: Option<f32>,

    #[serde(default)]
    pub max_output_tokens: Option<u32>,
}

// =============================================================================
// Server Events (sent from server to client)
// =============================================================================

/// Server events sent from the server to the client
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerEvent {
    #[serde(rename = "session.created")]
    SessionCreated(SessionCreated),

    #[serde(rename = "session.updated")]
    SessionUpdated(SessionUpdated),

    #[serde(rename = "session.ended")]
    SessionEnded(SessionEnded),

    #[serde(rename = "session.error")]
    SessionError(SessionError),

    #[serde(rename = "conversation.created")]
    ConversationCreated(ConversationCreated),

    #[serde(rename = "conversation.item.added")]
    ConversationItemAdded(ConversationItemAdded),

    #[serde(rename = "conversation.item.deleted")]
    ConversationItemDeleted(ConversationItemDeleted),

    #[serde(rename = "conversation.item.truncated")]
    ConversationItemTruncated(ConversationItemTruncated),

    #[serde(rename = "input_audio_buffer.speech_started")]
    InputAudioBufferSpeechStarted(InputAudioBufferSpeechStarted),

    #[serde(rename = "input_audio_buffer.speech_stopped")]
    InputAudioBufferSpeechStopped(InputAudioBufferSpeechStopped),

    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    InputAudioTranscriptionCompleted(InputAudioTranscriptionCompleted),

    #[serde(rename = "conversation.item.input_audio_transcription.delta")]
    InputAudioTranscriptionDelta(InputAudioTranscriptionDelta),

    #[serde(rename = "response.audio.delta")]
    ResponseAudioDelta(ResponseAudioDelta),

    #[serde(rename = "response.audio_transcript.delta")]
    ResponseAudioTranscriptDelta(ResponseAudioTranscriptDelta),

    #[serde(rename = "response.done")]
    ResponseDone(ResponseDone),

    #[serde(rename = "response.function_tool_call")]
    ResponseFunctionToolCall(ResponseFunctionToolCall),

    #[serde(rename = "error")]
    Error(ErrorEvent),
}

/// Session created
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCreated {
    pub id: String,

    #[serde(rename = "object")]
    pub object_type: String,

    #[serde(default)]
    pub model: Option<String>,

    #[serde(rename = "expires_at")]
    pub expires_at: Option<i64>,

    #[serde(default)]
    pub protocols: Option<Vec<String>>,

    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
}

/// Session updated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionUpdated {
    #[serde(default)]
    pub session: Option<serde_json::Value>,
}

/// Session ended
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEnded {
    #[serde(default)]
    pub reason: Option<String>,
}

/// Session error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionError {
    pub code: String,

    pub message: String,

    #[serde(default)]
    pub param: Option<String>,

    #[serde(default)]
    pub event_id: Option<String>,
}

/// Conversation created
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationCreated {
    pub id: String,

    #[serde(rename = "object")]
    pub object_type: String,
}

/// Conversation item added
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItemAdded {
    #[serde(default)]
    pub item: Option<serde_json::Value>,

    #[serde(rename = "previous_item_id", default)]
    pub previous_item_id: Option<String>,
}

/// Conversation item deleted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItemDeleted {
    pub id: String,
}

/// Conversation item truncated
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItemTruncated {
    pub id: String,

    #[serde(rename = "content_index", default)]
    pub content_index: Option<u32>,

    pub end_index: Option<u32>,
}

/// Input audio buffer speech started
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudioBufferSpeechStarted {
    #[serde(rename = "audio_start_ms")]
    pub audio_start_ms: u32,
}

/// Input audio buffer speech stopped
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudioBufferSpeechStopped {
    #[serde(rename = "audio_end_ms")]
    pub audio_end_ms: u32,
}

/// Input audio transcription completed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudioTranscriptionCompleted {
    pub id: String,

    #[serde(rename = "item_id")]
    pub item_id: String,

    #[serde(rename = "content_index", default)]
    pub content_index: Option<u32>,

    #[serde(default)]
    pub transcript: Option<String>,

    #[serde(default)]
    pub language: Option<String>,

    #[serde(default)]
    pub audio_ms: Option<u32>,

    #[serde(default)]
    pub with_tokens: Option<bool>,
}

/// Input audio transcription delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudioTranscriptionDelta {
    pub id: String,

    #[serde(rename = "item_id")]
    pub item_id: String,

    #[serde(rename = "content_index", default)]
    pub content_index: Option<u32>,

    #[serde(default)]
    pub delta: Option<String>,

    #[serde(default)]
    pub with_tokens: Option<bool>,
}

/// Response audio delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseAudioDelta {
    pub id: String,

    #[serde(rename = "item_id")]
    pub item_id: String,

    #[serde(rename = "content_index", default)]
    pub content_index: Option<u32>,

    #[serde(default)]
    pub delta: Option<String>,

    #[serde(default)]
    pub with_tokens: Option<bool>,
}

/// Response audio transcript delta
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseAudioTranscriptDelta {
    pub id: String,

    #[serde(rename = "item_id")]
    pub item_id: String,

    #[serde(rename = "content_index", default)]
    pub content_index: Option<u32>,

    #[serde(default)]
    pub delta: Option<String>,
}

/// Response done
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseDone {
    #[serde(default)]
    pub response: Option<serde_json::Value>,
}

/// Response function tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseFunctionToolCall {
    pub id: String,

    #[serde(rename = "item_id")]
    pub item_id: String,

    #[serde(rename = "function")]
    pub function_call: serde_json::Value,
}

/// Error event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEvent {
    pub code: String,

    pub message: String,

    #[serde(default)]
    pub param: Option<String>,

    #[serde(default)]
    pub event_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_session_update() {
        let json = r#"{
            "type": "session.update",
            "session": {
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "silence_duration_ms": 500
                },
                "voice": "alloy",
                "model": "gpt-realtime"
            }
        }"#;

        let event: ClientEvent = serde_json::from_str(json).unwrap();
        match event {
            ClientEvent::SessionUpdate(update) => {
                assert_eq!(update.session.voice, Some("alloy".to_string()));
                assert_eq!(update.session.model, Some("gpt-realtime".to_string()));
                let td = update.session.turn_detection.unwrap();
                assert_eq!(td.detection_type, Some("server_vad".to_string()));
                assert_eq!(td.threshold, Some(0.5));
            }
            _ => panic!("Expected SessionUpdate"),
        }
    }

    #[test]
    fn test_parse_input_audio_buffer_append() {
        let json = r#"{
            "type": "input_audio_buffer.append",
            "audio": "dGVzdCBhdWRpbw=="
        }"#;

        let event: ClientEvent = serde_json::from_str(json).unwrap();
        match event {
            ClientEvent::InputAudioBufferAppend(app) => {
                assert_eq!(app.audio_data, "dGVzdCBhdWRpbw==");
            }
            _ => panic!("Expected InputAudioBufferAppend"),
        }
    }

    #[test]
    fn test_parse_response_create() {
        let json = r#"{
            "type": "response.create",
            "response": {
                "modalities": ["audio", "text"],
                "voice": "alloy"
            }
        }"#;

        let event: ClientEvent = serde_json::from_str(json).unwrap();
        match event {
            ClientEvent::ResponseCreate(create) => {
                let modalities = create.response.unwrap().modalities.unwrap();
                assert!(modalities.contains(&"audio".to_string()));
                assert!(modalities.contains(&"text".to_string()));
            }
            _ => panic!("Expected ResponseCreate"),
        }
    }

    #[test]
    fn test_serialize_server_event_audio_delta() {
        let event = ServerEvent::ResponseAudioDelta(ResponseAudioDelta {
            id: "resp_abc123".to_string(),
            item_id: "item_xyz789".to_string(),
            content_index: Some(0),
            delta: Some("base64audiochunk".to_string()),
            with_tokens: Some(false),
        });

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"response.audio.delta""#));
        assert!(json.contains(r#""id":"resp_abc123""#));
    }

    #[test]
    fn test_serialize_server_event_transcription() {
        let event =
            ServerEvent::InputAudioTranscriptionCompleted(InputAudioTranscriptionCompleted {
                id: "trans_123".to_string(),
                item_id: "item_456".to_string(),
                content_index: Some(0),
                transcript: Some("Hello world".to_string()),
                language: Some("en".to_string()),
                audio_ms: Some(1500),
                with_tokens: Some(false),
            });

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"conversation.item.input_audio_transcription.completed""#));
        assert!(json.contains(r#""transcript":"Hello world""#));
    }

    #[test]
    fn test_parse_conversation_item_create() {
        let json = r#"{
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "id": "msg_123",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Hello"
                    }
                ]
            },
            "previous_item_id": "msg_122"
        }"#;

        let event: ClientEvent = serde_json::from_str(json).unwrap();
        match event {
            ClientEvent::ConversationItemCreate(create) => {
                assert_eq!(create.previous_item_id, Some("msg_122".to_string()));
                let item = create.item.unwrap();
                assert_eq!(item.id, Some("msg_123".to_string()));
            }
            _ => panic!("Expected ConversationItemCreate"),
        }
    }

    #[test]
    fn test_serialize_session_created() {
        let event = ServerEvent::SessionCreated(SessionCreated {
            id: "sess_abc".to_string(),
            object_type: "realtime.session".to_string(),
            model: Some("gpt-realtime".to_string()),
            expires_at: Some(1234567890),
            protocols: Some(vec!["realtime".to_string()]),
            tools: None,
        });

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"session.created""#));
        assert!(json.contains(r#""model":"gpt-realtime""#));
    }
}
