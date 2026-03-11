//! End-to-end integration tests for jack-voice-realtime
//!
//! These tests verify the full flow from WebSocket connection through the voice pipeline

use serde_json::json;

use crate::protocol::{ClientEvent, ServerEvent};
use crate::session::SessionManager;

/// Test protocol message parsing
#[tokio::test]
async fn test_protocol_message_parsing() {
    // Test session.update
    let json_str = r#"{
        "type": "session.update",
        "session": {
            "voice": "alloy",
            "language": "it",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.5
            }
        }
    }"#;

    let event: ClientEvent = serde_json::from_str(json_str).unwrap();
    match event {
        ClientEvent::SessionUpdate(update) => {
            assert_eq!(update.session.voice, Some("alloy".to_string()));
            assert_eq!(update.session.language, Some("it".to_string()));
        }
        _ => panic!("Expected SessionUpdate"),
    }

    // Test input_audio_buffer.append
    let json_str = r#"{
        "type": "input_audio_buffer.append",
        "audio": "dGVzdCBhdWRpbw=="
    }"#;

    let event: ClientEvent = serde_json::from_str(json_str).unwrap();
    match event {
        ClientEvent::InputAudioBufferAppend(append) => {
            assert_eq!(append.audio_data, "dGVzdCBhdWRpbw==");
        }
        _ => panic!("Expected InputAudioBufferAppend"),
    }

    // Test response.create
    let json_str = r#"{
        "type": "response.create",
        "response": {
            "modalities": ["audio", "text"],
            "language": "it"
        }
    }"#;

    let event: ClientEvent = serde_json::from_str(json_str).unwrap();
    match event {
        ClientEvent::ResponseCreate(create) => {
            assert_eq!(
                create.response.and_then(|response| response.language),
                Some("it".to_string())
            );
        }
        _ => panic!("Expected ResponseCreate"),
    }
}

/// Test server event serialization
#[tokio::test]
async fn test_server_event_serialization() {
    // Test session.created
    let event = ServerEvent::SessionCreated(crate::protocol::SessionCreated {
        id: "test_session".to_string(),
        object_type: "realtime.session".to_string(),
        model: Some("gpt-realtime".to_string()),
        expires_at: Some(1234567890),
        protocols: Some(vec!["realtime".to_string()]),
        tools: None,
    });

    let json_str = serde_json::to_string(&event).unwrap();
    assert!(json_str.contains(r#""type":"session.created""#));
    assert!(json_str.contains(r#""id":"test_session""#));

    // Test audio.delta
    let event = ServerEvent::ResponseAudioDelta(crate::protocol::ResponseAudioDelta {
        id: "resp_123".to_string(),
        item_id: "item_456".to_string(),
        content_index: Some(0),
        delta: Some("base64data".to_string()),
        with_tokens: Some(false),
    });

    let json_str = serde_json::to_string(&event).unwrap();
    assert!(json_str.contains(r#""type":"response.audio.delta""#));

    // Test transcription.completed
    let event = ServerEvent::InputAudioTranscriptionCompleted(
        crate::protocol::InputAudioTranscriptionCompleted {
            id: "trans_123".to_string(),
            item_id: "item_456".to_string(),
            content_index: Some(0),
            transcript: Some("Hello world".to_string()),
            language: Some("en".to_string()),
            audio_ms: Some(1500),
            with_tokens: Some(false),
        },
    );

    let json_str = serde_json::to_string(&event).unwrap();
    assert!(json_str.contains(r#""type":"conversation.item.input_audio_transcription.completed""#));
    assert!(json_str.contains(r#""transcript":"Hello world""#));
}

/// Test audio buffer integration
#[tokio::test]
async fn test_audio_buffer_integration() {
    use crate::audio::{pcm16_to_f32, AudioBuffer};

    let mut buffer = AudioBuffer::new(16000, 1);

    // Generate 100ms of test audio
    let test_audio: Vec<u8> = (0..1600).map(|i| (i % 256) as u8).collect();

    // Append audio
    buffer.append(&test_audio).unwrap();

    // Verify size
    assert_eq!(buffer.sample_count(), 800); // 100ms at 8kHz

    // Get as f32
    let f32_samples = buffer.get_f32();
    assert!(!f32_samples.is_empty());

    // Clear and verify
    buffer.clear();
    assert!(buffer.is_empty());
}

/// Test session manager with database
#[tokio::test]
async fn test_session_manager_integration() {
    use crate::protocol::SessionState;
    use crate::session::{CreateSessionRequest, UpdateSessionRequest};

    let manager = SessionManager::new_in_memory().await.unwrap();

    // Create session
    let response = manager
        .create_session(CreateSessionRequest {
            id: Some("test_session".to_string()),
            config: Some(json!({"voice": "alloy"})),
            metadata: None,
            expires_in_seconds: Some(3600),
        })
        .await
        .unwrap();

    assert_eq!(response.session.id, "test_session");
    assert_eq!(response.session.state, SessionState::Connecting);

    // Update session
    let updated = manager
        .update_session(
            "test_session",
            UpdateSessionRequest {
                state: Some(SessionState::Ready),
                config: Some(json!({"voice": "shimmer"})),
                metadata: None,
            },
        )
        .await
        .unwrap();

    assert_eq!(updated.state, SessionState::Ready);

    // Get session
    let retrieved = manager.get_session("test_session").await.unwrap();
    assert_eq!(retrieved.session.state, SessionState::Ready);

    // Delete session
    manager.delete_session("test_session").await.unwrap();

    // Verify deleted
    assert!(manager.get_session("test_session").await.is_err());
}

/// Test LLM config
#[tokio::test]
async fn test_llm_config() {
    use crate::pipeline::LlmConfig;

    let config = LlmConfig {
        base_url: "http://localhost:11434".to_string(),
        api_key: None,
        model: "llama3".to_string(),
    };

    assert_eq!(config.model, "llama3");
    assert_eq!(config.base_url, "http://localhost:11434");
}

/// Test pipeline configuration
#[tokio::test]
async fn test_pipeline_config() {
    use crate::pipeline::LlmConfig;
    use crate::pipeline::PipelineConfig;

    let config = PipelineConfig {
        vad_enabled: true,
        vad_threshold: 0.3,
        vad_silence_ms: 1000,
        stt_mode: jack_voice::SttMode::Batch,
        llm_config: LlmConfig {
            base_url: "http://localhost:11434".to_string(),
            api_key: None,
            model: "llama3".to_string(),
        },
        tts_engine: jack_voice::TtsEngine::Pocket,
        tts_language: Some("it".to_string()),
        tts_voice: Some("alba".to_string()),
        turn_detection: true,
    };

    assert!(config.vad_enabled);
    assert_eq!(config.vad_threshold, 0.3);
    assert_eq!(config.tts_language, Some("it".to_string()));
    assert_eq!(config.tts_voice, Some("alba".to_string()));
}
