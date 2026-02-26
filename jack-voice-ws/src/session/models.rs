use crate::protocol::SessionState;
use serde::{Deserialize, Serialize};

/// Session record stored in database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub state: SessionState,
    pub config: Option<String>,
    pub conversation_id: Option<String>,
    pub metadata: Option<String>,
    pub expires_at: Option<i64>,
}

/// Conversation item stored in database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationItem {
    pub id: String,
    pub session_id: String,
    pub item_type: String,
    pub role: Option<String>,
    pub content: Option<String>,
    pub created_at: i64,
}

/// Create session request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionRequest {
    pub id: Option<String>,
    pub config: Option<serde_json::Value>,
    pub metadata: Option<serde_json::Value>,
    pub expires_in_seconds: Option<i64>,
}

/// Create session response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateSessionResponse {
    pub session: Session,
    pub conversation: Conversation,
}

/// Conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub created_at: i64,
}

/// Update session request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateSessionRequest {
    pub state: Option<SessionState>,
    pub config: Option<serde_json::Value>,
    pub metadata: Option<serde_json::Value>,
}

/// Get session response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetSessionResponse {
    pub session: Session,
    pub conversation: Option<Conversation>,
    pub items: Vec<ConversationItem>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_serialization() {
        let session = Session {
            id: "sess_abc123".to_string(),
            created_at: 1234567890,
            updated_at: 1234567890,
            state: SessionState::Ready,
            config: Some(r#"{"voice":"alloy"}"#.to_string()),
            conversation_id: Some("conv_xyz".to_string()),
            metadata: None,
            expires_at: Some(1234567890 + 3600),
        };

        let json = serde_json::to_string(&session).unwrap();
        assert!(json.contains(r#""id":"sess_abc123""#));
        assert!(json.contains(r#""state":"ready""#));
    }

    #[test]
    fn test_session_deserialization() {
        let json = r#"{
            "id": "sess_test",
            "created_at": 1234567890,
            "updated_at": 1234567890,
            "state": "active",
            "config": null,
            "conversation_id": null,
            "metadata": null,
            "expires_at": null
        }"#;

        let session: Session = serde_json::from_str(json).unwrap();
        assert_eq!(session.id, "sess_test");
        assert_eq!(session.state, SessionState::Active);
    }

    #[test]
    fn test_conversation_item() {
        let item = ConversationItem {
            id: "item_123".to_string(),
            session_id: "sess_abc".to_string(),
            item_type: "message".to_string(),
            role: Some("user".to_string()),
            content: Some(r#"{"text":"Hello"}"#.to_string()),
            created_at: 1234567890,
        };

        let json = serde_json::to_string(&item).unwrap();
        assert!(json.contains(r#""role":"user""#));
    }

    #[test]
    fn test_create_session_request() {
        let req = CreateSessionRequest {
            id: None,
            config: Some(serde_json::json!({"voice": "alloy"})),
            metadata: None,
            expires_in_seconds: Some(3600),
        };

        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(r#""voice":"alloy""#));
    }
}
