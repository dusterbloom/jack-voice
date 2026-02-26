//! Request handler utilities

use crate::protocol::{ClientEvent, ServerEvent};

/// Handler result type
pub type HandlerResult = Result<Option<ServerEvent>, HandlerError>;

/// Handler errors
#[derive(Debug)]
pub enum HandlerError {
    InvalidMessage(String),
    SessionNotFound,
    PipelineError(String),
    AudioError(String),
}

impl std::fmt::Display for HandlerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandlerError::InvalidMessage(msg) => write!(f, "Invalid message: {}", msg),
            HandlerError::SessionNotFound => write!(f, "Session not found"),
            HandlerError::PipelineError(msg) => write!(f, "Pipeline error: {}", msg),
            HandlerError::AudioError(msg) => write!(f, "Audio error: {}", msg),
        }
    }
}

impl std::error::Error for HandlerError {}

/// Handle a client event and return optional server event
pub async fn handle_event(
    event: ClientEvent,
    session_id: &str,
) -> HandlerResult {
    match event {
        ClientEvent::SessionUpdate(_) => {
            // Would update session config
            Ok(None)
        }
        
        ClientEvent::InputAudioBufferAppend(append) => {
            // Would add to input buffer
            let _ = append.audio_data;
            Ok(None)
        }
        
        ClientEvent::InputAudioBufferCommit => {
            // Would trigger transcription
            Ok(None)
        }
        
        ClientEvent::InputAudioBufferClear => {
            // Would clear buffer
            Ok(None)
        }
        
        ClientEvent::ResponseCreate(_) => {
            // Would generate response
            Ok(None)
        }
        
        ClientEvent::ResponseCancel => {
            // Would cancel response
            Ok(None)
        }
        
        ClientEvent::ConversationItemCreate(_) => {
            // Would add item
            Ok(None)
        }
        
        ClientEvent::ConversationItemDelete(_) => {
            // Would delete item
            Ok(None)
        }
        
        ClientEvent::ConversationItemTruncate(_) => {
            // Would truncate item
            Ok(None)
        }
        
        ClientEvent::SessionCommit => {
            // Would commit session
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handler_error_display() {
        let err = HandlerError::InvalidMessage("test".to_string());
        assert_eq!(err.to_string(), "Invalid message: test");

        let err = HandlerError::SessionNotFound;
        assert_eq!(err.to_string(), "Session not found");
    }
}
