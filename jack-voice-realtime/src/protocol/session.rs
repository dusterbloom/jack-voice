use serde::{Deserialize, Serialize};

/// Session state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionState {
    Connecting,
    Authenticating,
    Ready,
    Active,
    Completed,
    Error,
    Ended,
}

impl Default for SessionState {
    fn default() -> Self {
        Self::Connecting
    }
}

impl std::fmt::Display for SessionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionState::Connecting => write!(f, "connecting"),
            SessionState::Authenticating => write!(f, "authenticating"),
            SessionState::Ready => write!(f, "ready"),
            SessionState::Active => write!(f, "active"),
            SessionState::Completed => write!(f, "completed"),
            SessionState::Error => write!(f, "error"),
            SessionState::Ended => write!(f, "ended"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_state_default() {
        let state: SessionState = SessionState::default();
        assert_eq!(state, SessionState::Connecting);
    }

    #[test]
    fn test_session_state_display() {
        assert_eq!(SessionState::Ready.to_string(), "ready");
        assert_eq!(SessionState::Active.to_string(), "active");
    }

    #[test]
    fn test_session_state_serialization() {
        let json = r#""ready""#;
        let state: SessionState = serde_json::from_str(json).unwrap();
        assert_eq!(state, SessionState::Ready);

        let serialized = serde_json::to_string(&SessionState::Active).unwrap();
        assert_eq!(serialized, r#""active""#);
    }
}
