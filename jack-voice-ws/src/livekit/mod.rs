//! LiveKit integration for jack-voice-realtime
//!
//! Provides WebRTC transport via LiveKit for the OpenAI-compatible realtime API

#[cfg(feature = "livekit")]
mod participant;

#[cfg(feature = "livekit")]
mod audio;

#[cfg(feature = "livekit")]
pub use participant::*;

#[cfg(feature = "livekit")]
pub use audio::*;

/// Placeholder types when LiveKit is not enabled
#[cfg(not(feature = "livekit"))]
mod placeholder {
    use anyhow::Result;

    /// LiveKit configuration (placeholder)
    #[derive(Debug, Clone)]
    pub struct LiveKitConfig {
        pub url: String,
    }

    impl LiveKitConfig {
        pub fn new(_url: &str) -> Self {
            Self { url: _url.to_string() }
        }
    }

    /// LiveKit participant (placeholder)
    pub struct LiveKitParticipant;

    impl LiveKitParticipant {
        pub fn new(_config: LiveKitConfig) -> Self {
            Self
        }

        pub async fn connect(&mut self) -> Result<()> {
            anyhow::bail!("LiveKit support not enabled. Enable with --features livekit")
        }
    }

    /// Audio config (placeholder)
    #[derive(Debug, Clone, Default)]
    pub struct LiveKitAudioConfig {
        pub sample_rate: u32,
    }
}

#[cfg(not(feature = "livekit"))]
pub use placeholder::*;
