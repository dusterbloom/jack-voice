//! jack-voice-realtime - OpenAI-compatible WebSocket server for local voice inference
//!
//! Implements the OpenAI Realtime API protocol for full-duplex voice streaming

mod audio;
mod pipeline;
mod protocol;
mod server;
mod session;

use std::env;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,jack_voice_realtime=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse arguments
    let addr = env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:8080".to_string());

    tracing::info!("Starting jack-voice-realtime server on {}", addr);

    // Start server
    server::start_server(&addr).await?;

    Ok(())
}
