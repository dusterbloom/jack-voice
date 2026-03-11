//! WebSocket connection handling

use std::net::SocketAddr;
use std::sync::Arc;

use futures::{SinkExt, StreamExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc};
use tokio_tungstenite::{accept_async, tungstenite::Message};
use tracing::{error, info, warn};

use crate::protocol::{
    ClientEvent, ConversationItemAdded, InputAudioBufferSpeechStopped, ResponseAudioDelta,
    ResponseDone, ServerEvent, SessionCreated,
};
use crate::session::{CreateSessionRequest, SessionManager, UpdateSessionRequest};

pub struct RealtimeServer {
    listener: TcpListener,
    session_manager: Arc<SessionManager>,
    broadcast_tx: broadcast::Sender<String>,
}

impl RealtimeServer {
    pub async fn new(addr: &str, session_manager: SessionManager) -> anyhow::Result<Self> {
        let listener = TcpListener::bind(addr).await?;
        info!("WebSocket server listening on {}", addr);
        let (broadcast_tx, _) = broadcast::channel(1000);
        Ok(Self {
            listener,
            session_manager: Arc::new(session_manager),
            broadcast_tx,
        })
    }

    pub async fn run(self) -> anyhow::Result<()> {
        let session_manager = self.session_manager.clone();
        let broadcast_tx = self.broadcast_tx.clone();

        while let Ok((stream, addr)) = self.listener.accept().await {
            let sm = session_manager.clone();
            let bt = broadcast_tx.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_connection(stream, addr, sm, bt).await {
                    error!("Connection error: {}", e);
                }
            });
        }
        Ok(())
    }
}

async fn handle_connection(
    stream: TcpStream,
    addr: SocketAddr,
    session_manager: Arc<SessionManager>,
    broadcast_tx: broadcast::Sender<String>,
) -> anyhow::Result<()> {
    let ws_stream = accept_async(stream).await?;
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();
    info!("New WebSocket connection from {}", addr);

    let session_response = session_manager
        .create_session(CreateSessionRequest {
            id: None,
            config: None,
            metadata: None,
            expires_in_seconds: Some(3600),
        })
        .await?;

    let session_id = session_response.session.id.clone();
    let _conversation_id = session_response.conversation.id.clone();

    let session_created = ServerEvent::SessionCreated(SessionCreated {
        id: session_id.clone(),
        object_type: "realtime.session".to_string(),
        model: Some("jack-voice-local".to_string()),
        expires_at: Some(1234567890),
        protocols: Some(vec!["realtime".to_string()]),
        tools: None,
    });
    let json = serde_json::to_string(&session_created)?;
    ws_sender.send(Message::Text(json)).await?;

    let mut broadcast_rx = broadcast_tx.subscribe();
    let (event_tx, mut event_rx) = mpsc::channel::<String>(100);
    let event_sender = event_tx.clone();

    tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            if ws_sender.send(Message::Text(event)).await.is_err() {
                break;
            }
        }
    });

    while let Some(msg_result) = ws_receiver.next().await {
        match msg_result {
            Ok(Message::Text(text)) => {
                if let Err(e) = handle_message(
                    &text,
                    &session_id,
                    session_manager.as_ref(),
                    event_sender.clone(),
                )
                .await
                {
                    warn!("Error handling message: {}", e);
                }
            }
            Ok(Message::Close(_)) => {
                info!("Connection closed from {}", addr);
                break;
            }
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }

    session_manager.delete_session(&session_id).await.ok();
    info!("Session {} cleaned up", session_id);
    Ok(())
}

async fn handle_message(
    text: &str,
    session_id: &str,
    session_manager: &SessionManager,
    event_sender: mpsc::Sender<String>,
) -> anyhow::Result<()> {
    let event: ClientEvent = serde_json::from_str(text)?;
    match event {
        ClientEvent::SessionUpdate(update) => {
            session_manager
                .update_session(
                    session_id,
                    UpdateSessionRequest {
                        state: Some(crate::protocol::SessionState::Ready),
                        config: Some(serde_json::json!(update.session)),
                        metadata: None,
                    },
                )
                .await?;
            let event = ServerEvent::SessionUpdated(crate::protocol::SessionUpdated {
                session: Some(serde_json::json!({
                    "id": session_id,
                    "object": "realtime.session"
                })),
            });
            event_sender.send(serde_json::to_string(&event)?).await?;
        }
        ClientEvent::InputAudioBufferAppend(append) => {
            let _audio_bytes = base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                &append.audio_data,
            );
            info!("Received audio buffer");
        }
        ClientEvent::InputAudioBufferCommit => {
            let event = ServerEvent::InputAudioBufferSpeechStopped(InputAudioBufferSpeechStopped {
                audio_end_ms: 1000,
            });
            event_sender.send(serde_json::to_string(&event)?).await?;
        }
        ClientEvent::InputAudioBufferClear => {}
        ClientEvent::ResponseCreate(_create) => {
            let event = ServerEvent::ResponseAudioDelta(ResponseAudioDelta {
                id: "resp_123".to_string(),
                item_id: "item_456".to_string(),
                content_index: Some(0),
                delta: Some("dGVzdCBhdWRpbw==".to_string()),
                with_tokens: Some(false),
            });
            event_sender.send(serde_json::to_string(&event)?).await?;
            let event = ServerEvent::ResponseDone(ResponseDone {
                response: Some(serde_json::json!({"id": "resp_123", "status": "completed"})),
            });
            event_sender.send(serde_json::to_string(&event)?).await?;
        }
        ClientEvent::ConversationItemCreate(item_create) => {
            let event = ServerEvent::ConversationItemAdded(ConversationItemAdded {
                item: item_create
                    .item
                    .map(|i| serde_json::to_value(i).ok())
                    .flatten(),
                previous_item_id: item_create.previous_item_id,
            });
            event_sender.send(serde_json::to_string(&event)?).await?;
        }
        ClientEvent::ResponseCancel => {}
        ClientEvent::SessionCommit => {}
        ClientEvent::ConversationItemDelete(delete) => {
            let event =
                ServerEvent::ConversationItemDeleted(crate::protocol::ConversationItemDeleted {
                    id: delete.id,
                });
            event_sender.send(serde_json::to_string(&event)?).await?;
        }
        ClientEvent::ConversationItemTruncate(truncate) => {
            let event = ServerEvent::ConversationItemTruncated(
                crate::protocol::ConversationItemTruncated {
                    id: truncate.id,
                    content_index: truncate.content_index,
                    end_index: truncate.end_index,
                },
            );
            event_sender.send(serde_json::to_string(&event)?).await?;
        }
    }
    Ok(())
}

pub async fn start_server(addr: &str) -> anyhow::Result<()> {
    let session_manager = SessionManager::new_in_memory().await?;
    let server = RealtimeServer::new(addr, session_manager).await?;
    server.run().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_created_event() {
        let event = ServerEvent::SessionCreated(SessionCreated {
            id: "test_session".to_string(),
            object_type: "realtime.session".to_string(),
            model: Some("gpt-realtime".to_string()),
            expires_at: Some(1234567890),
            protocols: Some(vec!["realtime".to_string()]),
            tools: None,
        });
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"session.created""#));
    }
}
