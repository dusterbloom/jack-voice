use anyhow::Result;
use sqlx::{sqlite::SqlitePoolOptions, Row, SqlitePool};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

use super::{
    Conversation, ConversationItem, CreateSessionRequest, CreateSessionResponse,
    GetSessionResponse, Session, UpdateSessionRequest,
};
use crate::protocol::SessionState;

pub struct SessionManager {
    pool: SqlitePool,
    cache: Arc<RwLock<lru::LruCache<String, Session>>>,
}

// Simple LRU cache implementation for sessions
mod lru {
    use std::collections::HashMap;
    use std::hash::Hash;
    use std::vec::Vec;

    pub struct LruCache<K, V> {
        capacity: usize,
        map: HashMap<K, V>,
        order: Vec<K>,
    }

    impl<K: Hash + Eq + Clone, V> LruCache<K, V> {
        pub fn new(capacity: usize) -> Self {
            Self {
                capacity,
                map: HashMap::new(),
                order: Vec::new(),
            }
        }

        pub fn get(&self, key: &K) -> Option<&V> {
            self.map.get(key)
        }

        pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
            self.map.get_mut(key)
        }

        pub fn put(&mut self, key: K, value: V) -> Option<V> {
            if let Some(old) = self.map.insert(key.clone(), value) {
                if let Some(pos) = self.order.iter().position(|k| k == &key) {
                    self.order.remove(pos);
                }
                self.order.push(key);
                Some(old)
            } else {
                if self.order.len() >= self.capacity {
                    if let Some(oldest) = self.order.first() {
                        self.map.remove(oldest);
                        self.order.remove(0);
                    }
                }
                self.order.push(key);
                None
            }
        }

        pub fn remove(&mut self, key: &K) -> Option<V> {
            if let Some(v) = self.map.remove(key) {
                if let Some(pos) = self.order.iter().position(|k| k == key) {
                    self.order.remove(pos);
                }
                Some(v)
            } else {
                None
            }
        }

        pub fn len(&self) -> usize {
            self.map.len()
        }
    }
}

impl SessionManager {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect(database_url)
            .await?;

        Ok(Self {
            pool,
            cache: Arc::new(RwLock::new(lru::LruCache::new(100))),
        })
    }

    pub async fn new_in_memory() -> Result<Self> {
        let manager = Self::new("sqlite::memory:").await?;
        manager.run_migrations().await?;
        Ok(manager)
    }

    pub async fn run_migrations(&self) -> Result<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                state TEXT NOT NULL DEFAULT 'connecting',
                config TEXT,
                conversation_id TEXT,
                metadata TEXT,
                expires_at INTEGER
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY NOT NULL,
                session_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query(
            "CREATE TABLE IF NOT EXISTS conversation_items (
                id TEXT PRIMARY KEY NOT NULL,
                session_id TEXT NOT NULL,
                item_type TEXT NOT NULL,
                role TEXT,
                content TEXT,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )",
        )
        .execute(&self.pool)
        .await?;

        sqlx::query("CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)")
            .execute(&self.pool)
            .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_sessions_conversation ON sessions(conversation_id)",
        )
        .execute(&self.pool)
        .await?;
        sqlx::query("CREATE INDEX IF NOT EXISTS idx_conversation_items_session ON conversation_items(session_id)")
            .execute(&self.pool)
            .await?;
        sqlx::query(
            "CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id)",
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    pub async fn create_session(&self, req: CreateSessionRequest) -> Result<CreateSessionResponse> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;

        let session_id = req.id.unwrap_or_else(|| format!("sess_{}", uuid_v4()));
        let conversation_id = format!("conv_{}", uuid_v4());

        let config = req
            .config
            .map(|c| serde_json::to_string(&c).unwrap_or_default());
        let metadata = req
            .metadata
            .map(|m| serde_json::to_string(&m).unwrap_or_default());

        let expires_at = req.expires_in_seconds.map(|s| now + s);

        // Insert session
        sqlx::query(
            "INSERT INTO sessions (id, created_at, updated_at, state, config, conversation_id, metadata, expires_at) 
             VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
        )
        .bind(&session_id)
        .bind(now)
        .bind(now)
        .bind(SessionState::Connecting.to_string())
        .bind(&config)
        .bind(&conversation_id)
        .bind(&metadata)
        .bind(expires_at)
        .execute(&self.pool)
        .await?;

        // Insert conversation
        sqlx::query("INSERT INTO conversations (id, session_id, created_at) VALUES (?, ?, ?)")
            .bind(&conversation_id)
            .bind(&session_id)
            .bind(now)
            .execute(&self.pool)
            .await?;

        let session = Session {
            id: session_id.clone(),
            created_at: now,
            updated_at: now,
            state: SessionState::Connecting,
            config,
            conversation_id: Some(conversation_id.clone()),
            metadata,
            expires_at,
        };

        Ok(CreateSessionResponse {
            session,
            conversation: Conversation {
                id: conversation_id,
                created_at: now,
            },
        })
    }

    pub async fn get_session(&self, session_id: &str) -> Result<GetSessionResponse> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            let session_id_str = session_id.to_string();
            if let Some(session) = cache.get(&session_id_str) {
                let items = self.get_conversation_items(session_id).await?;
                return Ok(GetSessionResponse {
                    session: session.clone(),
                    conversation: session.conversation_id.as_ref().map(|cid| Conversation {
                        id: cid.clone(),
                        created_at: session.created_at,
                    }),
                    items,
                });
            }
        }

        // Load from database
        let row: (String, i64, i64, String, Option<String>, Option<String>, Option<String>, Option<i64>) =
            sqlx::query_as(
                "SELECT id, created_at, updated_at, state, config, conversation_id, metadata, expires_at 
                 FROM sessions WHERE id = ?"
            )
            .bind(session_id)
            .fetch_one(&self.pool)
            .await?;

        let state: SessionState = row
            .3
            .as_str()
            .try_into()
            .unwrap_or(SessionState::Connecting);

        let session = Session {
            id: row.0,
            created_at: row.1,
            updated_at: row.2,
            state,
            config: row.4,
            conversation_id: row.5,
            metadata: row.6,
            expires_at: row.7,
        };

        // Update cache
        self.cache
            .write()
            .await
            .put(session_id.to_string(), session.clone());

        let items = self.get_conversation_items(session_id).await?;

        Ok(GetSessionResponse {
            conversation: session.conversation_id.as_ref().map(|cid| Conversation {
                id: cid.clone(),
                created_at: session.created_at,
            }),
            session,
            items,
        })
    }

    async fn get_conversation_items(&self, session_id: &str) -> Result<Vec<ConversationItem>> {
        let rows = sqlx::query_as(
            "SELECT id, session_id, item_type, role, content, created_at 
             FROM conversation_items WHERE session_id = ? ORDER BY created_at",
        )
        .bind(session_id)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(
                |row: (String, String, String, Option<String>, Option<String>, i64)| {
                    ConversationItem {
                        id: row.0,
                        session_id: row.1,
                        item_type: row.2,
                        role: row.3,
                        content: row.4,
                        created_at: row.5,
                    }
                },
            )
            .collect())
    }

    pub async fn update_session(
        &self,
        session_id: &str,
        req: UpdateSessionRequest,
    ) -> Result<Session> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;

        // Build and execute updates based on what was provided
        if let Some(state) = &req.state {
            sqlx::query("UPDATE sessions SET state = ?, updated_at = ? WHERE id = ?")
                .bind(state.to_string())
                .bind(now)
                .bind(session_id)
                .execute(&self.pool)
                .await?;
        }

        if let Some(ref config) = req.config {
            let config_str = serde_json::to_string(config).unwrap_or_default();
            sqlx::query("UPDATE sessions SET config = ?, updated_at = ? WHERE id = ?")
                .bind(config_str)
                .bind(now)
                .bind(session_id)
                .execute(&self.pool)
                .await?;
        }

        if let Some(ref metadata) = req.metadata {
            let metadata_str = serde_json::to_string(metadata).unwrap_or_default();
            sqlx::query("UPDATE sessions SET metadata = ?, updated_at = ? WHERE id = ?")
                .bind(metadata_str)
                .bind(now)
                .bind(session_id)
                .execute(&self.pool)
                .await?;
        }

        // Invalidate cache
        self.cache.write().await.remove(&session_id.to_string());

        // Return updated session
        Ok(self.get_session(session_id).await?.session)
    }

    pub async fn add_conversation_item(
        &self,
        session_id: &str,
        item: ConversationItem,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO conversation_items (id, session_id, item_type, role, content, created_at) 
             VALUES (?, ?, ?, ?, ?, ?)",
        )
        .bind(&item.id)
        .bind(session_id)
        .bind(&item.item_type)
        .bind(&item.role)
        .bind(&item.content)
        .bind(item.created_at)
        .execute(&self.pool)
        .await?;

        // Invalidate cache
        self.cache.write().await.remove(&session_id.to_string());

        Ok(())
    }

    pub async fn delete_session(&self, session_id: &str) -> Result<()> {
        // Delete conversation items first
        sqlx::query("DELETE FROM conversation_items WHERE session_id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await?;

        // Delete conversation
        sqlx::query("DELETE FROM conversations WHERE session_id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await?;

        // Delete session
        sqlx::query("DELETE FROM sessions WHERE id = ?")
            .bind(session_id)
            .execute(&self.pool)
            .await?;

        // Remove from cache
        self.cache.write().await.remove(&session_id.to_string());

        Ok(())
    }

    pub async fn cleanup_expired(&self) -> Result<u64> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64;

        let result =
            sqlx::query("SELECT id FROM sessions WHERE expires_at IS NOT NULL AND expires_at < ?")
                .bind(now)
                .fetch_all(&self.pool)
                .await?;

        let mut deleted = 0u64;
        for row in result {
            let id: String = row.get(0);
            self.delete_session(&id).await?;
            deleted += 1;
        }

        Ok(deleted)
    }
}

// Simple UUID v4 generator (for testing)
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:032x}", timestamp)
}

impl TryFrom<&str> for crate::protocol::SessionState {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, <Self as TryFrom<&str>>::Error> {
        match value {
            "connecting" => Ok(crate::protocol::SessionState::Connecting),
            "authenticating" => Ok(crate::protocol::SessionState::Authenticating),
            "ready" => Ok(crate::protocol::SessionState::Ready),
            "active" => Ok(crate::protocol::SessionState::Active),
            "completed" => Ok(crate::protocol::SessionState::Completed),
            "error" => Ok(crate::protocol::SessionState::Error),
            "ended" => Ok(crate::protocol::SessionState::Ended),
            _ => Err(format!("Unknown session state: {}", value)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_and_get_session() {
        let manager = SessionManager::new_in_memory().await.unwrap();

        let req = CreateSessionRequest {
            id: Some("test_session".to_string()),
            config: Some(serde_json::json!({"voice": "alloy"})),
            metadata: None,
            expires_in_seconds: Some(3600),
        };

        let response = manager.create_session(req).await.unwrap();

        assert_eq!(response.session.id, "test_session");
        assert_eq!(response.session.state, SessionState::Connecting);
        assert!(response.conversation.id.starts_with("conv_"));

        let retrieved = manager.get_session("test_session").await.unwrap();
        assert_eq!(retrieved.session.state, SessionState::Connecting);
    }

    #[tokio::test]
    async fn test_update_session() {
        let manager = SessionManager::new_in_memory().await.unwrap();

        manager
            .create_session(CreateSessionRequest {
                id: Some("update_test".to_string()),
                config: None,
                metadata: None,
                expires_in_seconds: None,
            })
            .await
            .unwrap();

        let updated = manager
            .update_session(
                "update_test",
                UpdateSessionRequest {
                    state: Some(SessionState::Ready),
                    config: Some(serde_json::json!({"voice": "shimmer"})),
                    metadata: None,
                },
            )
            .await
            .unwrap();

        assert_eq!(updated.state, SessionState::Ready);
    }

    #[tokio::test]
    async fn test_add_conversation_item() {
        let manager = SessionManager::new_in_memory().await.unwrap();

        manager
            .create_session(CreateSessionRequest {
                id: Some("item_test".to_string()),
                config: None,
                metadata: None,
                expires_in_seconds: None,
            })
            .await
            .unwrap();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        manager
            .add_conversation_item(
                "item_test",
                ConversationItem {
                    id: "msg_1".to_string(),
                    session_id: "item_test".to_string(),
                    item_type: "message".to_string(),
                    role: Some("user".to_string()),
                    content: Some(r#"{"text":"Hello"}"#.to_string()),
                    created_at: now,
                },
            )
            .await
            .unwrap();

        let session = manager.get_session("item_test").await.unwrap();
        assert_eq!(session.items.len(), 1);
        assert_eq!(session.items[0].id, "msg_1");
    }

    #[tokio::test]
    async fn test_delete_session() {
        let manager = SessionManager::new_in_memory().await.unwrap();

        manager
            .create_session(CreateSessionRequest {
                id: Some("delete_test".to_string()),
                config: None,
                metadata: None,
                expires_in_seconds: None,
            })
            .await
            .unwrap();

        manager.delete_session("delete_test").await.unwrap();

        let result = manager.get_session("delete_test").await;
        assert!(result.is_err());
    }
}
