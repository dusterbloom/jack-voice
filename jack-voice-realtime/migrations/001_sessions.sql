-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'connecting',
    config TEXT,
    conversation_id TEXT,
    metadata TEXT,
    expires_at INTEGER
);

-- Conversations table
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Conversation items table
CREATE TABLE IF NOT EXISTS conversation_items (
    id TEXT PRIMARY KEY NOT NULL,
    session_id TEXT NOT NULL,
    item_type TEXT NOT NULL,
    role TEXT,
    content TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_sessions_conversation ON sessions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversation_items_session ON conversation_items(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
