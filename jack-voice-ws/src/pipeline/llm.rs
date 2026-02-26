//! LLM (Large Language Model) connector
//! 
//! Provides OpenAI-compatible API for connecting to various LLM backends

use serde::{Deserialize, Serialize};
use reqwest::Client;
use anyhow::Result;
use futures::Stream;
use async_stream::stream;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            api_key: None,
            model: "llama3.2".to_string(),
        }
    }
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Chat request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub stream: bool,
}

/// Chat response choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

/// Chat response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<ChatChoice>,
}

/// LLM client for OpenAI-compatible APIs
pub struct LlmClient {
    client: Client,
    config: LlmConfig,
}

impl LlmClient {
    pub fn new(config: LlmConfig) -> Self {
        Self {
            client: Client::new(),
            config,
        }
    }

    pub fn ollama() -> Self {
        Self::new(LlmConfig::default())
    }

    /// Send chat request
    pub async fn chat(&self, messages: Vec<ChatMessage>) -> Result<String> {
        let request = ChatRequest {
            model: self.config.model.clone(),
            messages,
            stream: false,
        };

        let url = format!("{}/api/chat", self.config.base_url);
        
        let response = self.client
            .post(&url)
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            anyhow::bail!("LLM request failed: {}", response.status());
        }

        let chat_response: ChatResponse = response.json().await?;

        Ok(chat_response
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default())
    }

    /// Stream chat (returns receiver)
    pub fn chat_streaming(&self, messages: Vec<ChatMessage>) -> impl Stream<Item = Result<String>> + '_ {
        let request = ChatRequest {
            model: self.config.model.clone(),
            messages,
            stream: true,
        };

        let url = format!("{}/api/chat", self.config.base_url);
        let client = self.client.clone();

        stream! {
            let response = client
                .post(&url)
                .json(&request)
                .send()
                .await;

            match response {
                Ok(resp) => {
                    use futures::StreamExt;
                    let mut stream = resp.bytes_stream();

                    let mut buffer = String::new();
                    while let Some(chunk) = stream.next().await {
                        match chunk {
                            Ok(bytes) => {
                                if let Ok(text) = String::from_utf8(bytes.to_vec()) {
                                    buffer.push_str(&text);
                                    
                                    // Parse SSE lines
                                    for line in buffer.lines() {
                                        if let Some(data) = line.strip_prefix("data: ") {
                                            if data == "[DONE]" {
                                                return;
                                            }
                                            
                                            if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
                                                if let Some(content) = json.get("message")
                                                    .and_then(|m| m.get("content"))
                                                    .and_then(|c| c.as_str())
                                                {
                                                    yield Ok(content.to_string());
                                                }
                                            }
                                        }
                                    }
                                    buffer.clear();
                                }
                            }
                            Err(e) => {
                                yield Err(anyhow::anyhow!("Stream error: {}", e));
                            }
                        }
                    }
                }
                Err(e) => {
                    yield Err(anyhow::anyhow!("Request error: {}", e));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_config_default() {
        let config = LlmConfig::default();
        
        assert_eq!(config.base_url, "http://localhost:11434");
        assert_eq!(config.model, "llama3.2");
    }

    #[test]
    fn test_chat_message() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };
        
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""role":"user""#));
        assert!(json.contains(r#""content":"Hello""#));
    }

    #[test]
    fn test_chat_request() {
        let request = ChatRequest {
            model: "llama3".to_string(),
            messages: vec![
                ChatMessage { role: "system".to_string(), content: "You are helpful".to_string() },
                ChatMessage { role: "user".to_string(), content: "Hi".to_string() },
            ],
            stream: false,
        };
        
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains(r#""model":"llama3""#));
        assert!(json.contains(r#""stream":false"#));
    }

    #[test]
    fn test_llm_client_creation() {
        let client = LlmClient::ollama();
        // Just verify it can be created
        assert!(client.config.model == "llama3.2");
    }
}
