//! LLM client with provider-agnostic interface.

/// Error types for LLM operations.
pub mod error;
/// HTTP client with TLS and retry logic.
pub mod http;
/// Message and request/response models.
pub mod models;
/// LLM provider implementations (Anthropic, OpenAI).
pub mod providers;
/// Provider trait definition.
pub mod traits;

use futures::Stream;
use std::pin::Pin;

use error::LLMError;
use http::HttpClient;
use models::{Message, MessageOptions, StreamEvent};
use traits::LLMProvider;

/// This is the main LLM Client.
pub struct LLMClient {
    http_client: HttpClient,
    provider: Box<dyn LLMProvider + Send + Sync>,
}

impl LLMClient {
    /// Create a new LLM client with the specified provider.
    pub fn new(provider: Box<dyn LLMProvider + Send + Sync>) -> Result<Self, LLMError> {
        Ok(Self {
            http_client: HttpClient::new()?,
            provider,
        })
    }

    /// Send a message and wait for the complete response.
    pub async fn send_message(
        &self,
        messages: &[Message],
        options: &MessageOptions,
    ) -> Result<Message, LLMError> {
        self.provider
            .send_msg(&self.http_client, messages, options)
            .await
    }

    /// Send a message and receive a stream of response events.
    pub async fn send_message_stream(
        &self,
        messages: &[Message],
        options: &MessageOptions,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent, LLMError>> + Send>>, LLMError> {
        self.provider
            .send_msg_stream(&self.http_client, messages, options)
            .await
    }
}
