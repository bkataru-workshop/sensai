//! OpenAI API provider implementation.
//!
//! Also supports OpenAI-compatible APIs (Groq, Together, Fireworks, etc.)
//! and Azure OpenAI.

mod sse;
mod types;

use async_stream::stream;
use futures::{Stream, StreamExt};
use std::future::Future;
use std::pin::Pin;

use crate::client::error::LlmError;
use crate::client::http::HttpClient;
use crate::client::models::{Message, MessageOptions, StreamEvent};
use crate::client::traits::{LlmProvider, StreamMsgFuture};

// Azure OpenAI configuration
#[derive(Clone)]
pub struct AzureConfig {
    pub resource: String,
    pub deployment: String,
    pub api_version: String,
}

/// OpenAI API provider.
///
/// Also supports OpenAI-compatible APIs (Groq, Together, Fireworks, etc.)
/// via custom `base_url`, and Azure OpenAI via `AzureConfig`.
pub struct OpenAIProvider {
    api_key: String,
    model: String,
    base_url: Option<String>,
    azure_config: Option<AzureConfig>,
}

impl OpenAIProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            base_url: None,
            azure_config: None,
        }
    }

    /// Create a provider with a custom base URL (for OpenAI-compatible APIs).
    pub fn with_base_url(api_key: String, model: String, base_url: String) -> Self {
        Self {
            api_key,
            model,
            base_url: Some(base_url),
            azure_config: None,
        }
    }

    /// Create an Azure OpenAI provider.
    pub fn azure(
        api_key: String,
        resource: String,
        deployment: String,
        api_version: String,
    ) -> Self {
        Self {
            api_key,
            model: String::new(),
            base_url: None,
            azure_config: Some(AzureConfig {
                resource,
                deployment,
                api_version,
            }),
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn is_azure(&self) -> bool {
        self.azure_config.is_some()
    }

    fn api_url(&self) -> String {
        if let Some(azure) = &self.azure_config {
            types::get_azure_api_url(&azure.resource, &azure.deployment, &azure.api_version)
        } else {
            types::get_api_url_with_base(self.base_url.as_deref())
        }
    }

    fn get_headers(&self) -> Vec<(&'static str, String)> {
        if self.azure_config.is_some() {
            types::get_azure_request_headers(&self.api_key)
        } else {
            types::get_request_headers(&self.api_key)
        }
    }
}

impl LlmProvider for OpenAIProvider {
    fn send_msg(
        &self,
        client: &HttpClient,
        messages: &[Message],
        options: &MessageOptions,
    ) -> Pin<Box<dyn Future<Output = Result<Message, LlmError>> + Send>> {
        let client = client.clone();
        let model = self.model.clone();
        let api_url = self.api_url();
        let headers = self.get_headers();
        let messages = messages.to_vec();
        let options = options.clone();

        Box::pin(async move {
            let body = types::build_request_body(&messages, &options, &model, false)?;
            let headers_ref: Vec<(&str, &str)> =
                headers.iter().map(|(k, v)| (*k, v.as_str())).collect();

            let response = client.post(&api_url, &headers_ref, &body).await?;
            types::parse_response(&response)
        })
    }

    fn send_msg_stream(
        &self,
        client: &HttpClient,
        messages: &[Message],
        options: &MessageOptions,
    ) -> StreamMsgFuture {
        let client = client.clone();
        let model = self.model.clone();
        let api_url = self.api_url();
        let headers = self.get_headers();
        let messages = messages.to_vec();
        let options = options.clone();

        Box::pin(async move {
            let body = types::build_request_body(&messages, &options, &model, true)?;
            let headers_ref: Vec<(&str, &str)> =
                headers.iter().map(|(k, v)| (*k, v.as_str())).collect();

            let byte_stream = client.post_stream(&api_url, &headers_ref, &body).await?;

            let event_stream = stream! {
                let mut buffer = String::new();
                let mut byte_stream = byte_stream;
                let mut state = sse::StreamState::default();

                while let Some(chunk_result) = byte_stream.next().await {
                    match chunk_result {
                        Ok(bytes) => {
                            match std::str::from_utf8(&bytes) {
                                Ok(text) => buffer.push_str(text),
                                Err(_) => {
                                    yield Err(LlmError::new("SSE_DECODE_ERROR", "Invalid UTF-8 in stream"));
                                    return;
                                }
                            }

                            let (events, remaining) = sse::parse_sse_chunk(&buffer);
                            buffer = remaining;

                            for sse_event in events {
                                match sse::parse_stream_event(&sse_event, &mut state) {
                                    Ok(stream_events) => {
                                        for stream_event in stream_events {
                                            yield Ok(stream_event);
                                        }
                                    }
                                    Err(e) => {
                                        yield Err(e);
                                        return;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            yield Err(e);
                            return;
                        }
                    }
                }
            };

            Ok(Box::pin(event_stream)
                as Pin<
                    Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>,
                >)
        })
    }
}
