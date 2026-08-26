//! Amazon Bedrock API provider implementation.
//!
//! Uses the Converse API for chat completions with AWS SigV4 authentication.

mod signing;
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

// == Provider ==

/// AWS Bedrock credentials.
#[derive(Clone)]
pub struct BedrockCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

impl BedrockCredentials {
    pub fn new(access_key_id: impl Into<String>, secret_access_key: impl Into<String>) -> Self {
        Self {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token: None,
        }
    }

    pub fn with_session_token(
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        session_token: impl Into<String>,
    ) -> Self {
        Self {
            access_key_id: access_key_id.into(),
            secret_access_key: secret_access_key.into(),
            session_token: Some(session_token.into()),
        }
    }
}

/// Amazon Bedrock API provider using the Converse API.
pub struct BedrockProvider {
    credentials: BedrockCredentials,
    region: String,
    model: String,
}

impl BedrockProvider {
    pub fn new(credentials: BedrockCredentials, region: String, model: String) -> Self {
        Self {
            credentials,
            region,
            model,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn region(&self) -> &str {
        &self.region
    }
}

impl LlmProvider for BedrockProvider {
    fn send_msg(
        &self,
        client: &HttpClient,
        messages: &[Message],
        options: &MessageOptions,
    ) -> Pin<Box<dyn Future<Output = Result<Message, LlmError>> + Send>> {
        let client = client.clone();
        let credentials = self.credentials.clone();
        let region = self.region.clone();
        let model = options.model.as_deref().unwrap_or(&self.model).to_string();
        let messages = messages.to_vec();
        let options = options.clone();

        Box::pin(async move {
            let body = types::build_request_body(&messages, &options)?;
            let url = types::get_converse_url(&region, &model);
            let headers = signing::sign_request(&credentials, &region, "POST", &url, &body, false)?;
            let headers_ref: Vec<(&str, &str)> = headers
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();

            let response = client.post(&url, &headers_ref, &body).await?;
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
        let credentials = self.credentials.clone();
        let region = self.region.clone();
        let model = options.model.as_deref().unwrap_or(&self.model).to_string();
        let messages = messages.to_vec();
        let options = options.clone();

        Box::pin(async move {
            let body = types::build_request_body(&messages, &options)?;
            let url = types::get_converse_stream_url(&region, &model);
            let headers = signing::sign_request(&credentials, &region, "POST", &url, &body, true)?;
            let headers_ref: Vec<(&str, &str)> = headers
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();

            let byte_stream = client.post_stream(&url, &headers_ref, &body).await?;

            let event_stream = stream! {
                let mut buffer = Vec::new();
                let mut byte_stream = byte_stream;
                let mut state = sse::StreamState::default();
                let mut message_started = false;

                while let Some(chunk_result) = byte_stream.next().await {
                    match chunk_result {
                        Ok(bytes) => {
                            buffer.extend_from_slice(&bytes);

                            let (events, remaining) = sse::parse_event_stream(&buffer);
                            buffer = remaining;

                            for event in events {
                                match sse::parse_stream_event(&event, &mut state) {
                                    Ok(stream_events) => {
                                        if !message_started && !stream_events.is_empty() {
                                            message_started = true;
                                            yield Ok(StreamEvent::MessageStart {
                                                message_id: String::new(),
                                                model: model.clone(),
                                            });
                                        }
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

                if message_started {
                    yield Ok(StreamEvent::MessageStop);
                }
            };

            Ok(Box::pin(event_stream)
                as Pin<
                    Box<dyn Stream<Item = Result<StreamEvent, LlmError>> + Send>,
                >)
        })
    }
}
