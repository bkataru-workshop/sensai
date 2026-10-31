//! Cohere API provider implementation.

mod sse;
mod types;

use async_stream::stream;
use futures::{Stream, StreamExt};
use std::future::Future;
use std::pin::Pin;

use crate::client::error::LLMError;
use crate::client::http::HttpClient;
use crate::client::models::{Message, MessageOptions, StreamEvent};
use crate::client::traits::{LLMProvider, StreamMsgFuture};

/// Cohere API provider.
pub struct CohereProvider {
    api_key: String,
    model: String,
}

impl CohereProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self { api_key, model }
    }

    pub fn model(&self) -> &str {
        &self.model
    }
}

impl LLMProvider for CohereProvider {
    fn send_msg(
        &self,
        client: &HttpClient,
        messages: &[Message],
        options: &MessageOptions,
    ) -> Pin<Box<dyn Future<Output = Result<Message, LLMError>> + Send>> {
        let client = client.clone();
        let api_key = self.api_key.clone();
        let model = options.model.as_deref().unwrap_or(&self.model).to_string();
        let messages = messages.to_vec();
        let options = options.clone();

        Box::pin(async move {
            let body = types::build_request_body(&messages, &options, &model, false)?;
            let headers = types::get_request_headers(&api_key)?;
            let headers_ref: Vec<(&str, &str)> =
                headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
            let url = types::get_api_url();

            let response = client.post(url, &headers_ref, &body).await?;
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
        let api_key = self.api_key.clone();
        let model = options.model.as_deref().unwrap_or(&self.model).to_string();
        let messages = messages.to_vec();
        let options = options.clone();

        Box::pin(async move {
            let body = types::build_request_body(&messages, &options, &model, true)?;
            let headers = types::get_request_headers(&api_key)?;
            let headers_ref: Vec<(&str, &str)> =
                headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
            let url = types::get_api_url();

            let byte_stream = client.post_stream(url, &headers_ref, &body).await?;

            let event_stream = stream! {
                let mut buffer = String::new();
                let mut byte_stream = byte_stream;
                let mut state = sse::StreamState::default();
                let mut message_started = false;

                while let Some(chunk_result) = byte_stream.next().await {
                    match chunk_result {
                        Ok(bytes) => {
                            match std::str::from_utf8(&bytes) {
                                Ok(text) => buffer.push_str(text),
                                Err(_) => {
                                    yield Err(LLMError::new("SSE_DECODE_ERROR", "Invalid UTF-8 in stream"));
                                    return;
                                }
                            }

                            let (events, remaining) = sse::parse_sse_chunk(&buffer);
                            buffer = remaining;

                            for sse_event in events {
                                match sse::parse_stream_event(&sse_event, &mut state) {
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
                    Box<dyn Stream<Item = Result<StreamEvent, LLMError>> + Send>,
                >)
        })
    }
}
