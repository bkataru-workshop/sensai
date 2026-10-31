use super::error::LLMError;
use super::http::HttpClient;
use super::models::{Message, MessageOptions, StreamEvent};
use futures::Stream;
use std::future::Future;
use std::pin::Pin;

/// A boxed stream of LLM streaming events.
pub type StreamEventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, LLMError>> + Send>>;

/// A boxed future that resolves to a stream of LLM events.
pub type StreamMsgFuture =
    Pin<Box<dyn Future<Output = Result<StreamEventStream, LLMError>> + Send>>;

/// Provider interface for LLM APIs
///
/// Implement this trait to add support for new LLM providers.
pub trait LLMProvider {
    /// Send a message to the LLM.
    /// Returns the assistant's response message or an error.
    fn send_msg(
        &self,
        client: &HttpClient,
        messages: &[Message],
        options: &MessageOptions,
    ) -> Pin<Box<dyn Future<Output = Result<Message, LLMError>> + Send>>;

    /// Send a streaming message to the LLM.
    /// Returns a stream of events as they arrive from the API.
    /// TODO: why are these fields prefixed with '_'? are they unused or?
    /// need to look into this
    fn send_msg_stream(
        &self,
        _client: &HttpClient,
        _messages: &[Message],
        _options: &MessageOptions,
    ) -> StreamMsgFuture {
        Box::pin(async {
            Err(LLMError::new(
                "NOT_IMPLEMENTED",
                "Streaming not supported for this provider",
            ))
        })
    }
}
