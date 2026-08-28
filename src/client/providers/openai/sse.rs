//! Server-Sent Events (SSE) parser for OpenAI streaming responses.

use serde_json::Value;

use crate::client::error::LlmError;
use crate::client::models::{ContentBlockType, StreamEvent, Usage};

const ERROR_SSE_PARSE: &str = "SSE_PARSE_ERROR";
const STREAM_DONE_MARKER: &str = "[DONE]";

/// Parsed SSE event.
#[derive(Debug)]
pub struct SseEvent {
    pub data: String,
}

/// State tracker for OpenAI streaming (tool call accumulation).
#[derive(Debug, Default, Clone)]
pub struct StreamState {
    /// Current content block index.
    pub block_index: usize,
    /// Whether a text block is currently open.
    pub text_block_started: bool,
    /// Pending tool calls being accumulated.
    pub pending_tool_calls: Vec<PendingToolCall>,
    /// Model name from the first chunk.
    pub model: Option<String>,
    /// Message ID from the first chunk.
    pub message_id: Option<String>,
}

/// Tracks a tool call being accumulated across streaming chunks.
#[derive(Debug, Clone)]
pub struct PendingToolCall {
    /// Tool call index in the response delta.
    pub index: usize,
    /// Content block index for our StreamEvents.
    pub block_index: usize,
    /// Tool call ID.
    pub id: String,
    /// Function name.
    pub name: String,
    /// Whether the start event has been emitted.
    pub started: bool,
}

/// Parse SSE lines from a buffer, returning complete events and the remaining buffer.
pub fn parse_sse_chunk(buffer: &str) -> (Vec<SseEvent>, String) {
    let mut events = Vec::new();
    let mut data_lines: Vec<String> = Vec::new();
    let mut completed_through = 0usize;

    let mut idx = 0usize;
    while idx < buffer.len() {
        let line_end = match buffer[idx..].find('\n') {
            Some(rel) => idx + rel,
            None => buffer.len(),
        };
        let has_newline = line_end < buffer.len();
        let line = buffer[idx..line_end].trim_end_matches('\r');

        if line.is_empty() {
            if has_newline && !data_lines.is_empty() {
                events.push(SseEvent {
                    data: data_lines.join("\n"),
                });
                data_lines.clear();
                completed_through = line_end + 1;
            }
        } else if let Some(value) = line.strip_prefix("data:") {
            data_lines.push(value.strip_prefix(' ').unwrap_or(value).to_string());
        }

        if has_newline {
            idx = line_end + 1;
        } else {
            break;
        }
    }

    (events, buffer[completed_through..].to_string())
}

/// Parse an OpenAI SSE event into StreamEvents.
pub fn parse_stream_event(
    sse: &SseEvent,
    state: &mut StreamState,
) -> Result<Vec<StreamEvent>, LlmError> {
    if sse.data == STREAM_DONE_MARKER {
        return Ok(vec![StreamEvent::MessageStop]);
    }

    let json = parse_json(&sse.data)?;

    // Error response.
    if let Some(error) = json.get("error") {
        let error_type = error["type"].as_str().unwrap_or("api_error");
        let error_msg = error["message"].as_str().unwrap_or("Unknown error");
        return Err(LlmError::new(error_type, error_msg));
    }

    let mut events = Vec::new();

    // Extract metadata from first chunk.
    if state.message_id.is_none() {
        if let Some(id) = json["id"].as_str() {
            state.message_id = Some(id.to_string());
        }
        if let Some(model) = json["model"].as_str() {
            state.model = Some(model.to_string());
        }
        events.push(StreamEvent::MessageStart {
            message_id: state.message_id.clone().unwrap_or_default(),
            model: state.model.clone().unwrap_or_default(),
        });
    }

    // Process choices.
    if let Some(choices) = json["choices"].as_array()
        && let Some(choice) = choices.first()
    {
        let delta = &choice["delta"];

        // Text content.
        if let Some(content) = delta["content"].as_str()
            && !content.is_empty()
        {
            if !state.text_block_started {
                events.push(StreamEvent::ContentBlockStart {
                    index: state.block_index,
                    block_type: ContentBlockType::Text,
                });
                state.text_block_started = true;
            }
            events.push(StreamEvent::TextDelta {
                index: state.block_index,
                text: content.to_string(),
            });
        }

        // Tool calls.
        if let Some(tool_calls) = delta["tool_calls"].as_array() {
            for tc in tool_calls {
                let tc_index = tc["index"].as_u64().unwrap_or(0) as usize;

                // Find or create pending tool call.
                let pending_idx = if let Some(pos) = state
                    .pending_tool_calls
                    .iter()
                    .position(|p| p.index == tc_index)
                {
                    pos
                } else {
                    // Close text block if open.
                    if state.text_block_started {
                        events.push(StreamEvent::ContentBlockStop {
                            index: state.block_index,
                        });
                        state.block_index += 1;
                        state.text_block_started = false;
                    }

                    let block_idx = state.block_index + state.pending_tool_calls.len();
                    state.pending_tool_calls.push(PendingToolCall {
                        index: tc_index,
                        block_index: block_idx,
                        id: String::new(),
                        name: String::new(),
                        started: false,
                    });
                    state.pending_tool_calls.len() - 1
                };

                let pending = &mut state.pending_tool_calls[pending_idx];

                if let Some(id) = tc["id"].as_str() {
                    pending.id = id.to_string();
                }
                if let Some(name) = tc["function"]["name"].as_str() {
                    pending.name = name.to_string();
                }

                // Emit start once we have id and name.
                if !pending.started && !pending.id.is_empty() && !pending.name.is_empty() {
                    events.push(StreamEvent::ContentBlockStart {
                        index: pending.block_index,
                        block_type: ContentBlockType::ToolUse {
                            id: pending.id.clone(),
                            name: pending.name.clone(),
                        },
                    });
                    pending.started = true;
                }

                // Emit argument delta.
                if pending.started
                    && let Some(args) = tc["function"]["arguments"].as_str()
                    && !args.is_empty()
                {
                    events.push(StreamEvent::InputJsonDelta {
                        index: pending.block_index,
                        json: args.to_string(),
                    });
                }
            }
        }

        // Finish reason.
        if let Some(finish_reason) = choice["finish_reason"].as_str() {
            if state.text_block_started {
                events.push(StreamEvent::ContentBlockStop {
                    index: state.block_index,
                });
                state.text_block_started = false;
            }

            for pending in &state.pending_tool_calls {
                if pending.started {
                    events.push(StreamEvent::ContentBlockStop {
                        index: pending.block_index,
                    });
                }
            }

            let stop_reason = Some(match finish_reason {
                "stop" => "end_turn".to_string(),
                "length" => "max_tokens".to_string(),
                "tool_calls" => "tool_use".to_string(),
                "content_filter" => "content_filter".to_string(),
                other => other.to_string(),
            });

            let usage = json.get("usage").map(|u| Usage {
                input_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                output_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
            });

            events.push(StreamEvent::MessageDelta { stop_reason, usage });
        }
    }

    Ok(events)
}

fn parse_json(data: &str) -> Result<Value, LlmError> {
    serde_json::from_str(data)
        .map_err(|e| LlmError::new(ERROR_SSE_PARSE, format!("Invalid JSON: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse(data: &str) -> SseEvent {
        SseEvent {
            data: data.to_string(),
        }
    }

    #[test]
    fn test_parse_sse_chunk() {
        let chunk = "data: {\"test\":true}\n\ndata: {\"test\":false}\n\n";
        let (events, remaining) = parse_sse_chunk(chunk);

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].data, "{\"test\":true}");
        assert_eq!(events[1].data, "{\"test\":false}");
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_sse_chunk_crlf() {
        let chunk = "data: {\"ok\":true}\r\n\r\n";
        let (events, remaining) = parse_sse_chunk(chunk);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].data, "{\"ok\":true}");
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_incomplete_chunk() {
        let chunk = "data: {\"test\":true}";
        let (events, remaining) = parse_sse_chunk(chunk);

        assert!(events.is_empty());
        assert_eq!(remaining, chunk);
    }

    #[test]
    fn test_parse_done_marker() {
        let events =
            parse_stream_event(&sse(STREAM_DONE_MARKER), &mut StreamState::default()).unwrap();
        assert_eq!(events, vec![StreamEvent::MessageStop]);
    }

    #[test]
    fn test_parse_first_chunk() {
        let mut state = StreamState::default();
        let events = parse_stream_event(
            &sse(r#"{"id":"chatcmpl-123","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#),
            &mut state,
        )
        .unwrap();

        assert_eq!(
            events[0],
            StreamEvent::MessageStart {
                message_id: "chatcmpl-123".to_string(),
                model: "gpt-4".to_string(),
            }
        );
        assert_eq!(state.message_id, Some("chatcmpl-123".to_string()));
        assert_eq!(state.model, Some("gpt-4".to_string()));
    }

    #[test]
    fn test_parse_content_delta() {
        let mut state = StreamState {
            message_id: Some("chatcmpl-123".to_string()),
            model: Some("gpt-4".to_string()),
            ..Default::default()
        };

        let events = parse_stream_event(
            &sse(r#"{"id":"chatcmpl-123","model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}"#),
            &mut state,
        )
        .unwrap();

        assert_eq!(
            events[0],
            StreamEvent::ContentBlockStart {
                index: 0,
                block_type: ContentBlockType::Text,
            }
        );
        assert_eq!(
            events[1],
            StreamEvent::TextDelta {
                index: 0,
                text: "Hello".to_string(),
            }
        );
        assert!(state.text_block_started);
    }

    #[test]
    fn test_parse_tool_call() {
        let mut state = StreamState {
            message_id: Some("chatcmpl-123".to_string()),
            model: Some("gpt-4".to_string()),
            ..Default::default()
        };

        let events = parse_stream_event(
            &sse(r#"{"id":"chatcmpl-123","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":"{\"loc"}}]},"finish_reason":null}]}"#),
            &mut state,
        )
        .unwrap();

        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::ContentBlockStart {
                block_type: ContentBlockType::ToolUse { id, name },
                ..
            } if id == "call_abc" && name == "get_weather"
        )));
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::InputJsonDelta { json, .. } if json == "{\"loc"
        )));
    }

    #[test]
    fn test_parse_finish_reason_stop() {
        let mut state = StreamState {
            message_id: Some("chatcmpl-123".to_string()),
            model: Some("gpt-4".to_string()),
            text_block_started: true,
            ..Default::default()
        };

        let events = parse_stream_event(
            &sse(r#"{"id":"chatcmpl-123","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#),
            &mut state,
        )
        .unwrap();

        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::ContentBlockStop { index: 0 }))
        );
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::MessageDelta { stop_reason: Some(r), .. } if r == "end_turn"
        )));
    }

    #[test]
    fn test_parse_finish_reason_tool_calls() {
        let mut state = StreamState {
            message_id: Some("x".to_string()),
            model: Some("gpt-4".to_string()),
            pending_tool_calls: vec![PendingToolCall {
                index: 0,
                block_index: 0,
                id: "call_1".to_string(),
                name: "search".to_string(),
                started: true,
            }],
            ..Default::default()
        };

        let events = parse_stream_event(
            &sse(r#"{"id":"x","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#),
              &mut state,
        )
        .unwrap();

        assert!(
            events
                .iter()
                .any(|e| matches!(e, StreamEvent::ContentBlockStop { index: 0 }))
        );
        assert!(events.iter().any(|e| matches!(
            e,
            StreamEvent::MessageDelta { stop_reason: Some(r), .. } if r == "tool_use"
        )));
    }

    #[test]
    fn test_parse_error() {
        let err = parse_stream_event(
            &sse(r#"{"error":{"type":"invalid_request_error","message":"Invalid API key"}}"#),
            &mut StreamState::default(),
        )
        .unwrap_err();

        assert_eq!(err.error_code, "invalid_request_error");
    }
}
