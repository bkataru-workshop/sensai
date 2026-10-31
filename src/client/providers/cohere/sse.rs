//! Server-Sent Events (SSE) parser for Cohere streaming responses.

use serde_json::Value;

use crate::client::error::LLMError;
use crate::client::models::{ContentBlockType, StreamEvent, Usage};

const ERROR_SSE_PARSE: &str = "SSE_PARSE_ERROR";

/// Parsed SSE event with data payload.
#[derive(Debug)]
pub struct SseEvent {
    pub data: String,
}

/// State tracker for streaming content accumulation.
#[derive(Debug, Default, Clone)]
pub struct StreamState {
    /// Current content block index.
    pub block_index: usize,
    /// Whether a text block is currently open.
    pub text_block_started: bool,
    /// Indices of open tool call blocks.
    pub pending_tool_block_indices: Vec<usize>,
    /// Whether stream has ended.
    pub finished: bool,
}

/// Parse SSE lines from a buffer, returning complete events and the remaining buffer.
///
/// The remainder is returned verbatim to avoid losing bytes across chunk boundaries.
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
            data_lines.push(value.strip_prefix(' ').unwrap_or(value).to_string())
        }
        // Ignore event:, id:, retry:, and comment lines.

        if has_newline {
            idx = line_end + 1;
        } else {
            break;
        }
    }

    (events, buffer[completed_through..].to_string())
}

/// Parse a Cohere SSE event into StreamEvents.
pub fn parse_stream_event(
    sse: &SseEvent,
    state: &mut StreamState,
) -> Result<Vec<StreamEvent>, LLMError> {
    let json = parse_json(&sse.data)?;

    // Check for error response.
    if let Some(error) = json.get("error") {
        let msg = error["message"].as_str().unwrap_or("Unknown error");
        return Err(LLMError::new("COHERE_ERROR", msg));
    }

    let mut events = Vec::new();
    let event_type = json["type"].as_str().unwrap_or("");

    match event_type {
        "message-start" => {}

        "content-start" if !state.text_block_started => {
            events.push(StreamEvent::ContentBlockStart {
                index: state.block_index,
                block_type: ContentBlockType::Text,
            });
            state.text_block_started = true;
        }

        "content-delta" => {
            if let Some(text) = json
                .pointer("delta/message/content/text")
                .and_then(|v| v.as_str())
                && !text.is_empty()
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
                    text: text.to_string(),
                });
            }
        }

        "content-end" if state.text_block_started => {
            events.push(StreamEvent::ContentBlockStop {
                index: state.block_index,
            });
            state.block_index += 1;
            state.text_block_started = false;
        }

        "tool-call-start" => {
            if let Some(tool_call) = json.pointer("/delta/tool_call") {
                let id = tool_call["id"].as_str().unwrap_or_default().to_string();
                let name = tool_call["function"]["name"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string();

                // Close open text block first.
                if state.text_block_started {
                    events.push(StreamEvent::ContentBlockStop {
                        index: state.block_index,
                    });
                    state.block_index += 1;
                    state.text_block_started = false;
                }

                let block_idx = state.block_index + state.pending_tool_block_indices.len();
                events.push(StreamEvent::ContentBlockStart {
                    index: block_idx,
                    block_type: ContentBlockType::ToolUse { id, name },
                });
                state.pending_tool_block_indices.push(block_idx);
            }
        }

        "tool-call-delta" => {
            if let Some(args) = json
                .pointer("/delta/tool_call/function/arguments")
                .and_then(|v| v.as_str())
                && !args.is_empty()
            {
                if let Some(&block_idx) = state.pending_tool_block_indices.last() {
                    events.push(StreamEvent::InputJsonDelta {
                        index: block_idx,
                        json: args.to_string(),
                    });
                }
            }
        }

        "tool-call-end" => {
            if let Some(block_idx) = state.pending_tool_block_indices.pop() {
                events.push(StreamEvent::ContentBlockStop { index: block_idx });
            }
        }

        "message-end" => {
            state.finished = true;

            // Close any open blocks.
            if state.text_block_started {
                events.push(StreamEvent::ContentBlockStop {
                    index: state.block_index,
                });
                state.text_block_started = false;
            }
            for &block_idx in &state.pending_tool_block_indices {
                events.push(StreamEvent::ContentBlockStop { index: block_idx });
            }
            state.pending_tool_block_indices.clear();

            let usage = json.pointer("/delta/usage/billed_units").map(|u| Usage {
                input_tokens: u["input_tokens"].as_u64().unwrap_or(0) as u32,
                output_tokens: u["output_tokens"].as_u64().unwrap_or(0) as u32,
            });

            let stop_reason = json
                .pointer("/delta/finish_reason")
                .and_then(|r| r.as_str())
                .map(|r| match r {
                    "COMPLETE" => "end_turn".to_string(),
                    "MAX_TOKENS" => "max_tokens".to_string(),
                    "TOOL_CALL" => "tool_use".to_string(),
                    other => other.to_string(),
                });

            events.push(StreamEvent::MessageDelta { stop_reason, usage });
        }

        _ => {}
    }

    Ok(events)
}

fn parse_json(data: &str) -> Result<Value, LLMError> {
    serde_json::from_str(data)
        .map_err(|e| LLMError::new(ERROR_SSE_PARSE, format!("Invalid JSON: {}", e)))
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
    fn test_parse_text_delta() {
        let events = parse_stream_event(
            &sse(r#"{type":"content-delta","delta":{"message":{"content":{"text":"Hello"}}}}"#),
            &mut StreamState::default(),
        )
        .unwrap();

        // Should emit ContentBlockStart + TextDelta (text block wasn't started).
        assert_eq!(events.len(), 2);
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
    }

    #[test]
    fn test_parse_text_delta_block_already_open() {
        let mut state = StreamState {
            text_block_started: true,
            ..Default::default()
        };
        let events = parse_stream_event(
            &sse(r#"{"type":"content-delta","delta":{"message":{"content":{"text":"world"}}}}"#),
            &mut state,
        )
        .unwrap();

        // Only TextDelta, no extra ContentBlockStart.
        assert_eq!(
            events,
            vec![StreamEvent::TextDelta {
                index: 0,
                text: "world".to_string(),
            }]
        );
    }

    #[test]
    fn test_parse_tool_call_start() {
        let events = parse_stream_event(
              &sse(r#"{"type":"tool-call-start","delta":{"tool_call":{"id":"tc_1","function":{"name":"search"}}}}"#),
              &mut StreamState::default(),
        )
        .unwrap();

        assert_eq!(
            events,
            vec![StreamEvent::ContentBlockStart {
                index: 0,
                block_type: ContentBlockType::ToolUse {
                    id: "tc_1".to_string(),
                    name: "search".to_string(),
                },
            }]
        );
    }

    #[test]
    fn test_parse_tool_call_delta() {
        let mut state = StreamState {
            pending_tool_block_indices: vec![1],
            ..Default::default()
        };
        let events = parse_stream_event(
            &sse(r#"{"type":"tool-call-delta","delta":{"tool_call":{"function":{"arguments":"{\"q\":"}}}}"#),
            &mut state,
        )
        .unwrap();

        assert_eq!(
            events,
            vec![StreamEvent::InputJsonDelta {
                index: 1,
                json: "{\"q\":".to_string(),
            }]
        );
    }

    #[test]
    fn test_parse_tool_call_end() {
        let mut state = StreamState {
            pending_tool_block_indices: vec![2],
            ..Default::default()
        };
        let events = parse_stream_event(&sse(r#"{"type":"tool-call-end"}"#), &mut state).unwrap();

        assert_eq!(events, vec![StreamEvent::ContentBlockStop { index: 2 }]);
        assert!(state.pending_tool_block_indices.is_empty());
    }

    #[test]
    fn test_parse_message_end() {
        let events = parse_stream_event(
            &sse(r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE","usage":{"billed_units":{"input_tokens":10,"output_tokens":20}}}}"#),
            &mut StreamState::default(),
        )
        .unwrap();

        assert_eq!(
            events,
            vec![StreamEvent::MessageDelta {
                stop_reason: Some("end_turn".to_string()),
                usage: Some(Usage {
                    input_tokens: 10,
                    output_tokens: 20,
                }),
            }]
        );
    }

    #[test]
    fn test_parse_message_end_closes_open_block() {
        let mut state = StreamState {
            text_block_started: true,
            block_index: 0,
            pending_tool_block_indices: vec![1],
            finished: false,
        };
        let events = parse_stream_event(
            &sse(r#"{"type":"message-end","delta":{"finish_reason":"TOOL_CALL"}}"#),
            &mut state,
        )
        .unwrap();

        // Should close text block (0), tool block (1), then emit MessageDelta.
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], StreamEvent::ContentBlockStop { index: 0 });
        assert_eq!(events[1], StreamEvent::ContentBlockStop { index: 1 });
        assert_eq!(
            events[2],
            StreamEvent::MessageDelta {
                stop_reason: Some("tool_use".to_string()),
                usage: None,
            }
        );
        assert!(state.finished);
    }

    #[test]
    fn test_parse_error() {
        let err = parse_stream_event(
            &sse(r#"{"error":{"message":"Invalid API key"}}"#),
            &mut StreamState::default(),
        )
        .unwrap_err();

        assert_eq!(err.error_code, "COHERE_ERROR");
    }

    #[test]
    fn test_parse_unknown_event() {
        let events = parse_stream_event(
            &sse(r#"{"type":"something-new"}"#),
            &mut StreamState::default(),
        )
        .unwrap();

        assert!(events.is_empty());
    }
}
