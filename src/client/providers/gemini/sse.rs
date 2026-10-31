//! Server-Sent Events (SSE) parser for Gemini streaming responses.

use serde_json::Value;

use crate::client::error::LLMError;
use crate::client::models::{ContentBlockType, StreamEvent, Usage};

const ERROR_SSE_PARSE: &str = "SSE_PARSE_ERROR";

/// Parsed SSE event.
#[derive(Debug)]
pub struct SseEvent {
    pub data: String,
}

/// State tracker for streaming to handle indices and function call accummulation.
#[derive(Debug, Default, Clone)]
pub struct StreamState {
    /// Number of content blocks seen so far.
    pub block_count: usize,
    /// Whether we've seen at least one content block.
    pub has_content: bool,
    /// Pending function call being accumulated across chunks.
    pub pending_function_call: Option<PendingFunctionCall>,
}

/// Tracks a function call being accumulated across streaming chunks.
#[derive(Debug, Clone)]
pub struct PendingFunctionCall {
    pub index: usize,
    pub name: String,
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
        // Ignore event:, id:, retry:, and comment lines.

        if has_newline {
            idx = line_end + 1;
        } else {
            break;
        }
    }

    (events, buffer[completed_through..].to_string())
}

/// Parse a Gemini SSE event into StreamEvents.
pub fn parse_stream_event(
    sse: &SseEvent,
    state: &mut StreamState,
) -> Result<Vec<StreamEvent>, LLMError> {
    let json = parse_json(&sse.data)?;

    // Check for error response.
    if let Some(error) = json.get("error") {
        let code = error["code"].as_i64().unwrap_or(0);
        let msg = error["message"].as_str().unwrap_or("Unknown error");
        return Err(LLMError::new(format!("GEMINI_ERROR_{}", code), msg));
    }

    let mut events = Vec::new();

    if let Some(candidates_array) = json["candidates"].as_array()
        && let Some(candidate) = candidates_array.first()
    {
        if let Some(parts_array) = candidate["content"]["parts"].as_array() {
            for part in parts_array {
                let index = state.block_count;

                if let Some(text) = part["text"].as_str()
                    && !text.is_empty()
                {
                    // Finalize any pending function call before text.
                    if let Some(pending) = state.pending_function_call.take() {
                        events.push(StreamEvent::ContentBlockStop {
                            index: pending.index,
                        });
                    }

                    events.push(StreamEvent::ContentBlockStart {
                        index,
                        block_type: ContentBlockType::Text,
                    });
                    events.push(StreamEvent::TextDelta {
                        index,
                        text: text.to_string(),
                    });

                    state.has_content = true;
                    state.block_count = index + 1;
                } else if let Some(function_call) = part.get("functionCall") {
                    let name = function_call["name"].as_str().unwrap_or_default();
                    let args_json = &function_call["args"];

                    // Continuation of a pending function call with the same name.
                    if let Some(ref pending) = state.pending_function_call {
                        if pending.name == name {
                            events.push(StreamEvent::InputJsonDelta {
                                index: pending.index,
                                json: args_json.to_string(),
                            });

                            // If args are now complete, finalize.
                            if is_complete_json(&args_json.to_string()) {
                                let pending = state.pending_function_call.take().unwrap();
                                events.push(StreamEvent::ContentBlockStop {
                                    index: pending.index,
                                });
                            }
                            continue;
                        } else {
                            // Different function - close the previous one.
                            let pending = state.pending_function_call.take().unwrap();
                            events.push(StreamEvent::ContentBlockStop {
                                index: pending.index,
                            });
                        }
                    }

                    // Start a new function call.
                    // Gemini matches function responses by NAME, not unique ID.
                    let id = name.to_string();
                    let args_str = args_json.to_string();

                    events.push(StreamEvent::ContentBlockStart {
                        index,
                        block_type: ContentBlockType::ToolUse {
                            id: id.clone(),
                            name: name.to_string(),
                        },
                    });
                    events.push(StreamEvent::InputJsonDelta {
                        index,
                        json: args_str.clone(),
                    });

                    if is_complete_json(&args_str) {
                        events.push(StreamEvent::ContentBlockStop { index });
                    } else {
                        state.pending_function_call = Some(PendingFunctionCall {
                            index,
                            name: name.to_string(),
                        });
                    }

                    state.has_content = true;
                    state.block_count = index + 1;
                }
            }
        }

        // Handle finish reason.
        if let Some(finish_reason) = candidate["finishReason"].as_str() {
            // Finalize any pending function call.
            if let Some(pending) = state.pending_function_call.take() {
                events.push(StreamEvent::ContentBlockStop {
                    index: pending.index,
                });
            }

            // Emit stop for the last text block if not already stopped.
            if state.has_content && state.block_count > 0 {
                let last = state.block_count - 1;
                let already_stopped = events.iter().any(
                    |e| matches!(e, StreamEvent::ContentBlockStop { index } if *index == last),
                );
                if !already_stopped {
                    events.push(StreamEvent::ContentBlockStop { index: last });
                }
            }

            events.push(StreamEvent::MessageDelta {
                stop_reason: Some(map_finish_reason(finish_reason)),
                usage: extract_usage(&json),
            });
        }
    }

    // Usage-only message (final stats without candidates).
    if let Some(usage_meta) = json.get("usageMetadata") {
        let has_candidates = json["candidates"].as_array().is_some_and(|a| !a.is_empty());

        if !has_candidates {
            events.push(StreamEvent::MessageDelta {
                stop_reason: None,
                usage: Some(Usage {
                    input_tokens: usage_meta["promptTokenCount"].as_u64().unwrap_or(0) as u32,
                    output_tokens: usage_meta["candidatesTokenCount"].as_u64().unwrap_or(0) as u32,
                }),
            });
        }
    }

    Ok(events)
}

fn parse_json(data: &str) -> Result<Value, LLMError> {
    serde_json::from_str(data)
        .map_err(|e| LLMError::new(ERROR_SSE_PARSE, format!("Invalid JSON: {}", e)))
}

fn map_finish_reason(reason: &str) -> String {
    match reason {
        "STOP" => "end_turn".to_string(),
        "MAX_TOKENS" => "max_tokens".to_string(),
        "SAFETY" => "safety".to_string(),
        "RECITATION" => "recitation".to_string(),
        other => other.to_lowercase(),
    }
}

fn extract_usage(json: &Value) -> Option<Usage> {
    json.get("usageMetadata").map(|m| Usage {
        input_tokens: m["promptTokenCount"].as_u64().unwrap_or(0) as u32,
        output_tokens: m["candidatesTokenCount"].as_u64().unwrap_or(0) as u32,
    })
}

/// Check if a JSON string is syntactically complete.
fn is_complete_json(s: &str) -> bool {
    serde_json::from_str::<Value>(s.trim()).is_ok()
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
    fn test_parse_text_response() {
        let events = parse_stream_event(
          &sse(r#"{"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"finishReason":"STOP"}]}"#),
          &mut StreamState::default(),
      ).unwrap();

        assert_eq!(events.len(), 4);
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
        assert_eq!(events[2], StreamEvent::ContentBlockStop { index: 0 });
        assert!(matches!(
            &events[3],
            StreamEvent::MessageDelta { stop_reason: Some(r), .. } if r == "end_turn"
        ));
    }

    #[test]
    fn test_parse_function_call_complete() {
        let events = parse_stream_event(
             &sse(r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"get_weather","args":{"location":"SF"}}}],"role":"model"},"finishReason":"STOP"}]}"#),
             &mut StreamState::default(),
        ).unwrap();

        assert_eq!(
            events[0],
            StreamEvent::ContentBlockStart {
                index: 0,
                block_type: ContentBlockType::ToolUse {
                    id: "get_weather".to_string(),
                    name: "get_weather".to_string(),
                },
            }
        );
        assert!(
            matches!(&events[1], StreamEvent::InputJsonDelta { index: 0, json } if json.contains("SF"))
        );
        assert_eq!(events[2], StreamEvent::ContentBlockStop { index: 0 });
        assert!(matches!(
            &events[3],
            StreamEvent::MessageDelta { stop_reason: Some(r), .. } if r == "end_turn"
        ));
    }

    #[test]
    fn test_parse_error_response() {
        let err = parse_stream_event(
            &sse(r#"{"error":{"code":400,"message":"Invalid request"}}"#),
            &mut StreamState::default(),
        )
        .unwrap_err();

        assert_eq!(err.error_code, "GEMINI_ERROR_400");
    }

    #[test]
    fn test_parse_usage_only_message() {
        let events = parse_stream_event(
            &sse(r#"{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}"#),
            &mut StreamState::default(),
        )
        .unwrap();

        assert_eq!(
            events,
            vec![StreamEvent::MessageDelta {
                stop_reason: None,
                usage: Some(Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                }),
            }]
        );
    }

    #[test]
    fn test_parse_usage_with_candidates() {
        let events = parse_stream_event(
             &sse(r#"{"candidates":[{"content":{"parts":[{"text":"Hi"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}"#),
             &mut StreamState::default(),
        ).unwrap();

        // Usage should be in the MessageDelta from finishReason, not duplicated.
        let usage_events: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, StreamEvent::MessageDelta { usage: Some(_), .. }))
            .collect();
        assert_eq!(usage_events.len(), 1);
    }

    #[test]
    fn test_stream_state_tracking() {
        let mut state = StreamState::default();

        let _ = parse_stream_event(
            &sse(r#"{"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}"#),
            &mut state,
        )
        .unwrap();

        assert_eq!(state.block_count, 1);
        assert!(state.has_content);
    }

    #[test]
    fn test_partial_function_call_accumulation() {
        let mut state = StreamState::default();

        // First chunk: incomplete args.
        let events1 = parse_stream_event(
               &sse(r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"search","args":{"q":"hello"}}}],"role":"model"},"finishReason":"STOP"}]}"#),
               &mut state,
           )
           .unwrap();

        // With complete args it should start, delta, stop, and message delta.
        assert!(events1.iter().any(|e| matches!(
           e,
           StreamEvent::ContentBlockStart { block_type: ContentBlockType::ToolUse { name, .. }, .. }
           if name == "search"
       )));
        assert!(
            events1
                .iter()
                .any(|e| matches!(e, StreamEvent::ContentBlockStop { .. }))
        );
    }

    #[test]
    fn test_map_finish_reason() {
        assert_eq!(map_finish_reason("STOP"), "end_turn");
        assert_eq!(map_finish_reason("MAX_TOKENS"), "max_tokens");
        assert_eq!(map_finish_reason("SAFETY"), "safety");
        assert_eq!(map_finish_reason("RECITATION"), "recitation");
        assert_eq!(map_finish_reason("UNKNOWN"), "unknown");
    }

    #[test]
    fn test_is_complete_json() {
        assert!(is_complete_json(r#"{"key": "value"}"#));
        assert!(is_complete_json(r#"[1, 2, 3]"#));
        assert!(is_complete_json(r#"null"#));
        assert!(!is_complete_json(r#"{"key": "val"#));
        assert!(!is_complete_json(r#""#));
    }
}
