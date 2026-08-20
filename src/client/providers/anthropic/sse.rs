use serde_json::Value;

use crate::client::error::LlmError;
use crate::client::models::{ContentBlockType, StreamEvent, Usage};

/// Error code for SSE parsing errors (consistent with other providers).
const ERROR_SSE_PARSE: &str = "SSE_PARSE_ERROR";

/// Parsed SSE event with event type and data.
#[derive(Debug)]
pub struct SseEvent {
    /// Event type (e.g., "message_start", "content_block_delta")
    pub event: String,
    /// JSON data payload.
    pub data: String,
}

/// Parse SSE lines from a buffer, returning events and the remaining buffer.
///
/// The remaining buffer is returned verbatim so callers can prepend it to the
/// next chunk without losing any bytes (including `id:`, `retry:`, or comment)
/// lines, and exact multi-line `data:` formatting)
pub fn parse_sse_chunk(buffer: &str) -> (Vec<SseEvent>, String) {
    let mut events = Vec::new();
    let mut event: Option<String> = None;
    let mut data: Vec<String> = Vec::new();

    // Byte offset just past the last complete event. Everything after this is
    // an incomplete event and is returned unchanged.
    let mut completed_through = 0_usize;

    let mut idx = 0_usize;
    while idx < buffer.len() {
        let line_end = match buffer[idx..].find('\n') {
            Some(rel) => idx + rel,
            None => buffer.len(),
        };
        let has_newline = line_end < buffer.len();
        // Strip the trailing `\r` from CRLF line endings.
        let line = buffer[idx..line_end].trim_end_matches('\r');

        if line.is_empty() {
            // A blank line terminates an event only if it is a complete line.
            if has_newline {
                if event.is_some() || !data.is_empty() {
                    events.push(SseEvent {
                        event: event.take().unwrap_or_default(),
                        data: data.join("\n"),
                    });
                    data.clear();
                }
                completed_through = line_end + 1;
            }
        } else if let Some(value) = line.strip_prefix("event:") {
            // Strip a single leading space, per the SSE spec.
            event = Some(value.strip_prefix(' ').unwrap_or(value).to_string());
        } else if let Some(value) = line.strip_prefix("data:") {
            data.push(value.strip_prefix(' ').unwrap_or(value).to_string());
        }
        // Ignore `id:`, `retry:`, and comment lines (starting with ':').

        if has_newline {
            idx = line_end + 1;
        } else {
            break;
        }
    }

    (events, buffer[completed_through..].to_string())
}

/// Parse an SSE event into a `StreamEvent`
pub fn parse_stream_event(sse: &SseEvent) -> Result<Option<StreamEvent>, LlmError> {
    match sse.event.as_str() {
        "message_start" => Ok(parse_message_start(&parse_json(&sse.data)?)),
        "content_block_start" => Ok(parse_content_block_start(&parse_json(&sse.data)?)),
        "content_block_delta" => Ok(parse_content_block_delta(&parse_json(&sse.data)?)),
        "content_block_stop" => Ok(parse_content_block_stop(&parse_json(&sse.data)?)),
        "message_delta" => Ok(parse_message_delta(&parse_json(&sse.data)?)),
        "message_stop" => Ok(Some(StreamEvent::MessageStop)),
        "ping" => Ok(Some(StreamEvent::Ping)),
        "error" => Err(parse_error(&parse_json(&sse.data)?)),
        _ => Ok(None), // Unknown event type, skip
    }
}

fn parse_json(data: &str) -> Result<Value, LlmError> {
    serde_json::from_str(data)
        .map_err(|e| LlmError::new(ERROR_SSE_PARSE, format!("Invalid JSON: {}", e)))
}

fn parse_message_start(json: &Value) -> Option<StreamEvent> {
    let message = &json["message"];
    Some(StreamEvent::MessageStart {
        message_id: message["id"].as_str().unwrap_or_default().to_string(),
        model: message["model"].as_str().unwrap_or_default().to_string(),
    })
}

fn parse_content_block_start(json: &Value) -> Option<StreamEvent> {
    let index = json["index"].as_u64().unwrap_or(0) as usize;
    let content_block = json.get("content_block")?;
    let block_type = match content_block["type"].as_str().unwrap_or("text") {
        "tool_use" => ContentBlockType::ToolUse {
            id: content_block["id"].as_str().unwrap_or_default().to_string(),
            name: content_block["name"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        },
        _ => ContentBlockType::Text,
    };

    Some(StreamEvent::ContentBlockStart { index, block_type })
}

fn parse_content_block_delta(json: &Value) -> Option<StreamEvent> {
    let index = json["index"].as_u64().unwrap_or(0) as usize;
    let delta = json.get("delta")?;

    match delta["type"].as_str().unwrap_or("") {
        "text_delta" => Some(StreamEvent::TextDelta {
            index,
            text: delta["text"].as_str().unwrap_or_default().to_string(),
        }),
        "input_json_delta" => Some(StreamEvent::InputJsonDelta {
            index,
            json: delta["partial_json"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        }),
        _ => None,
    }
}

fn parse_content_block_stop(json: &Value) -> Option<StreamEvent> {
    let index = json["index"].as_u64().unwrap_or(0) as usize;
    Some(StreamEvent::ContentBlockStop { index })
}

fn parse_message_delta(json: &Value) -> Option<StreamEvent> {
    let stop_reason = json["delta"]["stop_reason"].as_str().map(|s| s.to_string());

    let usage = if json["usage"]["output_tokens"].as_u64().is_some() {
        Some(Usage {
            input_tokens: json["usage"]["input_tokens"].as_u64().unwrap_or(0) as u32,
            output_tokens: json["usage"]["output_tokens"].as_u64().unwrap_or(0) as u32,
        })
    } else {
        None
    };

    Some(StreamEvent::MessageDelta { stop_reason, usage })
}

fn parse_error(json: &Value) -> LlmError {
    let error = &json["error"];
    LlmError::new(
        error["type"].as_str().unwrap_or("unknown"),
        error["message"].as_str().unwrap_or("Unknown error"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse(event: &str, data: &str) -> SseEvent {
        SseEvent {
            event: event.to_string(),
            data: data.to_string(),
        }
    }

    #[test]
    fn test_parse_sse_chunk() {
        let chunk =
            "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: ping\ndata: {}\n\n";
        let (events, remaining) = parse_sse_chunk(chunk);

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event, "message_start");
        assert_eq!(events[0].data, "{\"type\":\"message_start\"}");
        assert_eq!(events[1].event, "ping");
        assert_eq!(events[1].data, "{}");
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_sse_chunk_multiline_data() {
        let chunk = "event: text\ndata: hello\ndata: world\n\n";
        let (events, remaining) = parse_sse_chunk(chunk);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event, "text");
        assert_eq!(events[0].data, "hello\nworld");
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_sse_chunk_crlf() {
        let chunk = "event: ping\r\ndata: {}\r\n\r\n";
        let (events, remaining) = parse_sse_chunk(chunk);

        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event, "ping");
        assert_eq!(events[0].data, "{}");
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_incomplete_chunk() {
        let chunk = "event: message_start\ndata: {\"test\":true}";
        let (events, remaining) = parse_sse_chunk(chunk);

        assert!(events.is_empty());
        assert_eq!(remaining, chunk);
    }

    #[test]
    fn test_parse_message_start() {
        let event = parse_stream_event(&sse(
            "message_start",
            r#"{"type":"message_start","message":{"id":"msg_1","model":"claude-3-5-sonnet"}}"#,
        ))
        .unwrap()
        .unwrap();

        assert_eq!(
            event,
            StreamEvent::MessageStart {
                message_id: "msg_1".to_string(),
                model: "claude-3-5-sonnet".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_text_delta() {
        let event = parse_stream_event(&sse(
             "content_block_delta",
             r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#,
         ))
         .unwrap()
         .unwrap();

        assert_eq!(
            event,
            StreamEvent::TextDelta {
                index: 0,
                text: "Hello".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_input_json_delta() {
        let event = parse_stream_event(&sse(
            "content_block_delta",
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":"}}"#,
        ))
        .unwrap()
        .unwrap();

        assert_eq!(
            event,
            StreamEvent::InputJsonDelta {
                index: 1,
                json: "{\"x\":".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_tool_use_start() {
        let event = parse_stream_event(&sse(
             "content_block_start",
             r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"tool_123","name":"get_weather"}}"#,
         ))
         .unwrap()
         .unwrap();

        assert_eq!(
            event,
            StreamEvent::ContentBlockStart {
                index: 1,
                block_type: ContentBlockType::ToolUse {
                    id: "tool_123".to_string(),
                    name: "get_weather".to_string(),
                },
            }
        );
    }

    #[test]
    fn test_parse_content_block_stop() {
        let event = parse_stream_event(&sse(
            "content_block_stop",
            r#"{"type":"content_block_stop","index":2}"#,
        ))
        .unwrap()
        .unwrap();

        assert_eq!(event, StreamEvent::ContentBlockStop { index: 2 });
    }

    #[test]
    fn test_parse_message_delta() {
        let event = parse_stream_event(&sse(
            "message_delta",
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}"#,
        ))
        .unwrap()
        .unwrap();

        assert_eq!(
            event,
            StreamEvent::MessageDelta {
                stop_reason: Some("end_turn".to_string()),
                usage: Some(Usage {
                    input_tokens: 0,
                    output_tokens: 15,
                }),
            }
        );
    }

    #[test]
    fn test_parse_message_stop() {
        let event = parse_stream_event(&sse("message_stop", "{}"))
            .unwrap()
            .unwrap();
        assert_eq!(event, StreamEvent::MessageStop);
    }

    #[test]
    fn test_parse_ping() {
        let event = parse_stream_event(&sse("ping", "{}")).unwrap().unwrap();
        assert_eq!(event, StreamEvent::Ping);
    }

    #[test]
    fn test_parse_error() {
        let err = parse_stream_event(&sse(
            "error",
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        ))
        .unwrap_err();

        assert_eq!(err.error_code, "overloaded_error");
    }

    #[test]
    fn test_parse_unknown_event() {
        let event = parse_stream_event(&sse("unknown_event", "{}")).unwrap();
        assert!(event.is_none());
    }
}
