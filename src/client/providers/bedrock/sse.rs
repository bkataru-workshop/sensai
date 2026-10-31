//! Event stream parser for Bedrock streaming responses.
//!
//! Bedrock uses AWS event-stream binary encoding, not standard SSE.

use serde_json::Value;

use crate::client::error::LLMError;
use crate::client::models::{ContentBlockType, StreamEvent, Usage};

// == Constants ==

/// Prelude length in event-stream format (total_length:4 + headers_length:4 + crc:4).
const PRELUDE_LENGTH: usize = 12;

/// CRC length at the end of each message.
const MESSAGE_CRC_LENGTH: usize = 4;

const ERROR_EVENT_PARSE: &str = "EVENT_STREAM_PARSE_ERROR";

// == Types ==

/// Parsed event from Bedrock event-stream
#[derive(Debug)]
pub struct BedrockEvent {
    /// Event type (e.g., "messageStart", "contentBlockDelta", "messageStop").
    pub event_type: String,
    /// JSON payload
    pub payload: String,
}

/// State tracker for streaming content block indices.
#[derive(Debug, Default, Clone)]
pub struct StreamState {
    /// Current content block index.
    pub block_index: usize,
    /// Whether we've emitted a text block start.
    pub text_block_started: bool,
    /// Whether stream has finished.
    pub finished: bool,
}

/// == Public API ==

/// Parse event-stream encoded data from a buffer.
///
/// Returns parsed events and remaining bytes that form an incomplete message.
pub fn parse_event_stream(buffer: &[u8]) -> (Vec<BedrockEvent>, Vec<u8>) {
    let mut events = Vec::new();
    let mut offset = 0;

    while offset + PRELUDE_LENGTH <= buffer.len() {
        let total_length = u32::from_be_bytes([
            buffer[offset],
            buffer[offset + 1],
            buffer[offset + 2],
            buffer[offset + 3],
        ]) as usize;

        // Not enough data for the complete message yet.
        if offset + total_length > buffer.len() {
            break;
        }

        let headers_length = u32::from_be_bytes([
            buffer[offset + 4],
            buffer[offset + 5],
            buffer[offset + 6],
            buffer[offset + 7],
        ]) as usize;

        // Parse headers to find the event type.
        let headers_start = offset + PRELUDE_LENGTH;
        let headers_end = headers_start + headers_length;
        let event_type = extract_event_type(&buffer[headers_start..headers_end]);

        // Extract payload.
        let payload_start = headers_end;
        let payload_end = offset + total_length + MESSAGE_CRC_LENGTH;

        if let Ok(payload) = std::str::from_utf8(&buffer[payload_start..payload_end]) {
            events.push(BedrockEvent {
                event_type,
                payload: payload.to_string(),
            });
        }

        offset += total_length;
    }

    (events, buffer[offset..].to_vec())
}

// Parse a Bedrock event into StreamEvents.
pub fn parse_stream_event(
    event: &BedrockEvent,
    state: &mut StreamState,
) -> Result<Vec<StreamEvent>, LLMError> {
    let json = parse_json(&event.payload)?;
    let mut events = Vec::new();

    match event.event_type.as_str() {
        "messageStart" => {}
        "contentBlockStart" => {
            let start = &json["start"];

            if start.get("toolUse").is_some() {
                let tool_use = &start["toolUse"];
                events.push(StreamEvent::ContentBlockStart {
                    index: state.block_index,
                    block_type: ContentBlockType::ToolUse {
                        id: tool_use["toolUseId"]
                            .as_str()
                            .unwrap_or_default()
                            .to_string(),
                        name: tool_use["name"].as_str().unwrap_or_default().to_string(),
                    },
                });
            } else {
                events.push(StreamEvent::ContentBlockStart {
                    index: state.block_index,
                    block_type: ContentBlockType::Text,
                });
                state.text_block_started = true;
            }
        }
        "contentBlockDelta" => {
            let delta = &json["delta"];

            if let Some(text) = delta["text"].as_str()
                && !text.is_empty()
            {
                events.push(StreamEvent::TextDelta {
                    index: state.block_index,
                    text: text.to_string(),
                });
            }

            if let Some(tool_use) = delta.get("toolUse")
                && let Some(input) = tool_use["input"].as_str()
                && !input.is_empty()
            {
                events.push(StreamEvent::InputJsonDelta {
                    index: state.block_index,
                    json: input.to_string(),
                });
            }
        }
        "contentBlockStop" => {
            events.push(StreamEvent::ContentBlockStop {
                index: state.block_index,
            });
            state.block_index += 1;
            state.text_block_started = false;
        }
        "messageStop" => {
            state.finished = true;

            let stop_reason = json["stopReason"].as_str().map(|s| s.to_string());
            events.push(StreamEvent::MessageDelta {
                stop_reason,
                usage: None,
            });
        }
        "metadata" => {
            if let Some(usage) = json.get("usage") {
                events.push(StreamEvent::MessageDelta {
                    stop_reason: None,
                    usage: Some(Usage {
                        input_tokens: usage["inputTokens"].as_u64().unwrap_or(0) as u32,
                        output_tokens: usage["outputTokens"].as_u64().unwrap_or(0) as u32,
                    }),
                });
            }
        }
        _ => {}
    }

    Ok(events)
}

// == Private helpers ==

fn parse_json(data: &str) -> Result<Value, LLMError> {
    serde_json::from_str(data)
        .map_err(|e| LLMError::new(ERROR_EVENT_PARSE, format!("Invalid JSON: {}", e)))
}

/// Extract `:event-type` header value from raw header bytes.
fn extract_event_type(data: &[u8]) -> String {
    // Walk through headers looking for `:event-type`
    let mut offset = 0;

    while offset < data.len() {
        // Name length (1 byte)
        let name_len = data[offset] as usize;
        offset += 1;

        if offset + name_len > data.len() {
            break;
        }
        let name = &data[offset..offset + name_len];
        offset += name_len;

        // Header value type (1 byte)
        if offset >= data.len() {
            break;
        }
        let header_type = data[offset];
        offset += 1;

        if header_type == 7 {
            // String type: 2-byte big-endian length + value
            if offset + 2 > data.len() {
                break;
            }
            let value_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;

            if offset + value_len > data.len() {
                break;
            }
            let value = &data[offset..offset + value_len];
            offset += value_len;

            if name == b":event-type" {
                return String::from_utf8_lossy(value).to_string();
            }
        } else {
            // Unknown header type - can't continue parsing safely.
            break;
        }
    }

    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bedrock_event(event_type: &str, payload: &str) -> BedrockEvent {
        BedrockEvent {
            event_type: event_type.to_string(),
            payload: payload.to_string(),
        }
    }

    #[test]
    fn test_parse_headers_event_type() {
        // Header: name_len(1) + ":event-type"(11) + type(1=7) + value_len(2) + "messageStart"(12)
        let data: Vec<u8> = vec![
            11, b':', b'e', b'v', b'e', b'n', b't', b'-', b't', b'y', b'p', b'e', 7, 0, 12, b'm',
            b'e', b's', b's', b'a', b'g', b'e', b'S', b't', b'a', b'r', b't',
        ];

        assert_eq!(extract_event_type(&data), "messageStart");
    }

    #[test]
    fn test_parse_stream_event_text_delta() {
        let event = bedrock_event(
            "contentBlockDelta",
            r#"{"contentBlockIndex":0,"delta":{"text":"Hello"}}"#,
        );
        let mut state = StreamState::default();
        let events = parse_stream_event(&event, &mut state).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            StreamEvent::TextDelta {
                index: 0,
                text: "Hello".to_string(),
            }
        );
    }

    #[test]
    fn test_parse_stream_event_tool_use_start() {
        let event = bedrock_event(
            "contentBlockStart",
            r#"{"start":{"toolUse":{"toolUseId":"call_1","name":"search"}}}"#,
        );
        let mut state = StreamState::default();
        let events = parse_stream_event(&event, &mut state).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            StreamEvent::ContentBlockStart {
                index: 0,
                block_type: ContentBlockType::ToolUse {
                    id: "call_1".to_string(),
                    name: "search".to_string(),
                },
            }
        );
    }

    #[test]
    fn test_parse_stream_event_text_block_start() {
        let event = bedrock_event("contentBlockStart", r#"{"start":{"text":""}}"#);
        let mut state = StreamState::default();
        let events = parse_stream_event(&event, &mut state).unwrap();

        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0],
            StreamEvent::ContentBlockStart {
                index: 0,
                block_type: ContentBlockType::Text,
            }
        );
        assert!(state.text_block_started);
    }

    #[test]
    fn test_parse_stream_event_content_block_stop() {
        let event = bedrock_event("contentBlockStop", r#"{}"#);
        let mut state = StreamState {
            block_index: 2,
            text_block_started: true,
            finished: false,
        };
        let events = parse_stream_event(&event, &mut state).unwrap();

        assert_eq!(events, vec![StreamEvent::ContentBlockStop { index: 2 }]);
        assert_eq!(state.block_index, 3);
        assert!(!state.text_block_started);
    }

    #[test]
    fn test_parse_stream_event_message_stop() {
        let event = bedrock_event("messageStop", r#"{"stopReason":"end_turn"}"#);
        let mut state = StreamState::default();
        let events = parse_stream_event(&event, &mut state).unwrap();

        assert_eq!(
            events,
            vec![StreamEvent::MessageDelta {
                stop_reason: Some("end_turn".to_string()),
                usage: None
            }]
        );
        assert!(state.finished);
    }

    #[test]
    fn test_parse_stream_event_metadata_usage() {
        let event = bedrock_event(
            "metadata",
            r#"{"usage":{"inputTokens":100,"outputTokens":50}}"#,
        );
        let mut state = StreamState::default();
        let events = parse_stream_event(&event, &mut state).unwrap();

        assert_eq!(
            events,
            vec![StreamEvent::MessageDelta {
                stop_reason: None,
                usage: Some(Usage {
                    input_tokens: 100,
                    output_tokens: 50,
                }),
            }]
        );
    }

    #[test]
    fn test_parse_stream_event_input_json_delta() {
        let event = bedrock_event(
            "contentBlockDelta",
            r#"{"delta":{"toolUse":{"input":"{\"q\":"}}}"#,
        );
        let mut state = StreamState {
            block_index: 1,
            ..Default::default()
        };
        let events = parse_stream_event(&event, &mut state).unwrap();

        assert_eq!(
            events,
            vec![StreamEvent::InputJsonDelta {
                index: 1,
                json: "{\"q\":".to_string(),
            }]
        );
    }

    #[test]
    fn test_parse_stream_event_unknown() {
        let event = bedrock_event("someFutureEvent", r#"{}"#);
        let mut state = StreamState::default();
        let events = parse_stream_event(&event, &mut state).unwrap();
        assert!(events.is_empty());
    }
}
