//! Cohere API request/response types and serialization.

use serde_json::{Map, Value, json};

use crate::client::error::LLMError;
use crate::client::models::{Content, Message, MessageOptions, Role, Tool, ToolChoice, ToolUse};

const API_URL: &str = "https://api.cohere.com/v2/chat";
const ERROR_INVALID_REQUEST: &str = "INVALID_REQUEST";
const ERROR_PARSE: &str = "PARSE_ERROR";

// === Public API ===

/// Returns the Cohere chat API endpoint URL.
pub fn get_api_url() -> &'static str {
    API_URL
}

/// Returns the HTTP headers required for the Cohere API.
pub fn get_request_headers(api_key: &str) -> Result<Vec<(&'static str, String)>, LLMError> {
    if api_key.is_empty() {
        return Err(LLMError::new(
            ERROR_INVALID_REQUEST,
            "API key cannot be empty",
        ));
    }
    Ok(vec![
        ("Content-Type", "application/json".to_string()),
        ("Authorization", format!("Bearer {}", api_key)),
    ])
}

/// Builds the JSON request body for the Cohere Chat API.
pub fn build_request_body(
    messages: &[Message],
    options: &MessageOptions,
    default_model: &str,
    streaming: bool,
) -> Result<String, LLMError> {
    if messages.is_empty() {
        return Err(LLMError::new(
            ERROR_INVALID_REQUEST,
            "At least one message is required",
        ));
    }

    let model = options.model.as_deref().unwrap_or(default_model);

    let mut body = Map::new();
    body.insert("model".to_string(), json!(model));

    if streaming {
        body.insert("stream".to_string(), json!(true));
    }

    // Messages array
    let messages_arr: Vec<Value> = messages.iter().map(message_to_value).collect();
    body.insert("messages".to_string(), Value::Array(messages_arr));

    // Optional parameters
    if let Some(max_tokens) = options.max_tokens {
        body.insert("max_tokens".to_string(), json!(max_tokens));
    }
    if let Some(temp) = options.temperature {
        body.insert("temperature".to_string(), json!(temp));
    }
    if let Some(top_p) = options.top_p {
        body.insert("p".to_string(), json!(top_p));
    }
    if let Some(top_k) = options.top_k {
        body.insert("k".to_string(), json!(top_k));
    }
    if let Some(stop_sequences) = &options.stop_sequences
        && !stop_sequences.is_empty()
    {
        body.insert("stop_sequences".to_string(), json!(stop_sequences));
    }

    // Tools
    if let Some(tools) = &options.tools
        && !tools.is_empty()
    {
        let tools_arr: Vec<Value> = tools.iter().map(tool_to_value).collect();
        body.insert("tools".to_string(), Value::Array(tools_arr));
    }

    // Tool choice
    if let Some(tool_choice) = &options.tool_choice {
        if let Some(choice_val) = tool_choice_to_value(tool_choice) {
            body.insert("tool_choice".to_string(), choice_val);
        }
    }

    serde_json::to_string(&body).map_err(|e| LLMError::new("SERIALIZE_ERROR", e.to_string()))
}

/// Parses the Cohere API response and extracts the assistant message.
pub fn parse_response(response_body: &str) -> Result<Message, LLMError> {
    let parsed: Value = serde_json::from_str(response_body)
        .map_err(|e| LLMError::new(ERROR_PARSE, format!("Failed to parse response: {}", e)))?;

    // Check for error response (Cohere returns errors with a "message" and no content).
    if let Some(error_msg) = parsed["message"].as_str() {
        if parsed.get("text").is_none() && parsed.get("tool_calls").is_none() {
            return Err(LLMError::new("COHERE_ERROR", error_msg));
        }
    }

    let mut content_blocks: Vec<Content> = Vec::new();

    // V2 format: message.content[] and message.tool_calls[]
    if let Some(message) = parsed.get("message") {
        if let Some(content_arr) = message["content"].as_array() {
            for block in content_arr {
                if let Some(text) = block["text"].as_str()
                    && !text.is_empty()
                {
                    content_blocks.push(Content::Text(text.to_string()));
                }
            }
        }

        if let Some(tool_calls) = message["tool_calls"].as_array() {
            for tc in tool_calls {
                content_blocks.push(Content::ToolUse(parse_tool_call(tc)));
            }
        }
    }

    // Fallback: older API format with top-level "text" and "tool_calls".
    if content_blocks.is_empty() {
        if let Some(text) = parsed["text"].as_str()
            && !text.is_empty()
        {
            content_blocks.push(Content::Text(text.to_string()));
        }
        if let Some(tool_calls) = parsed["tool_calls"].as_array() {
            for tc in tool_calls {
                content_blocks.push(Content::ToolUse(parse_tool_call(tc)));
            }
        }
    }

    if content_blocks.is_empty() {
        return Err(LLMError::new(ERROR_PARSE, "No content found in response"));
    }

    Ok(Message::with_content(Role::Assistant, content_blocks))
}

// === Private Helpers ===

fn message_to_value(msg: &Message) -> Value {
    let has_tool_results = msg
        .content
        .iter()
        .any(|c| matches!(c, Content::ToolResult(_)));
    let has_tool_calls = msg.content.iter().any(|c| matches!(c, Content::ToolUse(_)));

    // Tool results use role "tool" in Cohere.
    if has_tool_results {
        if let Some(Content::ToolResult(tr)) = msg
            .content
            .iter()
            .find(|c| matches!(c, Content::ToolResult(_)))
        {
            return json!({
                "role": "tool",
                "tool_call_id": tr.tool_use_id,
                "content": tr.content
            });
        }
    }

    // Assistant messages with tool calls.
    if has_tool_calls {
        let mut obj = Map::new();
        obj.insert("role".to_string(), json!("assistant"));

        // Include text contnet if present.
        let text: String = msg
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !text.is_empty() {
            obj.insert("content".to_string(), json!(text));
        }

        // Tool calls array.
        let tool_calls: Vec<Value> = msg
            .content
            .iter()
            .filter_map(|c| match c {
                Content::ToolUse(tu) => Some(json!({
                    "id": tu.id,
                    "type": "function",
                    "function": {
                        "name": tu.name,
                        "arguments": tu.input
                    }
                })),
                _ => None,
            })
            .collect();
        obj.insert("tool_calls".to_string(), Value::Array(tool_calls));

        return Value::Object(obj);
    }

    // Regular message (system, user, or assistant without tool calls).
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    };

    let text: String = msg
        .content
        .iter()
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    json!({ "role": role, "content": text })
}

fn tool_to_value(tool: &Tool) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema
        }
    })
}

fn tool_choice_to_value(choice: &ToolChoice) -> Option<Value> {
    match choice {
        ToolChoice::Auto => None, // Cohere default is auto
        ToolChoice::Any => Some(json!("required")),
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Tool(name) => Some(json!({
            "type": "function",
            "function": { "name": name }
        })),
    }
}

/// Parse a tool call object from the response into a ToolUse.
fn parse_tool_call(tc: &Value) -> ToolUse {
    let id = tc["id"].as_str().unwrap_or_default().to_string();
    let name = tc["function"]["name"]
        .as_str()
        .unwrap_or_default()
        .to_string();

    // Cohere may return arguments as a JSON string or as an object.
    let input = if let Some(args_str) = tc["function"]["arguments"].as_str() {
        serde_json::from_str(args_str).unwrap_or(json!({}))
    } else {
        tc["function"]["arguments"].clone()
    };

    ToolUse { id, name, input }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json_str: &str) -> Value {
        serde_json::from_str(json_str).unwrap()
    }

    #[test]
    fn test_get_api_url() {
        assert_eq!(get_api_url(), "https://api.cohere.com/v2/chat");
    }

    #[test]
    fn test_get_request_headers_valid() {
        let headers = get_request_headers("test-key").unwrap();
        assert_eq!(headers.len(), 2);
        assert_eq!(headers[0], ("Content-Type", "application/json".to_string()));
        assert_eq!(headers[1], ("Authorization", "Bearer test-key".to_string()));
    }

    #[test]
    fn test_get_request_headers_empty_key() {
        let err = get_request_headers("").unwrap_err();
        assert_eq!(err.error_code, ERROR_INVALID_REQUEST);
    }

    #[test]
    fn test_build_request_body_simple() {
        let messages = vec![Message::user("Hello")];
        let options = MessageOptions::default();

        let body =
            parse(&build_request_body(&messages, &options, "command-r-plus", false).unwrap());

        assert_eq!(body["model"], "command-r-plus");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "Hello");
        assert!(body.get("stream").is_none());
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("temperature").is_none());
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn test_build_request_body_streaming() {
        let messages = vec![Message::user("Hello")];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options, "command-r-plus", true).unwrap());
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn test_build_request_body_with_options() {
        let messages = vec![Message::user("Hi")];
        let options = MessageOptions {
            max_tokens: Some(512),
            temperature: Some(0.5),
            top_p: Some(0.9),
            top_k: Some(40),
            stop_sequences: Some(vec!["END".to_string()]),
            ..Default::default()
        };

        let body = parse(&build_request_body(&messages, &options, "command-r", false).unwrap());

        assert_eq!(body["max_tokens"], 512);
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["p"], 0.9);
        assert_eq!(body["k"], 40);
        assert_eq!(body["stop_sequences"], json!(["END"]));
    }

    #[test]
    fn test_build_request_body_with_system() {
        let messages = vec![Message::system("Be helpful"), Message::user("Hello")];
        let options = MessageOptions::default();

        let body =
            parse(&build_request_body(&messages, &options, "command-r-plus", false).unwrap());
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "Be helpful");
        assert_eq!(body["messages"][1]["role"], "user");
    }

    #[test]
    fn test_build_request_body_with_tools() {
        let messages = vec![Message::user("What's the weather?")];
        let options = MessageOptions {
            tools: Some(vec![Tool::new(
                "get_weather",
                "Get weather for a location",
                json!({ "type": "object", "properties": { "location": { "type": "string" } } }),
            )]),
            tool_choice: Some(ToolChoice::Any),
            ..Default::default()
        };

        let body =
            parse(&build_request_body(&messages, &options, "command-r-plus", false).unwrap());

        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(
            body["tools"][0]["function"]["parameters"],
            json!({ "type": "object", "properties": { "location": { "type": "string" } } })
        );
        assert_eq!(body["tool_choice"], "required");
    }

    #[test]
    fn test_build_request_body_tool_choice_specific() {
        let messages = vec![Message::user("Hi")];
        let options = MessageOptions {
            tools: Some(vec![Tool::new("search", "Search", json!({}))]),
            tool_choice: Some(ToolChoice::Tool("search".to_string())),
            ..Default::default()
        };

        let body = parse(&build_request_body(&messages, &options, "command-r", false).unwrap());

        assert_eq!(
            body["tool_choice"],
            json!({ "type": "function", "function": { "name": "search" } })
        );
    }

    #[test]
    fn test_build_request_body_with_tool_use_and_result() {
        let messages = vec![
            Message::user("Weather?"),
            Message::with_content(
                Role::Assistant,
                vec![Content::ToolUse(ToolUse {
                    id: "tc_1".to_string(),
                    name: "get_weather".to_string(),
                    input: json!({ "location": "NYC "}),
                })],
            ),
            Message::tool_result("tc_1", "72F sunny", false),
        ];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options, "command-r", false).unwrap());

        // Assistant with tool calls.
        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(body["messages"][1]["tool_calls"][0]["id"], "tc_1");
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["arguments"],
            json!({ "location": "NYC" })
        );

        // Tool result.
        assert_eq!(body["messages"][2]["role"], "tool");
        assert_eq!(body["messages"][2]["tool_call_id"], "tc_1");
        assert_eq!(body["messages"][2]["content"], "72F sunny");
    }

    #[test]
    fn test_build_request_body_empty_messages() {
        let err = build_request_body(&[], &MessageOptions::default(), "model", false).unwrap_err();
        assert_eq!(err.error_code, ERROR_INVALID_REQUEST);
    }

    #[test]
    fn test_parse_response_text() {
        let response = r#"{
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello! How can I help?"}]
            }
        }"#;

        let msg = parse_response(response).unwrap();
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            Content::Text(t) => assert_eq!(t, "Hello! How can I help?"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_parse_response_tool_calls_string_args() {
        let response = r#"{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"location\":\"SF\"}"
                    }
                }]
            }
        }"#;

        let msg = parse_response(response).unwrap();
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            Content::ToolUse(tu) => {
                assert_eq!(tu.id, "call_123");
                assert_eq!(tu.name, "get_weather");
                assert_eq!(tu.input, json!({ "location": "SF" }));
            }
            _ => panic!("Expected tool use"),
        }
    }

    #[test]
    fn test_parse_response_tool_calls_object_args() {
        let response = r#"{
            "message": {
                "role": "assistant",
                "tool_calls": [{
                    "id": "call_456",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": {"query": "rust"}
                    }
                }]
            }
        }"#;

        let msg = parse_response(response).unwrap();
        match &msg.content[0] {
            Content::ToolUse(tu) => {
                assert_eq!(tu.name, "search");
                assert_eq!(tu.input, json!({ "query": "rust" }));
            }
            _ => panic!("Expected tool use"),
        }
    }

    #[test]
    fn test_parse_response_legacy_format() {
        let response = r#"{"text": "Hello from v1"}"#;

        let msg = parse_response(response).unwrap();
        match &msg.content[0] {
            Content::Text(t) => assert_eq!(t, "Hello from v1"),
            _ => panic!("Expected text"),
        }
    }

    #[test]
    fn test_parse_response_error() {
        let response = r#"{"message": "Invalid API key"}"#;

        let err = parse_response(response).unwrap_err();
        assert_eq!(err.error_code, "COHERE_ERROR");
    }

    #[test]
    fn test_parse_response_no_content() {
        let response = r#"{"message": {"role": "assistant", "content": []}}"#;

        let err = parse_response(response).unwrap_err();
        assert_eq!(err.error_code, ERROR_PARSE);
    }
}
