use serde_json::{Map, Value, json};

use crate::client::error::LLMError;
use crate::client::models::{
    Content, ImageSource, Message, MessageOptions, Role, Tool, ToolChoice, ToolUse,
};
use crate::client::providers::common::extract_text_content;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Builds the (streamed or sync) JSON request body for the Anthropic Messages API.
/// Anthropic requires system messages in a separate "system" field.
pub fn build_request_body(
    messages: &[Message],
    options: &MessageOptions,
    default_model: &str,
    streaming: bool,
) -> Result<String, LLMError> {
    let mut system_prompt: Option<String> = None;
    let mut conversation_messages: Vec<&Message> = Vec::new();

    // Separate system messages from conversation messages
    for msg in messages {
        match msg.role {
            Role::System => {
                // Combine multiple system messages if present
                let text = extract_text_content(msg);
                if let Some(existing) = system_prompt.take() {
                    system_prompt = Some(format!("{}\n{}", existing, text));
                } else {
                    system_prompt = Some(text);
                }
            }
            Role::User | Role::Assistant => {
                conversation_messages.push(msg);
            }
        }
    }

    if conversation_messages.is_empty() {
        return Err(LLMError::new(
            "INVALID_REQUEST",
            "At least one user or assistant message is required",
        ));
    }

    let model = options.model.as_deref().unwrap_or(default_model);
    let max_tokens = options.max_tokens.unwrap_or(1024);

    let mut body = Map::new();

    // Model
    body.insert("model".to_string(), json!(model));
    // Max tokens
    body.insert("max_tokens".to_string(), json!(max_tokens));

    // Temperature (optional)
    if let Some(temp) = options.temperature {
        body.insert("temperature".to_string(), json!(temp));
    }
    // Top P (optional)
    if let Some(top_p) = options.top_p {
        body.insert("top_p".to_string(), json!(top_p));
    }
    // Top K (optional)
    if let Some(top_k) = options.top_k {
        body.insert("top_k".to_string(), json!(top_k));
    }
    // Stop sequences (optional)
    if let Some(stop_sequences) = &options.stop_sequences
        && !stop_sequences.is_empty()
    {
        body.insert("stop_sequences".to_string(), json!(stop_sequences));
    }
    // System prompt (optional)
    if let Some(system) = &system_prompt {
        body.insert("system".to_string(), json!(system));
    }
    // Tools (optional)
    if let Some(tools) = &options.tools
        && !tools.is_empty()
    {
        let tools: Vec<Value> = tools.iter().map(tool_to_value).collect();
        body.insert("tools".to_string(), Value::Array(tools));
    }
    // Tool choice (optional)
    if let Some(tool_choice) = &options.tool_choice {
        body.insert("tool_choice".to_string(), tool_choice_to_value(tool_choice));
    }
    // Metadata (optional)
    if let Some(user_id) = options.metadata.as_ref().and_then(|m| m.user_id.as_ref()) {
        body.insert("metadata".to_string(), json!({ "user_id": user_id }));
    }

    // Messages array
    let messages: Vec<Value> = conversation_messages
        .iter()
        .map(|msg| message_to_value(msg))
        .collect();
    body.insert("messages".to_string(), Value::Array(messages));

    if streaming {
        body.insert("stream".to_string(), json!(true));
    }

    serde_json::to_string(&body).map_err(|e| LLMError::new("SERIALIZE_ERROR", e.to_string()))
}

fn message_to_value(msg: &Message) -> Value {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => unreachable!("system messages are separated before serialization"),
    };

    let content: Vec<Value> = msg.content.iter().map(content_to_value).collect();
    json!({ "role": role, "content": content })
}

fn content_to_value(content: &Content) -> Value {
    match content {
        Content::Text(text) => json!({ "type": "text", "text": text }),
        Content::Image(ImageSource::Base64 { media_type, data }) => json!({
            "type": "image",
            "source": { "type": "base64", "media_type": media_type, "data": data }
        }),
        Content::Image(ImageSource::Url(url)) => json!({
            "type": "image",
            "source": { "type": "url", "url": url }
        }),
        Content::ToolUse(tool_use) => json!({
            "type": "tool_use",
            "id": tool_use.id,
            "name": tool_use.name,
            "input": tool_use.input
        }),
        Content::ToolResult(tool_result) => {
            let mut block = Map::new();
            block.insert("type".to_string(), json!("tool_result"));
            block.insert("tool_use_id".to_string(), json!(tool_result.tool_use_id));
            block.insert("content".to_string(), json!(tool_result.content));
            if tool_result.is_error {
                block.insert("is_error".to_string(), json!(true));
            }
            Value::Object(block)
        }
    }
}

fn tool_to_value(tool: &Tool) -> Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema
    })
}

fn tool_choice_to_value(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({ "type": "auto" }),
        ToolChoice::Any => json!({ "type": "any" }),
        ToolChoice::None => json!({ "type": "none" }),
        ToolChoice::Tool(name) => json!({ "type": "tool", "name": name }),
    }
}

/// Returns the HTTP headers required for the Anthropic API.
pub fn get_request_headers(api_key: &str) -> Vec<(&'static str, String)> {
    vec![
        ("Content-Type", "application/json".to_string()),
        ("x-api-key", api_key.to_string()),
        ("anthropic-version", ANTHROPIC_VERSION.to_string()),
    ]
}

/// Returns the Anthropic API endpoint URL.
pub fn get_api_url() -> &'static str {
    ANTHROPIC_API_URL
}

/// Parses the Anthropic API response and extracts the assistant message.
pub fn parse_response(response_body: &str) -> Result<Message, LLMError> {
    let parsed: Value = serde_json::from_str(response_body)
        .map_err(|e| LLMError::new("PARSE_ERROR", format!("Failed to parse response: {}", e)))?;

    // Check for API error response
    if let Some(error_type) = parsed["error"]["type"].as_str() {
        let error_msg = parsed["error"]["message"]
            .as_str()
            .unwrap_or("Malformed error response");
        return Err(LLMError::new(error_type, error_msg));
    }

    // Extract content from successful response
    // Anthropic returns: { "content": [{"type": "text", "text": "..."}, {"type": "tool_use", ...}], ... }
    let content_array = &parsed["content"];

    let mut content_blocks: Vec<Content> = Vec::new();

    // Use as_array() iterator for arrays
    if let Some(elements) = content_array.as_array() {
        for block in elements {
            if let Some(block_type) = block["type"].as_str() {
                match block_type {
                    "text" => {
                        // serde_json properly unescapes JSON strings
                        if let Some(text) = block["text"].as_str() {
                            content_blocks.push(Content::Text(text.to_string()));
                        }
                    }
                    "tool_use" => {
                        let id = block["id"].as_str().unwrap_or("").to_string();
                        let name = block["name"].as_str().unwrap_or("").to_string();
                        // Get the input as parsed JSON value
                        let input = block["input"].clone();

                        content_blocks.push(Content::ToolUse(ToolUse { id, name, input }));
                    }
                    _ => {
                        // Skip unknown block types
                    }
                }
            }
        }
    }

    if content_blocks.is_empty() {
        return Err(LLMError::new("PARSE_ERROR", "No content found in response"));
    }

    Ok(Message::with_content(Role::Assistant, content_blocks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::models::Tool;

    fn parse_json(json: &str) -> Value {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn test_build_request_body_simple() {
        let messages = vec![Message::user("Hello")];
        let options = MessageOptions::default();

        let json =
            build_request_body(&messages, &options, "claude-3-sonnet-20240229", false).unwrap();
        let body: Value = parse_json(&json);

        assert_eq!(body["model"], json!("claude-3-sonnet-20240229"));
        assert_eq!(body["max_tokens"], json!(1024));
        assert_eq!(body["messages"][0]["role"], json!("user"));
        assert_eq!(body["messages"][0]["content"][0]["type"], json!("text"));
        assert_eq!(body["messages"][0]["content"][0]["text"], json!("Hello"));

        // Optional fields are omitted when unset.
        assert!(body.get("system").is_none());
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
        assert!(body.get("top_k").is_none());
        assert!(body.get("stop_sequences").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("metadata").is_none());
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn test_build_request_body_with_system() {
        let messages = vec![Message::system("You are helpful"), Message::user("Hello")];
        let options = MessageOptions::default();

        let json =
            build_request_body(&messages, &options, "claude-3-sonnet-20240229", false).unwrap();
        let body = parse_json(&json);

        assert_eq!(body["system"], json!("You are helpful"));
        assert_eq!(body["messages"][0]["role"], json!("user"));
    }

    #[test]
    fn test_build_request_body_with_tools() {
        let messages = vec![Message::user("What's the weather?")];
        let options = MessageOptions {
            tools: Some(vec![Tool::new(
                "get_weather",
                "Get weather for a location",
                json!({
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"]
                }),
            )]),
            tool_choice: Some(ToolChoice::Auto),
            ..Default::default()
        };

        let json =
            build_request_body(&messages, &options, "claude-3-sonnet-20240229", false).unwrap();
        let body = parse_json(&json);

        assert_eq!(body["tools"][0]["name"], json!("get_weather"));
        assert_eq!(
            body["tools"][0]["description"],
            json!("Get weather for a location")
        );
        assert_eq!(
            body["tools"][0]["input_schema"],
            json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            })
        );
        assert_eq!(body["tool_choice"], json!({ "type": "auto" }));
    }

    #[test]
    fn test_build_request_body_with_tool_result() {
        let messages = vec![
            Message::user("What's the weather in SF?"),
            Message::with_content(
                Role::Assistant,
                vec![Content::ToolUse(ToolUse {
                    id: "toolu_123".to_string(),
                    name: "get_weather".to_string(),
                    input: json!({ "location": "San Francisco" }),
                })],
            ),
            Message::tool_result("toolu_123", "72F and sunny", false),
        ];
        let options = MessageOptions::default();

        let json =
            build_request_body(&messages, &options, "claude-3-sonnet-20240229", false).unwrap();
        let body = parse_json(&json);

        // Assistant tool_use block (messages[1]).
        assert_eq!(body["messages"][1]["content"][0]["type"], json!("tool_use"));
        assert_eq!(body["messages"][1]["content"][0]["id"], json!("toolu_123"));
        assert_eq!(
            body["messages"][1]["content"][0]["name"],
            json!("get_weather")
        );
        assert_eq!(
            body["messages"][1]["content"][0]["input"],
            json!({ "location": "San Francisco" })
        );

        // Tool result block (messages[2]).
        assert_eq!(
            body["messages"][2]["content"][0]["type"],
            json!("tool_result")
        );
        assert_eq!(
            body["messages"][2]["content"][0]["tool_use_id"],
            json!("toolu_123")
        );
        assert_eq!(
            body["messages"][2]["content"][0]["content"],
            json!("72F and sunny")
        );
        assert!(body["messages"][2]["content"][0].get("is_error").is_none());
    }

    #[test]
    fn test_build_request_body_with_tool_result_error() {
        let messages = vec![
            Message::user("What's the weather in SF?"),
            Message::tool_result("toolu_123", "Failed to fetch weather", true),
        ];
        let options = MessageOptions::default();

        let json =
            build_request_body(&messages, &options, "claude-3-sonnet-20240229", false).unwrap();
        let body = parse_json(&json);

        assert_eq!(
            body["messages"][1]["content"][0]["type"],
            json!("tool_result")
        );
        assert_eq!(
            body["messages"][1]["content"][0]["content"],
            json!("Failed to fetch weather")
        );
        assert_eq!(body["messages"][1]["content"][0]["is_error"], json!(true));
    }

    #[test]
    fn test_build_request_body_streaming() {
        let messages = vec![Message::user("Hello")];
        let options = MessageOptions::default();

        let json =
            build_request_body(&messages, &options, "claude-3-sonnet-20240229", true).unwrap();
        let body = parse_json(&json);
        assert_eq!(body["stream"], json!(true));

        let json =
            build_request_body(&messages, &options, "claude-3-sonnet-20240229", false).unwrap();
        let body = parse_json(&json);
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn test_parse_response_text() {
        let response = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! How can I help you?"}],
            "model": "claude-3-sonnet-20240229",
            "stop_reason": "end_turn"
        }"#;

        let msg = parse_response(response).unwrap();
        assert_eq!(msg.role, Role::Assistant);
        match &msg.content[0] {
            Content::Text(t) => assert_eq!(t, "Hello! How can I help you?"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_parse_response_tool_use() {
        let response = r#"{
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me check the weather."},
                {"type": "tool_use", "id": "toolu_456", "name": "get_weather", "input": {"location": "SF"}}
            ],
            "model": "claude-3-sonnet-20240229",
            "stop_reason": "tool_use"
        }"#;

        let msg = parse_response(response).unwrap();
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content.len(), 2);

        match &msg.content[0] {
            Content::Text(t) => assert_eq!(t, "Let me check the weather."),
            _ => panic!("Expected text content"),
        }

        match &msg.content[1] {
            Content::ToolUse(tu) => {
                assert_eq!(tu.id, "toolu_456");
                assert_eq!(tu.name, "get_weather");
                assert_eq!(tu.input, json!({ "location": "SF" }));
            }
            _ => panic!("Expected tool use content"),
        }
    }

    #[test]
    fn test_parse_response_error() {
        let response = r#"{
            "type": "error",
            "error": {
                "type": "invalid_api_key",
                "message": "Invalid API key provided"
            }
        }"#;

        let err = parse_response(response).unwrap_err();
        assert_eq!(err.error_code, "invalid_api_key");
    }
}
