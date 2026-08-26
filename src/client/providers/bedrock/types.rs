//! Bedrock Converse API request/response types and serialization.

use serde_json::{Map, Value, json};

use crate::client::error::LlmError;
use crate::client::models::{Content, Message, MessageOptions, Role, Tool, ToolChoice, ToolUse};
use crate::client::providers::common::extract_text_content;

// Constants

/// Bedrock runtime API base URL template.
const API_BASE_TEMPLATE: &str = "https://bedrock-runtime.{region}.amazonaws.com";

/// Converse API path template.
const CONVERSE_PATH: &str = "/model/{model}/converse";

/// Converse stream API path template.
const CONVERSE_STREAM_PATH: &str = "/model/{model}/converse-stream";

/// Error code for invalid request.
const ERROR_INVALID_REQUEST: &str = "INVALID_REQUEST";

/// Error code for parse errors.
const ERROR_PARSE: &str = "PARSE_ERROR";

/// Error message when no messages provided.
const MSG_NO_MESSAGES: &str = "At least one message is required";

/// Error message when no content in response.
const MSG_NO_CONTENT: &str = "No content found in response";

// Public API

/// Returns the Bedrock Converse API endpoint URL.
pub fn get_converse_url(region: &str, model: &str) -> String {
    let base = API_BASE_TEMPLATE.replace("{region}", region);
    let path = CONVERSE_PATH.replace("{model}", &urlencoding::encode(model));
    format!("{}{}", base, path)
}

/// Returns the Bedrock Converse Stream API endpoint URL.
pub fn get_converse_stream_url(region: &str, model: &str) -> String {
    let base = API_BASE_TEMPLATE.replace("{region}", region);
    let path = CONVERSE_STREAM_PATH.replace("{model}", &urlencoding::encode(model));
    format!("{}{}", base, path)
}

/// Builds the JSON request body for the Bedrock Converse API.
///
/// Bedrock Converse format:
/// ```json
/// {
///   "messages": [{"role": "user", "content": [{"text": "..."}]}],
///   "system": [{"text": "..."}],
///   "inferenceConfig": {"maxTokens": 1024, "temperature": 0.7},
///   "toolConfig": {"tools": [...]}
/// }
/// ```
pub fn build_request_body(
    messages: &[Message],
    options: &MessageOptions,
) -> Result<String, LlmError> {
    // Separate system messages from conversation
    let mut system_texts: Vec<String> = Vec::new();
    let mut conversation_messages: Vec<&Message> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                for content in &msg.content {
                    if let Content::Text(text) = content {
                        system_texts.push(text.clone());
                    }
                }
            }
            Role::User | Role::Assistant => {
                conversation_messages.push(msg);
            }
        }
    }

    if conversation_messages.is_empty() {
        return Err(LlmError::new(ERROR_INVALID_REQUEST, MSG_NO_MESSAGES));
    }

    let mut body = Map::new();

    // Messages array
    let messages_arr: Vec<Value> = conversation_messages
        .iter()
        .map(|msg| message_to_value(msg))
        .collect();
    body.insert("messages".to_string(), Value::Array(messages_arr));

    // System instruction (optional)
    if !system_texts.is_empty() {
        let system_arr: Vec<Value> = system_texts.iter().map(|t| json!({ "text": t })).collect();
        body.insert("system".to_string(), Value::Array(system_arr));
    }

    // Inference config (optional)
    let mut inference = Map::new();
    if let Some(max_tokens) = options.max_tokens {
        inference.insert("maxTokens".to_string(), json!(max_tokens));
    }
    if let Some(temp) = options.temperature {
        inference.insert("temperature".to_string(), json!(temp));
    }
    if let Some(top_p) = options.top_p {
        inference.insert("topP".to_string(), json!(top_p));
    }
    if let Some(stop_sequences) = &options.stop_sequences
        && !stop_sequences.is_empty()
    {
        inference.insert("stopSequences".to_string(), json!(stop_sequences));
    }
    if !inference.is_empty() {
        body.insert("inferenceConfig".to_string(), Value::Object(inference));
    }

    // Tool config (optional)
    if let Some(tools) = &options.tools
        && !tools.is_empty()
    {
        let mut tool_config = Map::new();

        let tools_arr: Vec<Value> = tools.iter().map(tool_to_value).collect();
        tool_config.insert("tools".to_string(), Value::Array(tools_arr));

        if let Some(tool_choice) = &options.tool_choice {
            if let Some(choice_val) = tool_choice_to_value(tool_choice) {
                tool_config.insert("toolChoice".to_string(), choice_val);
            }
        }

        body.insert("toolConfig".to_string(), Value::Object(tool_config));
    }

    serde_json::to_string(&body).map_err(|e| LlmError::new("SERIALIZE_ERROR", e.to_string()))
}

/// Parses the Bedrock Converse API response and extracts the assistant message.
pub fn parse_response(response_body: &str) -> Result<Message, LlmError> {
    let parsed: Value = serde_json::from_str(response_body)
        .map_err(|e| LlmError::new(ERROR_PARSE, format!("Failed to parse response: {}", e)))?;

    // Check for API error response
    if let Some(error_msg) = parsed["message"].as_str()
        && parsed.get("output").is_none()
    {
        let error_type = parsed["__type"].as_str().unwrap_or("BedrockError");
        return Err(LlmError::new(error_type, error_msg));
    }

    // Extract output message
    let output = &parsed["output"]["message"];
    if output.is_null() {
        return Err(LlmError::new(ERROR_PARSE, "No content found in response"));
    }

    let mut content_blocks: Vec<Content> = Vec::new();

    if let Some(content_arr) = output["content"].as_array() {
        for block in content_arr {
            if let Some(text) = block["text"].as_str()
                && !text.is_empty()
            {
                content_blocks.push(Content::Text(text.to_string()));
            }
            if let Some(tool_use) = block.get("toolUse") {
                let id = tool_use["toolUseId"].as_str().unwrap_or("").to_string();
                let name = tool_use["name"].as_str().unwrap_or("").to_string();
                let input = tool_use["input"].clone();
                content_blocks.push(Content::ToolUse(ToolUse { id, name, input }));
            }
        }
    }

    if content_blocks.is_empty() {
        return Err(LlmError::new(ERROR_PARSE, "No content found in response"));
    }

    Ok(Message::with_content(Role::Assistant, content_blocks))
}

// Private Helpers

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
        Content::Text(text) => json!({ "text": text }),
        Content::Image(_) => {
            // TODO: Bedrock image support requires base64 + media type
            json!({ "text": "[unsupported image content]" })
        }
        Content::ToolUse(tool_use) => json!({
            "toolUse": {
                "toolUseId": tool_use.id,
                "name": tool_use.name,
                "input": tool_use.input
            }
        }),
        Content::ToolResult(tool_result) => {
            let status = if tool_result.is_error {
                "error"
            } else {
                "success"
            };
            json!({
                "toolResult": {
                    "toolUseId": tool_result.tool_use_id,
                    "content": [{ "text": tool_result.content }],
                    "status": status
                }
            })
        }
    }
}

fn tool_to_value(tool: &Tool) -> Value {
    json!({
        "toolSpec": {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": { "json": tool.input_schema }
        }
    })
}

fn tool_choice_to_value(choice: &ToolChoice) -> Option<Value> {
    match choice {
        ToolChoice::Auto => Some(json!({ "auto": {} })),
        ToolChoice::Any => Some(json!({ "any": {} })),
        ToolChoice::Tool(name) => Some(json!({ "tool": { "name": name }})),
        ToolChoice::None => None, // Bedrock has no "none" - omit tool config entirely
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json_str: &str) -> Value {
        serde_json::from_str(json_str).unwrap()
    }

    #[test]
    fn test_get_converse_url() {
        let url = get_converse_url("us-east-1", "anthropic.claude-3-sonnet-20240229-v1:0");
        assert_eq!(
            url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-sonnet-20240229-v1%3A0/converse"
        );
    }

    #[test]
    fn test_get_converse_stream_url() {
        let url = get_converse_stream_url("us-west-2", "anthropic.claude-3-haiku-20240307-v1:0");
        assert_eq!(
            url,
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3-haiku-20240307-v1%3A0/converse-stream"
        );
    }

    #[test]
    fn test_build_request_body_simple() {
        let messages = vec![Message::user("Hello")];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options).unwrap());

        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["text"], "Hello");
        assert!(body.get("system").is_none());
        assert!(body.get("inferenceConfig").is_none());
        assert!(body.get("toolConfig").is_none());
    }

    #[test]
    fn test_build_request_body_with_system() {
        let messages = vec![Message::system("You are helpful"), Message::user("Hello")];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options).unwrap());

        assert_eq!(body["system"][0]["text"], "You are helpful");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"][0]["role"], "user");
    }

    #[test]
    fn test_build_request_body_with_options() {
        let messages = vec![Message::user("Hello")];
        let options = MessageOptions {
            max_tokens: Some(1024),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop_sequences: Some(vec!["STOP".to_string()]),
            ..Default::default()
        };

        let body = parse(&build_request_body(&messages, &options).unwrap());

        assert_eq!(body["inferenceConfig"]["maxTokens"], 1024);
        assert_eq!(body["inferenceConfig"]["temperature"], 0.7);
        assert_eq!(body["inferenceConfig"]["topP"], 0.9);
        assert_eq!(body["inferenceConfig"]["stopSequences"], json!(["STOP"]));
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
            tool_choice: Some(ToolChoice::Auto),
            ..Default::default()
        };

        let body = parse(&build_request_body(&messages, &options).unwrap());

        assert_eq!(
            body["toolConfig"]["tools"][0]["toolSpec"]["name"],
            "get_weather"
        );
        assert_eq!(
            body["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"],
            json!({ "type": "object", "properties": { "location": { "type": "string" } } })
        );
        assert_eq!(body["toolConfig"]["toolChoice"], json!({ "auto": {} }));
    }

    #[test]
    fn test_build_request_body_with_tool_use_and_result() {
        let messages = vec![
            Message::user("What's the weather in SF?"),
            Message::with_content(
                Role::Assistant,
                vec![Content::ToolUse(ToolUse {
                    id: "call_123".to_string(),
                    name: "get_weather".to_string(),
                    input: json!({ "location": "San Francisco" }),
                })],
            ),
            Message::tool_result("call_123", "72F and sunny", false),
        ];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options).unwrap());

        // Assistant tool use
        let tool_use = &body["messages"][1]["content"][0]["toolUse"];
        assert_eq!(tool_use["toolUseId"], "call_123");
        assert_eq!(tool_use["name"], "get_weather");
        assert_eq!(tool_use["input"], json!({ "location": "San Francisco" }));

        // Tool result
        let tool_result = &body["messages"][2]["content"][0]["toolResult"];
        assert_eq!(tool_result["toolUseId"], "call_123");
        assert_eq!(tool_result["content"][0]["text"], "72F and sunny");
        assert_eq!(tool_result["status"], "success");
    }

    #[test]
    fn test_build_request_body_tool_result_error() {
        let messages = vec![
            Message::user("Do something"),
            Message::tool_result("call_456", "Something went wrong", true),
        ];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options).unwrap());

        let tool_result = &body["messages"][1]["content"][0]["toolResult"];
        assert_eq!(tool_result["status"], "error");
        assert_eq!(tool_result["content"][0]["text"], "Something went wrong");
    }

    #[test]
    fn test_build_request_body_empty_messages() {
        let messages = vec![Message::system("system only")];
        let options = MessageOptions::default();

        let err = build_request_body(&messages, &options).unwrap_err();
        assert_eq!(err.error_code, ERROR_INVALID_REQUEST);
    }

    #[test]
    fn test_parse_response_text() {
        let response = r#"{
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello! How can I help?"}]
                }
            },
            "stopReason": "end_turn"
        }"#;

        let msg = parse_response(response).unwrap();
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            Content::Text(t) => assert_eq!(t, "Hello!, How can I help?"),
            _ => panic!("Expected text content"),
        }
    }

    #[test]
    fn test_parse_response_tool_use() {
        let response = r#"{
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "call_123", "name": "get_weather", "input": {"location": "SF"}}}
                    ]
                }
            },
            "stopReason": "tool_use"
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
    fn test_parse_response_error() {
        let response = r#"{
            "__type": "ValidationException",
            "message": "Invalid model ID"
        }"#;

        let err = parse_response(response).unwrap_err();
        assert_eq!(err.error_code, "ValidationException");
    }

    #[test]
    fn test_parse_response_no_content() {
        let response = r#"{
            "output": {
                "message": {
                    "role": "assistant",
                    "content": []
                }
            }
        }"#;

        let err = parse_response(response).unwrap_err();
        assert_eq!(err.error_code, ERROR_PARSE);
    }
}
