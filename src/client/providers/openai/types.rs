//! OpenAI Chat Completions API request/response types and serialization.

use serde_json::{Map, Value, json};

use crate::client::error::LlmError;
use crate::client::models::{
    Content, ImageSource, Message, MessageOptions, Role, Tool, ToolChoice, ToolUse,
};

/// Default OpenAI API endpoint.
const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";
const CHAT_COMPLETIONS_PATH: &str = "/chat/completions";
const AZURE_API_URL_TEMPLATE: &str = "https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={version}";

const ERROR_INVALID_REQUEST: &str = "INVALID_REQUEST";
const ERROR_PARSE: &str = "PARSE_ERROR";

// === Public API ===

/// Returns the API endpoint URL, appending the chat completions path as needed.
pub fn get_api_url_with_base(base_url: Option<&str>) -> String {
    match base_url {
        Some(base) => {
            let base = base.trim_end_matches('/');
            if base.ends_with("/chat/completions") {
                base.to_string()
            } else if base.ends_with("/v1") {
                format!("{}{}", base, CHAT_COMPLETIONS_PATH)
            } else {
                format!("{}/v1{}", base, CHAT_COMPLETIONS_PATH)
            }
        }
        None => OPENAI_API_URL.to_string(),
    }
}

/// Returns the Azure OpenAI API endpoint URL.
pub fn get_azure_api_url(resource: &str, deployment: &str, api_version: &str) -> String {
    AZURE_API_URL_TEMPLATE
        .replace("{resource}", resource)
        .replace("{deployment}", deployment)
        .replace("{version}", api_version)
}

/// Returns standard OpenAI request headers.
pub fn get_request_headers(api_key: &str) -> Vec<(&'static str, String)> {
    vec![
        ("Content-Type", "application/json".to_string()),
        ("Authorization", format!("Bearer {}", api_key)),
    ]
}

/// Returns Azure OpenAI request headers.
pub fn get_azure_request_headers(api_key: &str) -> Vec<(&'static str, String)> {
    vec![
        ("Content-Type", "application/json".to_string()),
        ("api-key", api_key.to_string()),
    ]
}

/// Builds the JSON request body for the OpenAI Chat Completions API.
pub fn build_request_body(
    messages: &[Message],
    options: &MessageOptions,
    default_model: &str,
    streaming: bool,
) -> Result<String, LlmError> {
    if messages.is_empty() {
        return Err(LlmError::new(
            ERROR_INVALID_REQUEST,
            "At least one message is required",
        ));
    }

    let model = options.model.as_deref().unwrap_or(default_model);

    let mut body = Map::new();
    body.insert("model".to_string(), json!(model));

    if streaming {
        body.insert("stream".to_string(), json!(true));
        body.insert(
            "stream_options".to_string(),
            json!({ "include_usage": true }),
        );
    }

    // Optional parameters.
    if let Some(max_tokens) = options.max_tokens {
        body.insert("max_tokens".to_string(), json!(max_tokens));
    }
    if let Some(temp) = options.temperature {
        body.insert("temperature".to_string(), json!(temp));
    }
    if let Some(top_p) = options.top_p {
        body.insert("top_p".to_string(), json!(top_p));
    }
    if let Some(stop_sequences) = &options.stop_sequences
        && !stop_sequences.is_empty()
    {
        body.insert("stop".to_string(), json!(stop_sequences));
    }

    // Tools.
    if let Some(tools) = &options.tools
        && !tools.is_empty()
    {
        let tools_arr: Vec<Value> = tools.iter().map(tool_to_value).collect();
        body.insert("tools".to_string(), Value::Array(tools_arr));
    }

    // Tool choice.
    if let Some(tool_choice) = &options.tool_choice {
        if let Some(choice_val) = tool_choice_to_value(tool_choice) {
            body.insert("tool_choice".to_string(), choice_val);
        }
    }

    // User ID.
    if let Some(user_id) = options.metadata.as_ref().and_then(|m| m.user_id.as_ref()) {
        body.insert("user".to_string(), json!(user_id));
    }

    // Messages array.
    let messages_arr: Vec<Value> = messages
        .iter()
        .map(message_to_value)
        .collect::<Result<Vec<_>, _>>()?;
    body.insert("messages".to_string(), Value::Array(messages_arr));

    serde_json::to_string(&body).map_err(|e| LlmError::new("SERIALIZE_ERROR", e.to_string()))
}

/// Parses the OpenAI API response and extracts the assistant message.
pub fn parse_response(response_body: &str) -> Result<Message, LlmError> {
    let parsed: Value = serde_json::from_str(response_body)
        .map_err(|e| LlmError::new(ERROR_PARSE, format!("Failed to parse response: {}", e)))?;

    // API error.
    if let Some(error_msg) = parsed["error"]["message"].as_str() {
        let error_type = parsed["error"]["type"].as_str().unwrap_or("api_error");
        return Err(LlmError::new(error_type, error_msg));
    }

    let message = &parsed["choices"][0]["message"];
    if message.is_null() {
        return Err(LlmError::new(ERROR_PARSE, "No message found in response"));
    }

    let mut content_blocks: Vec<Content> = Vec::new();

    // Text content (may be null when tool calls are present).
    if let Some(text) = message["content"].as_str()
        && !text.is_empty()
    {
        content_blocks.push(Content::Text(text.to_string()));
    }

    // Tool calls.
    if let Some(tool_calls) = message["tool_calls"].as_array() {
        for tc in tool_calls {
            content_blocks.push(Content::ToolUse(parse_tool_call(tc)));
        }
    }

    if content_blocks.is_empty() {
        return Err(LlmError::new(ERROR_PARSE, "No content found in response"));
    }

    Ok(Message::with_content(Role::Assistant, content_blocks))
}

// === Private Helpers ===

fn message_to_value(msg: &Message) -> Result<Value, LlmError> {
    let has_tool_results = msg
        .content
        .iter()
        .any(|c| matches!(c, Content::ToolResult(_)));
    let has_tool_calls = msg.content.iter().any(|c| matches!(c, Content::ToolUse(_)));

    // Tool results use role "tool".
    if has_tool_results {
        if let Some(Content::ToolResult(tr)) = msg
            .content
            .iter()
            .find(|c| matches!(c, Content::ToolResult(_)))
        {
            return Ok(json!({
                "role": "tool",
                "tool_call_id": tr.tool_use_id,
                "content": tr.content
            }));
        }
    }

    // Assistant messages with tool calls.
    if has_tool_calls {
        let mut obj = Map::new();
        obj.insert("role".to_string(), json!("assistant"));

        let text: String = msg
            .content
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if text.is_empty() {
            obj.insert("content".to_string(), Value::Null);
        } else {
            obj.insert("content".to_string(), json!(text));
        }

        let tool_calls: Vec<Value> = msg
            .content
            .iter()
            .filter_map(|c| match c {
                Content::ToolUse(tu) => Some(json!({
                    "id": tu.id,
                    "type": "function",
                    "function": {
                        "name": tu.name,
                        "arguments": tu.input,
                    },
                })),
                _ => None,
            })
            .collect();
        obj.insert("tool_calls".to_string(), Value::Array(tool_calls));

        return Ok(Value::Object(obj));
    }

    // Regular messages.
    let role = match msg.role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
    };

    // Simple single-text message -> content as string.
    if msg.content.len() == 1 && matches!(&msg.content[0], Content::Text(_)) {
        if let Content::Text(text) = &msg.content[0] {
            return Ok(json!({ "role": role, "content": text }));
        }
    }

    // Complex content (images, multiple parts) -> content as array.
    let parts: Vec<Value> = msg
        .content
        .iter()
        .filter_map(|c| content_part_to_value(c))
        .collect();

    Ok(json!({ "role": role, "content": parts }))
}

fn content_part_to_value(content: &Content) -> Option<Value> {
    match content {
        Content::Text(text) => Some(json!({ "type": "text", "text": text })),
        Content::Image(ImageSource::Base64 { media_type, data }) => Some(json!({
            "type": "image_url",
            "image_url": { "url": format!("data:{};base64,{}", media_type, data) }
        })),
        Content::Image(ImageSource::Url(url)) => Some(json!({
            "type": "image_url",
            "image_url": { "url": url }
        })),
        Content::ToolUse(_) | Content::ToolResult(_) => None,
    }
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
        ToolChoice::Auto => Some(json!("auto")),
        ToolChoice::Any => Some(json!("required")),
        ToolChoice::None => Some(json!("none")),
        ToolChoice::Tool(name) => Some(json!({
            "type": "function",
            "function": { "name": name }
        })),
    }
}

/// Parse a tool call from the response into a ToolUse.
fn parse_tool_call(tc: &Value) -> ToolUse {
    let id = tc["id"].as_str().unwrap_or_default().to_string();
    let name = tc["function"]["name"]
        .as_str()
        .unwrap_or_default()
        .to_string();

    // OpenAI returns arguments as a JSON string.
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
    fn test_get_api_url_default() {
        assert_eq!(
            get_api_url_with_base(None),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_get_api_url_full_path() {
        assert_eq!(
            get_api_url_with_base(Some("https://api.perplexity.ai/chat/completions")),
            "https://api.perplexity.ai/chat/completions"
        );
    }

    #[test]
    fn test_get_api_url_v1() {
        assert_eq!(
            get_api_url_with_base(Some("https://api.groq.com/openai/v1")),
            "https://api.groq.com/openai/v1/chat/completions"
        );
    }

    #[test]
    fn test_get_api_url_bare() {
        assert_eq!(
            get_api_url_with_base(Some("https://api.deepseek.com")),
            "https://api.deepseek.com/v1/chat/completions"
        )
    }

    #[test]
    fn test_get_api_url_trailing_slash() {
        assert_eq!(
            get_api_url_with_base(Some("https://api.groq.com/openai/v1/")),
            "https://api.groq.com/openai/v1/chat/completions"
        );
    }

    #[test]
    fn test_get_azure_api_url() {
        assert_eq!(
            get_azure_api_url("my-resource", "gpt-4-deploy", "2024-10-21"),
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4-deploy/chat/completions?api-version=2024-10-21"
        );
    }

    #[test]
    fn test_get_request_headers() {
        let headers = get_request_headers("sk-test");
        assert_eq!(headers[0], ("Content-Type", "application/json".to_string()));
        assert_eq!(headers[1], ("Authorization", "Bearer sk-test".to_string()));
    }

    #[test]
    fn test_get_azure_request_headers() {
        let headers = get_azure_request_headers("azure-key");
        assert_eq!(headers[0], ("Content-Type", "application/json".to_string()));
        assert_eq!(headers[1], ("api-key", "azure-key".to_string()));
    }

    #[test]
    fn test_build_request_body_simple() {
        let messages = vec![Message::user("Hello")];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options, "gpt-4", false).unwrap());

        assert_eq!(body["model"], "gpt-4");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "Hello");
        assert!(body.get("stream").is_none());
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn test_build_request_body_streaming() {
        let messages = vec![Message::user("Hello")];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options, "gpt-4", true).unwrap());

        assert_eq!(body["stream"], true);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn test_build_request_body_with_options() {
        let messages = vec![Message::user("Hi")];
        let options = MessageOptions {
            max_tokens: Some(512),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stop_sequences: Some(vec!["END".to_string()]),
            ..Default::default()
        };

        let body = parse(&build_request_body(&messages, &options, "gpt-4", false).unwrap());

        assert_eq!(body["max_tokens"], 512);
        assert_eq!(body["temperature"], 0.7);
        assert_eq!(body["top_p"], 0.9);
        assert_eq!(body["stop"], json!(["END"]));
    }

    #[test]
    fn test_build_request_body_with_system() {
        let messages = vec![Message::system("Be helpful"), Message::user("Hello")];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options, "gpt-4", false).unwrap());

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
                json!({
                    "type": "object",
                    "properties": { "location": { "type": "string " } },
                    "required": ["location"]
                }),
            )]),
            tool_choice: Some(ToolChoice::Auto),
            ..Default::default()
        };

        let body = parse(&build_request_body(&messages, &options, "gpt-4", false).unwrap());

        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(
            body["tools"][0]["function"]["parameters"],
            json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            })
        );
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn test_build_request_body_tool_choice_variants() {
        let messages = vec![Message::user("Hi")];
        let tools = Some(vec![Tool::new("t", "d", json!({}))]);

        let body = parse(
            &build_request_body(
                &messages,
                &MessageOptions {
                    tools: tools.clone(),
                    tool_choice: Some(ToolChoice::Any),
                    ..Default::default()
                },
                "gpt-4",
                false,
            )
            .unwrap(),
        );
        assert_eq!(body["tool_choice"], "required");

        let body = parse(
            &build_request_body(
                &messages,
                &MessageOptions {
                    tools: tools.clone(),
                    tool_choice: Some(ToolChoice::None),
                    ..Default::default()
                },
                "gpt-4",
                false,
            )
            .unwrap(),
        );
        assert_eq!(body["tool_choice"], "none");

        let body = parse(
            &build_request_body(
                &messages,
                &MessageOptions {
                    tools,
                    tool_choice: Some(ToolChoice::Tool("search".to_string())),
                    ..Default::default()
                },
                "gpt-4",
                false,
            )
            .unwrap(),
        );
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
                    id: "call_123".to_string(),
                    name: "get_weather".to_string(),
                    input: json!({ "location": "SF" }),
                })],
            ),
            Message::tool_result("call_123", "72F sunny", false),
        ];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options, "gpt-4", false).unwrap());

        // Assistant with tool calls.
        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(body["messages"][1]["content"], Value::Null);
        assert_eq!(body["messages"][1]["tool_calls"][0]["id"], "call_123");
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["arguments"],
            json!({ "location": "SF" })
        );

        // Tool result.
        assert_eq!(body["messages"][2]["role"], "tool");
        assert_eq!(body["messages"][2]["tool_call_id"], "call_123");
        assert_eq!(body["messages"][2]["content"], "72F sunny");
    }

    #[test]
    fn test_build_request_body_image_url() {
        let messages = vec![Message::with_content(
            Role::User,
            vec![
                Content::Text("What's in this image?".to_string()),
                Content::Image(ImageSource::Url("https://example.com/img.png".to_string())),
            ],
        )];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options, "gpt-4", false).unwrap());

        let parts = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "https://example.com/img.png");
    }

    #[test]
    fn test_build_request_body_empty_messages() {
        let err = build_request_body(&[], &MessageOptions::default(), "gpt-4", false).unwrap_err();
        assert_eq!(err.error_code, ERROR_INVALID_REQUEST);
    }

    #[test]
    fn test_parse_response_text() {
        let response = r#"{
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "Hello!" },
                "finish_reason": "stop"
            }]
        }"#;

        let msg = parse_response(response).unwrap();
        assert_eq!(msg.role, Role::Assistant);
        match &msg.content[0] {
            Content::Text(t) => assert_eq!(t, "Hello!"),
            _ => panic!("Expected text"),
        }
    }

    #[test]
    fn test_parse_response_tool_calls_string_args() {
        let response = r#"{
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_456",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"SF\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }"#;

        let msg = parse_response(response).unwrap();
        assert_eq!(msg.content.len(), 1);
        match &msg.content[0] {
            Content::ToolUse(tu) => {
                assert_eq!(tu.id, "call_456");
                assert_eq!(tu.name, "get_weather");
                assert_eq!(tu.input, json!({ "location": "SF" }));
            }
            _ => panic!("Expected tool use"),
        }
    }

    #[test]
    fn test_parse_response_error() {
        let response = r#"{
            "error": { "message": "Invalid API key", "type": "invalid_request_error" }
        }"#;

        let err = parse_response(response).unwrap_err();
        assert_eq!(err.error_code, "invalid_request_error");
    }

    #[test]
    fn test_parse_response_no_message() {
        let response = r#"{"choices": [{}]}"#;
        let err = parse_response(response).unwrap_err();
        assert_eq!(err.error_code, ERROR_PARSE);
    }
}
