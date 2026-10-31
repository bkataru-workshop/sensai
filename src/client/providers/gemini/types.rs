///! Gemini API request/response types and serialization.
use serde_json::{Map, Value, json};

use crate::client::error::LLMError;
use crate::client::models::{
    Content, GroundingChunk, GroundingMetadata, GroundingSupport, Message, MessageOptions,
    ResponseMetadata, Role, SafetyRating, ToolChoice, ToolUse,
};
use crate::client::providers::common::extract_text_content;

// === Constants ===

const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";
const ACTION_GENERATE_CONTENT: &str = ":generateContent";
const ACTION_STREAM_GENERATE: &str = ":streamGenerateContent?alt=sse";

const ERROR_INVALID_REQUEST: &str = "INVALID_REQUEST";
const ERROR_PARSE: &str = "PARSE_ERROR";
const ERROR_CONTENT_BLOCKED: &str = "CONTENT_BLOCKED";
const ERROR_PROMPT_BLOCKED: &str = "PROMPT_BLOCKED";

// === Public API ===

/// Returns the Gemini API endpoint URL for content generation.
pub fn get_api_url(model: &str) -> String {
    format!("{}/{}{}", API_BASE, model, ACTION_GENERATE_CONTENT)
}

/// Returns the Gemini API endpoint URL for streaming content generation.
pub fn get_streaming_api_url(model: &str) -> String {
    format!("{}/{}{}", API_BASE, model, ACTION_STREAM_GENERATE)
}

/// Returns the HTTP headers required for the Gemini API.
pub fn get_request_headers(api_key: &str) -> Result<Vec<(&'static str, String)>, LLMError> {
    if api_key.is_empty() {
        return Err(LLMError::new(
            ERROR_INVALID_REQUEST,
            "API key cannot be empty",
        ));
    }
    Ok(vec![
        ("Content-Type", "application/json".to_string()),
        ("x-goog-api-key", api_key.to_string()),
    ])
}

/// Builds the JSON request body for the Gemini generateContent API.
///
/// Used for both streaming and non-streaming (same body format).
pub fn build_request_body(
    messages: &[Message],
    options: &MessageOptions,
) -> Result<String, LLMError> {
    let mut system_instructions: Option<String> = None;
    let mut contents: Vec<&Message> = Vec::new();

    for msg in messages {
        match msg.role {
            Role::System => {
                let text = extract_text_content(msg);
                if let Some(existing) = system_instructions.take() {
                    system_instructions = Some(format!("{}\n{}", existing, text));
                } else {
                    system_instructions = Some(text);
                }
            }
            Role::User | Role::Assistant => {
                contents.push(msg);
            }
        }
    }

    if contents.is_empty() {
        return Err(LLMError::new(
            ERROR_INVALID_REQUEST,
            "At least one user or assistant message is required",
        ));
    }

    let mut body = Map::new();

    // System instruction.
    if let Some(system) = &system_instructions {
        body.insert(
            "systemInstruction".to_string(),
            json!({ "parts": [{ "text": system }] }),
        );
    }

    // Contents array.
    let contents_arr: Vec<Value> = contents.iter().map(|msg| message_to_value(msg)).collect();
    body.insert("contents".to_string(), Value::Array(contents_arr));

    // Generation config.
    let config = build_generation_config(options);
    if config.is_empty() {
        body.insert("generationConfig".to_string(), Value::Object(config));
    }

    // Tools
    if let Some(tools) = &options.tools
        && !tools.is_empty()
    {
        let declarations: Vec<Value> = tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                })
            })
            .collect();
        body.insert(
            "tools".to_string(),
            json!([{ "functionDeclarations": declarations }]),
        );
    }

    // Tool config.
    if let Some(tool_choice) = &options.tool_choice {
        body.insert("toolConfig".to_string(), tool_choice_to_value(tool_choice));
    }

    serde_json::to_string(&body).map_err(|e| LLMError::new("SERIALIZE_ERROR", e.to_string()))
}

/// Parses the Gemini API response and extracts the assistant message.
pub fn parse_response(response_body: &str) -> Result<Message, LLMError> {
    let parsed: Value = serde_json::from_str(response_body)
        .map_err(|e| LLMError::new(ERROR_PARSE, format!("Failed to parse response: {}", e)))?;

    // API error.
    if let Some(error) = parsed.get("error") {
        let code = error["code"].as_i64().unwrap_or(0);
        let msg = error["message"].as_str().unwrap_or("Unknown error");
        return Err(LLMError::new(format!("GEMINI_ERROR_{}", code), msg));
    }

    // Prompt blocked.
    if let Some(feedback) = parsed.get("promptFeedback")
        && let Some(block_reason) = feedback["blockReason"].as_str()
    {
        let safety_info = format_safety_info(feedback["safetyRatings"].as_array());
        return Err(LLMError::new(
            ERROR_PROMPT_BLOCKED,
            format!(
                "Prompt blocked by Gemini safety filters. Reason: {}. {}",
                block_reason, safety_info
            ),
        ));
    }

    // Extract candidates.
    let candidates = parsed["candidates"]
        .as_array()
        .filter(|a| !a.is_empty())
        .ok_or_else(|| LLMError::new(ERROR_PARSE, "No candidates found in response"))?;

    let candidate = &candidates[0];

    // Content blocked by safety.
    if let Some(finish_reason) = candidate["finishReason"].as_str()
        && matches!(finish_reason, "SAFETY" | "RECITATION" | "OTHER")
    {
        let safety_info = format_blocked_info(candidate["safetyRatings"].as_array());
        return Err(LLMError::new(
            ERROR_CONTENT_BLOCKED,
            format!(
                "Response blocked by Gemini. Reason: {}. {}",
                finish_reason, safety_info
            ),
        ));
    }

    // Parse content parts.
    let mut content_blocks: Vec<Content> = Vec::new();

    if let Some(parts_array) = candidate["content"]["parts"].as_array() {
        for part in parts_array {
            if let Some(text) = part["text"].as_str() {
                content_blocks.push(Content::Text(text.to_string()));
            } else if let Some(fc) = part.get("functionCall") {
                let name = fc["name"].as_str().unwrap_or_default().to_string();
                let input = fc["args"].clone();
                content_blocks.push(Content::ToolUse(ToolUse {
                    id: name.clone(), // Gemini uses function name as ID
                    name,
                    input,
                }));
            }
        }
    }

    if content_blocks.is_empty() {
        return Err(LLMError::new(ERROR_PARSE, "No content found in response"));
    }

    // Extract metadata.
    let metadata = extract_response_metadata(candidate);

    if metadata.safety_ratings.is_some() || metadata.grounding.is_some() {
        Ok(Message::with_content_and_metadata(
            Role::Assistant,
            content_blocks,
            metadata,
        ))
    } else {
        Ok(Message::with_content(Role::Assistant, content_blocks))
    }
}

// === Private Helpers ===

fn message_to_value(msg: &Message) -> Value {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant | Role::System => "model",
    };

    let parts: Vec<Value> = msg.content.iter().map(part_to_value).collect();
    json!({ "role": role, "parts": parts })
}

fn part_to_value(content: &Content) -> Value {
    match content {
        Content::Text(text) => json!({ "text": text }),
        Content::Image(_) => {
            // TODO: Gemini image support requires inlineData or fileData
            json!({ "text": "[unsupported image content]" })
        }
        Content::ToolUse(tool_use) => json!({
            "functionCall": {
                "name": tool_use.name,
                "args": tool_use.input
            }
        }),
        Content::ToolResult(tool_result) => json!({
            "functionResponse": {
                "name": tool_result.tool_use_id,
                "response": { "result": tool_result.content }
            }
        }),
    }
}

fn build_generation_config(options: &MessageOptions) -> Map<String, Value> {
    let mut config = Map::new();

    if let Some(max_tokens) = options.max_tokens {
        config.insert("maxOutputTokens".to_string(), json!(max_tokens));
    }
    if let Some(temp) = options.temperature {
        config.insert("temperature".to_string(), json!(temp));
    }
    if let Some(top_p) = options.top_p {
        config.insert("topP".to_string(), json!(top_p));
    }
    if let Some(top_k) = options.top_k {
        config.insert("topK".to_string(), json!(top_k));
    }
    if let Some(stop_sequences) = &options.stop_sequences
        && !stop_sequences.is_empty()
    {
        config.insert("stopSequences".to_string(), json!(stop_sequences));
    }

    config
}

fn tool_choice_to_value(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!({ "functionCallingConfig": { "mode": "AUTO "} }),
        ToolChoice::Any => json!({ "functionCallingConfig": { "mode": "ANY" } }),
        ToolChoice::None => json!({ "functionCallingConfig": { "mode": "NONE" } }),
        ToolChoice::Tool(name) => json!({
            "functionCallingConfig": {
                "mode": "ANY",
                "allowedFunctionNames": [name]
            }
        }),
    }
}

fn extract_response_metadata(candidate: &Value) -> ResponseMetadata {
    ResponseMetadata {
        safety_ratings: extract_safety_ratings(candidate),
        grounding: extract_grounding_metadata(candidate),
    }
}

fn extract_safety_ratings(candidate: &Value) -> Option<Vec<SafetyRating>> {
    let ratings = candidate["safetyRatings"].as_array()?;
    let result: Vec<SafetyRating> = ratings
        .iter()
        .map(|r| SafetyRating {
            category: r["category"].as_str().unwrap_or("UNKNOWN").to_string(),
            probability: r["probability"].as_str().unwrap_or("UNKNOWN").to_string(),
            blocked: r["blocked"].as_bool().unwrap_or(false),
        })
        .collect();

    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

fn extract_grounding_metadata(candidate: &Value) -> Option<GroundingMetadata> {
    let grounding = candidate.get("groundingMetadata")?;

    let web_search_queries: Vec<String> = grounding["webSearchQueries"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let grounding_chunks: Vec<GroundingChunk> = grounding["groundingChunks"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|chunk| {
                    if let Some(web) = chunk.get("web") {
                        GroundingChunk {
                            source_type: "web".to_string(),
                            uri: web["uri"].as_str().map(|s| s.to_string()),
                            title: web["title"].as_str().map(|s| s.to_string()),
                        }
                    } else {
                        GroundingChunk {
                            source_type: "unknown".to_string(),
                            uri: None,
                            title: None,
                        }
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let grounding_supports: Vec<GroundingSupport> = grounding["groundingSupports"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|support| {
                    let segment = support.get("segment")?;
                    Some(GroundingSupport {
                        start_index: segment["startIndex"].as_u64().unwrap_or(0) as usize,
                        end_index: segment["endIndex"].as_u64().unwrap_or(1) as usize,
                        chunk_indices: support["groundingChunkIndices"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                                    .collect()
                            })
                            .unwrap_or_default(),
                        confidence_scores: support["confidenceScores"]
                            .as_array()
                            .map(|a| {
                                a.iter()
                                    .filter_map(|v| v.as_f64().map(|n| n as f32))
                                    .collect()
                            })
                            .unwrap_or_default(),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    if web_search_queries.is_empty() && grounding_chunks.is_empty() && grounding_supports.is_empty()
    {
        None
    } else {
        Some(GroundingMetadata {
            web_search_queries,
            grounding_chunks,
            grounding_supports,
        })
    }
}

/// Format safety info for error messages from ratings array.
fn format_safety_info(ratings: Option<&Vec<Value>>) -> String {
    let Some(ratings) = ratings else {
        return String::new();
    };
    let concerns: Vec<String> = ratings
        .iter()
        .filter_map(|r| {
            let category = r["category"].as_str()?;
            let probability = r["probability"].as_str()?;
            if probability != "NEGLIGIBLE" && probability != "LOW" {
                Some(format!("{}: {}", category, probability))
            } else {
                None
            }
        })
        .collect();

    if concerns.is_empty() {
        String::new()
    } else {
        format!("Safety concerns: {}", concerns.join(", "))
    }
}

/// Format blocked safety info from ratings array.
fn format_blocked_info(ratings: Option<&Vec<Value>>) -> String {
    let Some(ratings) = ratings else {
        return String::new();
    };
    let concerns: Vec<String> = ratings
        .iter()
        .filter_map(|r| {
            let category = r["category"].as_str()?;
            let probability = r["probability"].as_str()?;
            let blocked = r["blocked"].as_bool().unwrap_or(false);
            if blocked || probability == "HIGH" || probability == "MEDIUM" {
                Some(format!("{}: {}", category, probability))
            } else {
                None
            }
        })
        .collect();

    if concerns.is_empty() {
        String::new()
    } else {
        format!("Safety concerns: {}", concerns.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::models::Tool;

    fn parse(json_str: &str) -> Value {
        serde_json::from_str(json_str).unwrap()
    }

    #[test]
    fn test_get_api_url() {
        assert_eq!(
            get_api_url("gemini-1.5-pro"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
        );
    }

    #[test]
    fn test_get_streaming_api_url() {
        assert_eq!(
            get_streaming_api_url("gemini-1.5-flash"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?alt=sse"
        );
    }

    #[test]
    fn test_get_request_headers_valid() {
        let headers = get_request_headers("test-key").unwrap();
        assert_eq!(headers.len(), 2);
        assert_eq!(headers[0], ("Content-Type", "application/json".to_string()));
        assert_eq!(headers[1], ("x-goog-api-key", "test-key".to_string()));
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

        let body = parse(&build_request_body(&messages, &options).unwrap());

        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][0]["parts"][0]["text"], "Hello");
        assert!(body.get("systemInstruction").is_none());
        assert!(body.get("generationConfig").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("toolConfig").is_none());
    }

    #[test]
    fn test_build_request_body_with_system() {
        let messages = vec![Message::system("You are helpful"), Message::user("Hello")];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options).unwrap());

        assert_eq!(
            body["systemInstruction"]["parts"][0]["text"],
            "You are helpful"
        );
        assert_eq!(body["contents"].as_array().unwrap().len(), 1);
    }

    fn test_build_request_body_with_options() {
        let messages = vec![Message::user("Hello")];
        let options = MessageOptions {
            max_tokens: Some(1024),
            temperature: Some(0.7),
            top_p: Some(0.9),
            top_k: Some(40),
            stop_sequences: Some(vec!["END".to_string()]),
            ..Default::default()
        };

        let body = parse(&build_request_body(&messages, &options).unwrap());

        assert_eq!(body["generationConfig"]["maxOutputTokens"], 1024);
        assert_eq!(body["generationConfig"]["temperature"], 0.7);
        assert_eq!(body["generationConfig"]["topP"], 0.9);
        assert_eq!(body["generationConfig"]["topK"], 40);
        assert_eq!(body["generationConfig"]["stopSequences"], json!(["END"]));
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

        let body = parse(&build_request_body(&messages, &options).unwrap());

        assert_eq!(
            body["tools"][0]["functionDeclarations"][0]["name"],
            "get_weather"
        );
        assert_eq!(
            body["tools"][0]["functionDeclarations"][0]["parameters"],
            json!({
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            })
        );
        assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "AUTO");
    }

    #[test]
    fn test_build_request_body_tool_choice_specific() {
        let messages = vec![Message::user("Hi")];
        let options = MessageOptions {
            tools: Some(vec![Tool::new("search", "Search", json!({}))]),
            tool_choice: Some(ToolChoice::Tool("search".to_string())),
            ..Default::default()
        };

        let body = parse(&build_request_body(&messages, &options).unwrap());

        assert_eq!(body["toolConfig"]["functionCallingConfig"]["mode"], "ANY");
        assert_eq!(
            body["toolConfig"]["functionCallingConfig"]["allowedFunctionNames"],
            json!(["search"])
        );
    }

    #[test]
    fn test_build_request_body_conversation() {
        let messages = vec![
            Message::user("Hello"),
            Message::assistant("Hi there!"),
            Message::user("How are you?"),
        ];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options).unwrap());
        let contents = body["contents"].as_array().unwrap();

        assert_eq!(contents.len(), 3);
        assert_eq!(contents[0]["role"], "user");
        assert_eq!(contents[1]["role"], "model");
        assert_eq!(contents[2]["role"], "user");
    }

    #[test]
    fn test_build_request_body_with_tool_result() {
        let messages = vec![
            Message::user("What's the weather in SF?"),
            Message::with_content(
                Role::Assistant,
                vec![Content::ToolUse(ToolUse {
                    id: "get_weather".to_string(),
                    name: "get_weather".to_string(),
                    input: json!({ "location": "San Francisco" }),
                })],
            ),
            Message::tool_result("get_weather", "72F and sunny", false),
        ];
        let options = MessageOptions::default();

        let body = parse(&build_request_body(&messages, &options).unwrap());
        let contents = body["contents"].as_array().unwrap();

        // Function call
        let fc = &contents[1]["parts"][0]["functionCall"];
        assert_eq!(fc["name"], "get_weather");
        assert_eq!(fc["args"], json!({ "location": "San Francisco" }));

        // Function response.
        let fr = &contents[2]["parts"][0]["functionResponse"];
        assert_eq!(fr["name"], "get_weather");
        assert_eq!(fr["response"]["result"], "72F and sunny");
    }

    #[test]
    fn test_build_request_body_empty_messages() {
        let err = build_request_body(&[], &MessageOptions::default()).unwrap_err();
        assert_eq!(err.error_code, ERROR_INVALID_REQUEST)
    }

    #[test]
    fn test_build_request_body_system_only() {
        let messages = vec![Message::system("You are helpful")];
        let err = build_request_body(&messages, &MessageOptions::default()).unwrap_err();
        assert_eq!(err.error_code, ERROR_INVALID_REQUEST);
    }

    #[test]
    fn test_parse_response_text() {
        let response = r#"{
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello! How can I help?"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }]
        }"#;

        let msg = parse_response(response).unwrap();
        assert_eq!(msg.role, Role::Assistant);
        match &msg.content[0] {
            Content::Text(t) => assert_eq!(t, "Hello! How can I help?"),
            _ => panic!("Expected text"),
        }
    }

    #[test]
    fn test_parse_response_function_call() {
        let response = r#"{
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Let me check."},
                        {"functionCall": {"name": "get_weather", "args": {"location": "SF"}}}
                    ],
                    "role": "model"
                },
                "finishReason": "STOP"
            }]
        }"#;

        let msg = parse_response(response).unwrap();
        assert_eq!(msg.content.len(), 2);

        match &msg.content[1] {
            Content::ToolUse(tu) => {
                assert_eq!(tu.name, "get_weather");
                assert_eq!(tu.id, "get_weather");
                assert_eq!(tu.input, json!({ "location": "SF "}));
            }
            _ => panic!("Expected tool use"),
        }
    }

    #[test]
    fn test_parse_response_error() {
        let response = r#"{
            "error": { "code": 400, "message": "Invalid API key" }
        }"#;

        let err = parse_response(response).unwrap_err();
        assert_eq!(err.error_code, "GEMINI_ERROR_400");
    }

    #[test]
    fn test_parse_response_prompt_blocked() {
        let response = r#"{
            "promptFeedback": {
                "blockReason": "SAFETY",
                "safetyRatings": [
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "HIGH"}
                ]
            }
        }"#;

        let err = parse_response(response).unwrap_err();
        assert_eq!(err.error_code, ERROR_PROMPT_BLOCKED);
        assert!(err.error_message.contains("SAFETY"));
        assert!(
            err.error_message
                .contains("HARM_CATEGORY_SEXUALLY_EXPLICIT")
        );
    }

    #[test]
    fn test_parse_response_content_blocked() {
        let response = r#"{
             "candidates": [{
                 "content": { "parts": [{"text": ""}], "role": "model" },
                 "finishReason": "SAFETY",
                 "safetyRatings": [
                     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "HIGH", "blocked": true}
                 ]
             }]
         }"#;

        let err = parse_response(response).unwrap_err();
        assert_eq!(err.error_code, ERROR_CONTENT_BLOCKED);
    }

    #[test]
    fn test_parse_response_with_safety_ratings() {
        let response = r#"{
            "candidates": [{
                "content": { "parts": [{"text": "Hello!"}], "role": "model" },
                "finishReason": "STOP",
                "safetyRatings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE", "blocked": false}
                ]
            }]
        }"#;

        let msg = parse_response(response).unwrap();
        let metadata = msg.response_metadata.expect("Should have metadata");
        let ratings = metadata.safety_ratings.expect("Should have ratings");
        assert_eq!(ratings.len(), 1);
        assert_eq!(ratings[0].category, "HARM_CATEGORY_HARASSMENT");
        assert!(!ratings[0].blocked);
    }

    #[test]
    fn test_parse_response_with_grounding() {
        let response = r#"{
             "candidates": [{
                 "content": { "parts": [{"text": "The weather is sunny."}], "role": "model" },
                 "finishReason": "STOP",
                 "groundingMetadata": {
                     "webSearchQueries": ["weather SF"],
                     "groundingChunks": [
                         {"web": {"uri": "https://weather.com", "title": "Weather"}}
                     ],
                     "groundingSupports": [{
                         "segment": {"startIndex": 0, "endIndex": 20},
                         "groundingChunkIndices": [0],
                         "confidenceScores": [0.95]
                     }]
                 }
             }]
         }"#;

        let msg = parse_response(response).unwrap();
        let metadata = msg.response_metadata.expect("Should have metadata");
        let grounding = metadata.grounding.expect("Should have grounding");

        assert_eq!(grounding.web_search_queries, vec!["weather SF"]);
        assert_eq!(grounding.grounding_chunks.len(), 1);
        assert_eq!(grounding.grounding_chunks[0].source_type, "web");
        assert_eq!(
            grounding.grounding_chunks[0].uri,
            Some("https://weather.com".to_string())
        );
        assert_eq!(grounding.grounding_supports.len(), 1);
        assert_eq!(grounding.grounding_supports[0].end_index, 20);
        assert_eq!(grounding.grounding_supports[0].chunk_indices, vec![0]);
    }

    #[test]
    fn test_parse_response_no_candidates() {
        let response = r#"{"candidates": []}"#;
        let err = parse_response(response).unwrap_err();
        assert_eq!(err.error_code, ERROR_PARSE);
    }
}
