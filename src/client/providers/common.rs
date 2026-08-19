//! Common utilities shared across LLM providers.

use crate::client::models::{Content, Message};
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique IDs.
static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique ID for tool calls.
/// Uses an atomic counter combined with timestamp for uniqueness across restarts.
pub fn generate_unique_id(prefix: &str) -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let counter = ID_COUNTER.fetch_add(1, Ordering::SeqCst);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{}{}_{}", prefix, timestamp, counter)
}

/// Extract text content from a message, joining multiple text blocks with newlines.
pub fn extract_text_content(msg: &Message) -> String {
    msg.content
        .iter()
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.as_str()),
            Content::Image(_) | Content::ToolUse(_) | Content::ToolResult(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_unique_id() {
        let id1 = generate_unique_id("test_");
        let id2 = generate_unique_id("test_");
        let id3 = generate_unique_id("other_");

        assert!(id1.starts_with("test_"));
        assert!(id2.starts_with("test_"));
        assert!(id3.starts_with("other_"));
        assert_ne!(id1, id2); // Should be unique
    }

    #[test]
    fn test_extract_text_content() {
        let msg = Message::user("Hello world");
        assert_eq!(extract_text_content(&msg), "Hello world");
    }

    #[test]
    fn test_extract_text_content_multiple() {
        let msg = Message::with_content(
            crate::client::models::Role::User,
            vec![
                Content::Text("First".to_string()),
                Content::Text("Second".to_string()),
            ],
        );
        assert_eq!(extract_text_content(&msg), "First\nSecond");
    }
}
