use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use crate::client::models::Tool as LLMTool;
use crate::controller::types::TurnId;
use crate::permissions::PermissionRequest;

/// Tool type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolType {
    ApiCall,
    BashCmd,
    WebSearch,
    TextEdit,
    Custom,
    FileRead,
    UserInteraction,
}

/// Context provided to tools during execution.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Session that requested the tool execution.
    pub session_id: i64,
    /// Unique ID for this tool use request.
    pub tool_use_id: String,
    /// Assistant turn ID.
    pub turn_id: Option<TurnId>,
    /// Whether permissions were pre-approved by the batch executor.
    /// When true, tools should skip their internal permission checks.
    #[doc(hidden)]
    pub permissions_pre_approved: bool,
}

impl ToolContext {
    /// Create a new ToolContext.
    pub fn new(session_id: i64, tool_use_id: impl Into<String>, turn_id: Option<TurnId>) -> Self {
        Self {
            session_id,
            tool_use_id: tool_use_id.into(),
            turn_id,
            permissions_pre_approved: false,
        }
    }

    /// Create a new ToolContext with pre-approved permissions.
    pub fn with_pre_approved_permissions(
        session_id: i64,
        tool_use_id: impl Into<String>,
        turn_id: Option<TurnId>,
    ) -> Self {
        Self {
            session_id,
            tool_use_id: tool_use_id.into(),
            turn_id,
            permissions_pre_approved: true,
        }
    }
}

/// Result status from tool execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolResultStatus {
    Success,
    Error,
    Timeout,
}

/// Content type for formatting tool results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResultContentType {
    Json,
    Markdown,
    Yaml,
    #[default]
    PlainText,
    Xml,
    Auto,
}

/// Formatted display result from a tool.
#[derive(Debug, Clone)]
pub struct DisplayResult {
    /// The formatted content to display.
    pub content: String,
    /// How to render the content.
    pub content_type: ResultContentType,
    /// True if content was truncated.
    pub is_truncated: bool,
    /// Original line count for "50 of 200 lines" display.
    pub full_length: usize,
}

/// Closure type for generating a dynamic display title from tool input.
pub type DisplayTitleFn = Box<dyn Fn(&HashMap<String, serde_json::Value>) -> String + Send + Sync>;

/// Closure type for generating display content from tool input and result.
pub type DisplayContentFn = Box<dyn Fn(&HashMap<String, serde_json::Value>, &str) -> DisplayResult + Send + Sync>;

/// Configuration for how a tool should be displayed in the UI.
pub struct DisplayConfig {
    /// UI-friendly name (e.g., "Web Search" vs "web_search").
    pub display_name: String,
    /// Dynamic title based on input (e.g., "AWS/DynamoDB" for tool input).
    pub display_title: DisplayTitleFn,
    /// Dynamic content based on input and result.
    pub display_content: DisplayContentFn,
}

impl DisplayConfig {
    /// Create a default display config for a tool.
    pub fn default_for(name: &str) -> Self {
        let name_owned = name.to_owned();


    }
}
