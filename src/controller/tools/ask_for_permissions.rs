//! AskForPermissions tool implementation
//!
//! This tool allows the LLM to request permission from the user before
//! performing sensitive actions like file writes, deletions, or network operations.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use super::types::{
    DisplayConfig, DisplayResult, Executable, ResultContentType, ToolContext, ToolType,
};
use crate::permissions::{GrantTarget, PermissionLevel, PermissionRegistry, PermissionRequest};

/// AskForPermissions tool name constant.
pub const ASK_FOR_PERMISSIONS_TOOL_NAME: &str = "ask_for_permissions";

/// AskForPermissions tool description constant.
pub const ASK_FOR_PERMISSIONS_TOOL_DESCRIPTION: &str = "Request permission from the user before performing sensitive actions like file writes, \
    deletions, network operations, or system commands. The user can grant permission for \
    this request only (once) or for the remainder of the session.";

/// AskForPermissions tool JSON schema constant.
///
/// Uses the standardizes permission format with explicit target type and level.
pub const ASK_FOR_PERMISSIONS_TOOL_SCHEMA: &str = r#"{
    "type": "object",
    "properties": {
        "target_type": {
            "type": "string",
            "enum": ["path", "domain", "command"],
            "description": "Type of resource: 'path' for files/directories, 'domain' for network endpoints, 'command' for shell commands"
        },
        "target": {
            "type": "string",
            "description": "The resource being accessed: file path, domain pattern, or command pattern"
        },
        "level": {
            "type": "string",
            "enum": ["read", "write", "execute", "admin"],
            "description": "Permission level required: 'read' for viewing, 'write' for modification, 'execute' for running, 'admin' for deletion/full control"
        },
        "recursive": {
            "type": "boolean",
            "description": "For path targets only: whether to include subdirectories",
            "default": false
        },
        "description": {
            "type": "string",
            "description": "Human-readable description of the action requiring permission"
        }
        "reason": {
            "type": "string",
            "description": "Why this action is needed"
        }
    },
    "required": ["target_type", "target", "level", "description"]
}"#;

/// Tool that requests permission from the user for sensitive actions.
pub struct AskForPermissionsTool {
    /// Registry for managing pending permission requests
    registry: Arc<PermissionRegistry>,
}

impl AskForPermissionsTool {
    /// Create a new AskForPermissionsTool instance.
    ///
    /// # Arguments
    /// * `registry` - The permission registry to use for tracking requests.
    pub fn new(registry: Arc<PermissionRegistry>) -> Self {
        Self { registry }
    }
}

impl Executable for AskForPermissionsTool {
    fn name(&self) -> &str {
        ASK_FOR_PERMISSIONS_TOOL_NAME
    }

    fn description(&self) -> &str {
        ASK_FOR_PERMISSIONS_TOOL_DESCRIPTION
    }
}
