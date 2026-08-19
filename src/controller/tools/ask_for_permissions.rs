//! AskForPermissions tool implementation
//!
//! This tool allows the LLM to request permission from the user before
//! performing sensitive actions like file writes, deletions, or network operations.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
