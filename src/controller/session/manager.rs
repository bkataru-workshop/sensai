// This implements a session manager that handles multiple sessions

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{RwLock, mpsc};
use tokio_util::sync::CancellationToken;

use super::LLMSession;
use super::config::LLMSessionConfig;
use crate::client::error::LLMError;
use crate::controller::types::FromLLMPayload;
