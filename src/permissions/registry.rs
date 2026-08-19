//! Permission registry for managing grants and permission requests.
//!
//! The registry tracks:
//! - Active grants per session
//! - Pending permission requests (both individual and batched)
//! - Provides methods for checking, granting, and revoking permissions

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::{Mutex, mpsc, oneshot};

/// Maximum number of pending requests before triggering cleanup.
const PENDING_CLEANUP_THRESHOLD: usize = 50;

/// Maximum age for pending requests before they're considered stale (5 minutes).
const PENDING_MAX_AGE: Duration = Duration::from_secs(300);

use super::{
    BatchPermissionRequest, BatchPermissionResponse, Grant, GrantTarget, PermissionRequest,
};
use crate::controller::types::{ControllerEvent, TurnId};
