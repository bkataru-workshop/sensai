//! LLM client with provider-agnostic interface.

/// Error types for LLM operations.
pub mod error;
/// HTTP client with TLS and retry logic.
pub mod http;
/// Message and request/response models.
pub mod models;
/// LLM provider implementations (Anthropic, OpenAI).
pub mod providers;
/// Provider trait definition.
pub mod traits;
