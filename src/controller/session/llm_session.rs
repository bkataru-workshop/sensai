// This implements a single session with an LLM

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, Ordering};
use std::time::Instant;

use tokio::sync::{Mutex, RwLock, mpsc};
use tokio_util::sync::CancellationToken;

use crate::client::LLMClient;
use crate::client::error::LLMError;
use crate::client::models::Tool as LLMTool;
use crate::client::providers::anthropic::AnthropicProvider;
use crate::client::providers::bedrock::{BedrockCredentials, BedrockProvider};
use crate::client::providers::cohere::CohereProvider;
use crate::client::providers::gemini::GeminiProvider;
use crate::client::providers::openai::OpenAIProvider;

use super::compactor::{AsyncCompactor, Compactor, LLMCompactor, ThresholdCompactor};
use super::config::{CompactorType, LLMProvider, LLMSessionConfig};

/// Creates an LLMClient from the session configuration.
fn create_llm_client(config: &LLMSessionConfig) -> Result<LLMClient, LLMError> {
    match config.provider {
        LLMProvider::Anthropic => {
            let provider = AnthropicProvider::new(config.api_key.clone(), config.model.clone());
            LLMClient::new(Box::new(provider))
        }
        LLMProvider::OpenAI => {
            // Check for Azure configuration first
            let provider = if let (Some(resource), Some(deployment)) =
                (&config.azure_resource, &config.azure_deployment)
            {
                let api_version = config
                    .azure_api_version
                    .clone()
                    .unwrap_or_else(|| "2024-10-21".to_string());
                OpenAIProvider::azure(
                    config.api_key.clone(),
                    resource.clone(),
                    deployment.clone(),
                    api_version,
                )
            } else if let Some(base_url) = &config.base_url {
                OpenAIProvider::with_base_url(
                    config.api_key.clone(),
                    config.model.clone(),
                    base_url.clone(),
                )
            } else {
                OpenAIProvider::new(config.api_key.clone(), config.model.clone())
            };
            LLMClient::new(Box::new(provider))
        }
        LLMProvider::Google => {
            let provider = GeminiProvider::new(config.api_key.clone(), config.model.clone());
            LLMClient::new(Box::new(provider))
        }
        LLMProvider::Cohere => {
            let provider = CohereProvider::new(config.api_key.clone(), config.model.clone());
            LLMClient::new(Box::new(provider))
        }
        LLMProvider::Bedrock => {
            // Bedrock requires all four credential/region fields
            let region = config.bedrock_region.clone().ok_or_else(|| {
                LLMError::new("MISSING_CONFIG", "Bedrock requires bedrock_region")
            })?;
            let access_key_id = config.bedrock_access_key_id.clone().ok_or_else(|| {
                LLMError::new("MISSING_CONFIG", "Bedrock requires bedrock_access_key_id")
            })?;
            let secret_access_key = config.bedrock_secret_access_key.clone().ok_or_else(|| {
                LLMError::new(
                    "MISSING_CONFIG",
                    "Bedrock requires bedrock_secret_access_key",
                )
            })?;

            let credentials = match &config.bedrock_session_token {
                Some(token) => BedrockCredentials::with_session_token(
                    access_key_id,
                    secret_access_key,
                    token.clone(),
                ),
                None => BedrockCredentials::new(access_key_id, secret_access_key),
            };

            let provider = BedrockProvider::new(credentials, region, config.model.clone());
            LLMClient::new(Box::new(provider))
        }
    }
}
