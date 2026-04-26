//! Llama.cpp (OpenAI-compatible API) client implementation.
//!
//! This module integrates with llama.cpp's local server using the framework's
//! generic `OpenAICompatibleProvider` base. Custom logic is only implemented
//! for endpoints not covered by the base (completions, embeddings).

use crate::builder::LLMBuilder;
use crate::{
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::{EmbeddingBuilder, EmbeddingProvider},
    error::LLMError,
    models::ModelsProvider,
    providers::openai_compatible::{OpenAICompatibleProvider, OpenAIProviderConfig},
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration struct for llama.cpp
pub struct LlamaCppConfig;

// Fix 2 & Default Issue: Required by LLMBuilder::default() which calls L::Config::default()
impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self
    }
}

impl OpenAIProviderConfig for LlamaCppConfig {
    const PROVIDER_NAME: &'static str = "LlamaCpp";
    const DEFAULT_BASE_URL: &'static str = "http://127.0.0.1:8080/v1";
    const DEFAULT_MODEL: &'static str = "qwen2.5:7b";
    const CHAT_ENDPOINT: &'static str = "chat/completions";

    // llama.cpp configuration flags
    const SUPPORTS_REASONING_EFFORT: bool = false;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
    const SUPPORTS_STREAM_OPTIONS: bool = false;
}

/// Type alias mapping the generic OpenAI-compatible provider to llama.cpp
pub type LlamaCpp = OpenAICompatibleProvider<LlamaCppConfig>;

// Fix 2: Satisfy the `HasConfig` trait bound required by `LLMBuilder<L>`
impl crate::HasConfig for LlamaCpp {
    type Config = LlamaCppConfig;
}

// -----------------------------------------------------------------------------
// Custom structs for endpoints not covered by the base provider
// -----------------------------------------------------------------------------

#[derive(Serialize)]
struct LlamaCppCompletionRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
}

#[derive(Deserialize, Debug)]
struct LlamaCppCompletionResponse {
    choices: Vec<LlamaCppCompletionChoice>,
}

#[derive(Deserialize, Debug)]
struct LlamaCppCompletionChoice {
    text: String,
}

#[derive(Serialize)]
struct LlamaCppEmbeddingRequest {
    model: String,
    input: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct LlamaCppEmbeddingResponse {
    data: Vec<LlamaCppEmbeddingData>,
}

#[derive(Deserialize, Debug)]
struct LlamaCppEmbeddingData {
    embedding: Vec<f32>,
}

// -----------------------------------------------------------------------------
// Provider Trait Implementations
// -----------------------------------------------------------------------------

#[async_trait]
impl CompletionProvider for LlamaCpp {
    async fn complete(
        &self,
        req: &CompletionRequest,
        _json_schema: Option<crate::chat::StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        let url = self
            .base_url
            .join("completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let body = LlamaCppCompletionRequest {
            model: &self.model,
            prompt: &req.prompt,
            stream: false,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };

        let mut request = self.client.post(url).json(&body);
        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?.error_for_status()?;
        let json_resp: LlamaCppCompletionResponse = resp.json().await?;

        if let Some(choice) = json_resp.choices.first() {
            Ok(CompletionResponse {
                text: choice.text.clone(),
            })
        } else {
            Err(LLMError::ProviderError(
                "No answer returned by llama.cpp".to_string(),
            ))
        }
    }
}

#[async_trait]
impl EmbeddingProvider for LlamaCpp {
    async fn embed(&self, text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        let url = self
            .base_url
            .join("embeddings")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let body = LlamaCppEmbeddingRequest {
            model: self.model.clone(),
            input: text,
        };

        let resp = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;
        let json_resp: LlamaCppEmbeddingResponse = resp.json().await?;
        Ok(json_resp.data.into_iter().map(|d| d.embedding).collect())
    }
}

#[async_trait]
impl ModelsProvider for LlamaCpp {}

impl crate::LLMProvider for LlamaCpp {}

// -----------------------------------------------------------------------------
// NOTE: ChatProvider & Tool Calling Support
// -----------------------------------------------------------------------------
// OpenAICompatibleProvider already implements ChatProvider natively.
// Type aliases in Rust automatically inherit trait implementations, so LlamaCpp
// now supports `chat()`, `chat_with_tools()`, and all tool calling features
// out of the box without custom implementation.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Builder Implementations
// -----------------------------------------------------------------------------

impl LLMBuilder<LlamaCpp> {
    pub fn build(self) -> Result<Arc<LlamaCpp>, LLMError> {
        let provider = OpenAICompatibleProvider::<LlamaCppConfig>::new(
            self.api_key.unwrap_or_default(),
            self.base_url,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.top_p,
            self.top_k,
            self.tool_choice,
            self.reasoning_effort,
            None, // Fix 3: `voice` is not applicable for local llama.cpp servers. Pass `None`.
            self.extra_body,
            self.enable_parallel_tool_use, // Fix 4: Mapped from `parallel_tool_calls` to match `LLMBuilder` field name
            self.normalize_response,
            self.embedding_encoding_format,
            self.embedding_dimensions,
        );
        Ok(Arc::new(provider))
    }
}

impl EmbeddingBuilder<LlamaCpp> {
    pub fn build(self) -> Result<Arc<LlamaCpp>, LLMError> {
        let model = self.model.ok_or_else(|| {
            LLMError::InvalidRequest("No model provided for llama.cpp embeddings".to_string())
        })?;

        let provider = OpenAICompatibleProvider::<LlamaCppConfig>::new(
            self.api_key.unwrap_or_default(),
            self.base_url,
            Some(model),
            None,
            None,
            self.timeout_seconds,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );
        Ok(Arc::new(provider))
    }
}
