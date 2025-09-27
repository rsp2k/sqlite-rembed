// Proof of concept: Using genai crate instead of custom implementations
// This would replace the entire clients.rs file with a much simpler implementation

use sqlite_loadable::{Error, Result};
use genai::Client;
use std::sync::Arc;

/// A wrapper around genai::Client for SQLite extension use
#[derive(Clone)]
pub struct GenAIClient {
    client: Arc<Client>,
    model: String,
}

impl GenAIClient {
    /// Create a new GenAI client for any supported provider
    /// Provider is automatically detected from the model name format
    pub fn new(model: String) -> Result<Self> {
        // genai automatically detects the provider from the model name
        // e.g., "openai::text-embedding-3-small" or "gemini::text-embedding-004"
        let client = Client::default();

        Ok(Self {
            client: Arc::new(client),
            model,
        })
    }

    /// Generate embeddings for a single text input
    pub async fn infer_single(&self, input: &str) -> Result<Vec<f32>> {
        self.client
            .embed(&self.model, input, None)
            .await
            .map_err(|e| Error::new_message(format!("Embedding generation failed: {}", e)))
            .and_then(|response| {
                response
                    .first_embedding()
                    .ok_or_else(|| Error::new_message("No embedding in response"))
                    .map(|embedding| {
                        // Convert to Vec<f32> - genai uses f64 internally
                        embedding.vec().iter().map(|&v| v as f32).collect()
                    })
            })
    }

    /// Generate embeddings for multiple texts (batch processing)
    pub async fn infer_batch(&self, inputs: Vec<&str>) -> Result<Vec<Vec<f32>>> {
        self.client
            .embed_batch(&self.model, inputs, None)
            .await
            .map_err(|e| Error::new_message(format!("Batch embedding generation failed: {}", e)))
            .map(|response| {
                response
                    .embeddings
                    .into_iter()
                    .map(|embedding| {
                        embedding.vec().iter().map(|&v| v as f32).collect()
                    })
                    .collect()
            })
    }
}

/// Configuration helper for different providers
pub fn configure_genai_client(provider: &str, model: &str, api_key: Option<String>) -> Result<String> {
    // With genai, configuration is done through environment variables:
    // OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.

    // For local providers like Ollama, no API key is needed

    // The model name format determines the provider:
    let full_model_name = match provider {
        "openai" => format!("openai::{}", model),
        "gemini" => format!("gemini::{}", model),
        "cohere" => format!("cohere::{}", model),
        "ollama" => format!("ollama::{}", model),
        // For custom endpoints, genai supports configuration through environment variables
        _ => model.to_string(),
    };

    // If an API key is provided, we could set it as an environment variable
    // (though this is typically done outside the application)
    if let Some(key) = api_key {
        match provider {
            "openai" => std::env::set_var("OPENAI_API_KEY", key),
            "gemini" => std::env::set_var("GEMINI_API_KEY", key),
            "cohere" => std::env::set_var("CO_API_KEY", key),
            _ => {}
        }
    }

    Ok(full_model_name)
}

// Example of how simple the virtual table integration would be:
/*
Usage in SQL:

-- Register a client using genai's model naming convention
INSERT INTO temp.rembed_clients(name, options) VALUES
  ('openai-small', 'openai::text-embedding-3-small'),
  ('gemini-latest', 'gemini::text-embedding-004'),
  ('ollama-local', 'ollama::nomic-embed-text');

-- Use embeddings exactly the same way
SELECT rembed('openai-small', 'Some text to embed');
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_name_formatting() {
        let model_name = configure_genai_client("openai", "text-embedding-3-small", None).unwrap();
        assert_eq!(model_name, "openai::text-embedding-3-small");

        let model_name = configure_genai_client("gemini", "text-embedding-004", None).unwrap();
        assert_eq!(model_name, "gemini::text-embedding-004");
    }
}