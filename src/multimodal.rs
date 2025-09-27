// Hybrid multimodal support using the LLaVA → text → embedding approach
// Based on the examples from rsp2k/rust-genai fork

use genai::{Client as GenAiClient, chat::{ChatMessage, ChatRequest, ContentPart}};
use sqlite_loadable::{Error, Result};
use std::sync::Arc;
use tokio::runtime::Runtime;
use once_cell::sync::Lazy;

/// Global tokio runtime for async operations
static RUNTIME: Lazy<Runtime> = Lazy::new(|| {
    Runtime::new().expect("Failed to create tokio runtime")
});

/// Provider capabilities for intelligent routing
#[derive(Debug, Clone)]
pub struct ProviderCapabilities {
    pub supports_image_embeddings: bool,
    pub supports_multimodal_batch: bool,
    pub max_batch_size: usize,
    pub supported_formats: Vec<String>,
}

/// Hybrid multimodal client that combines vision and embedding models
/// with future-ready support for native image embeddings
#[derive(Clone)]
pub struct MultimodalClient {
    client: Arc<GenAiClient>,
    vision_model: String,
    embedding_model: String,
    capabilities: ProviderCapabilities,
}

impl MultimodalClient {
    /// Create a new multimodal client
    pub fn new(vision_model: String, embedding_model: String) -> Result<Self> {
        // Detect provider capabilities
        let capabilities = Self::detect_capabilities(&embedding_model);

        Ok(Self {
            client: Arc::new(GenAiClient::default()),
            vision_model,
            embedding_model,
            capabilities,
        })
    }

    /// Detect provider capabilities for intelligent routing
    fn detect_capabilities(model: &str) -> ProviderCapabilities {
        // Extract provider from model string (e.g., "openai::model" -> "openai")
        let provider = model.split("::").next().unwrap_or("unknown");

        match provider {
            "openai" => ProviderCapabilities {
                supports_image_embeddings: false,  // Coming soon
                supports_multimodal_batch: false,
                max_batch_size: 100,
                supported_formats: vec!["jpeg".to_string(), "png".to_string()],
            },
            "ollama" => ProviderCapabilities {
                supports_image_embeddings: false,  // Under development
                supports_multimodal_batch: false,
                max_batch_size: 50,
                supported_formats: vec!["jpeg".to_string(), "png".to_string()],
            },
            "voyage" => ProviderCapabilities {
                supports_image_embeddings: true,   // Future provider
                supports_multimodal_batch: true,
                max_batch_size: 20,
                supported_formats: vec!["jpeg".to_string(), "png".to_string(), "webp".to_string()],
            },
            "jina" => ProviderCapabilities {
                supports_image_embeddings: true,   // Future capability
                supports_multimodal_batch: true,
                max_batch_size: 16,
                supported_formats: vec!["jpeg".to_string(), "png".to_string()],
            },
            _ => ProviderCapabilities {
                supports_image_embeddings: false,
                supports_multimodal_batch: false,
                max_batch_size: 10,
                supported_formats: vec!["jpeg".to_string()],
            },
        }
    }

    /// Process an image with intelligent routing:
    /// - Uses native image embeddings if provider supports it (future)
    /// - Falls back to hybrid approach (vision → text → embedding) otherwise
    pub fn embed_image_sync(&self, image_data: &[u8]) -> Result<Vec<f32>> {
        // Check if provider supports native image embeddings
        if self.capabilities.supports_image_embeddings {
            // Future: Use native image embedding API when available
            eprintln!("Note: Provider claims image embedding support, but using hybrid approach until native API is available");
        }
        let client = self.client.clone();
        let vision_model = self.vision_model.clone();
        let embedding_model = self.embedding_model.clone();
        let image_base64 = base64::encode(image_data);

        RUNTIME.block_on(async move {
            // Step 1: Describe the image using vision model
            let description = describe_image(&client, &vision_model, &image_base64).await?;

            // Step 2: Embed the description
            client
                .embed(&embedding_model, description, None)
                .await
                .map_err(|e| Error::new_message(format!("Embedding failed: {}", e)))
                .and_then(|response| {
                    response
                        .first_embedding()
                        .ok_or_else(|| Error::new_message("No embedding in response"))
                        .map(|embedding| {
                            embedding.vector().iter().map(|&v| v as f32).collect()
                        })
                })
        })
    }

    /// Process multiple images in batch
    pub fn embed_images_batch_sync(&self, images: Vec<&[u8]>) -> Result<Vec<Vec<f32>>> {
        let client = self.client.clone();
        let vision_model = self.vision_model.clone();
        let embedding_model = self.embedding_model.clone();

        RUNTIME.block_on(async move {
            // Step 1: Describe all images
            let mut descriptions = Vec::new();
            for image_data in images {
                let image_base64 = base64::encode(image_data);
                let description = describe_image(&client, &vision_model, &image_base64).await?;
                descriptions.push(description);
            }

            // Step 2: Batch embed all descriptions
            client
                .embed_batch(&embedding_model, descriptions, None)
                .await
                .map_err(|e| Error::new_message(format!("Batch embedding failed: {}", e)))
                .map(|response| {
                    response
                        .embeddings
                        .into_iter()
                        .map(|embedding| {
                            embedding.vector().iter().map(|&v| v as f32).collect()
                        })
                        .collect()
                })
        })
    }

    /// Process image with custom prompt
    pub fn embed_image_with_prompt_sync(&self, image_data: &[u8], prompt: &str) -> Result<Vec<f32>> {
        let client = self.client.clone();
        let vision_model = self.vision_model.clone();
        let embedding_model = self.embedding_model.clone();
        let image_base64 = base64::encode(image_data);
        let prompt = prompt.to_string();

        RUNTIME.block_on(async move {
            // Step 1: Describe the image with custom prompt
            let description = describe_image_with_prompt(
                &client,
                &vision_model,
                &image_base64,
                &prompt
            ).await?;

            // Step 2: Embed the description
            client
                .embed(&embedding_model, description, None)
                .await
                .map_err(|e| Error::new_message(format!("Embedding failed: {}", e)))
                .and_then(|response| {
                    response
                        .first_embedding()
                        .ok_or_else(|| Error::new_message("No embedding in response"))
                        .map(|embedding| {
                            embedding.vector().iter().map(|&v| v as f32).collect()
                        })
                })
        })
    }
}

/// Describe an image using a vision model
async fn describe_image(
    client: &GenAiClient,
    vision_model: &str,
    image_base64: &str,
) -> Result<String> {
    let chat_req = ChatRequest::new(vec![
        ChatMessage::system(
            "You are a helpful vision AI. Describe images accurately and concisely \
             for embedding purposes. Focus on key visual elements, objects, scene context, \
             colors, and composition."
        ),
        ChatMessage::user(vec![
            ContentPart::from_text("Describe this image in detail for search and embedding purposes:"),
            ContentPart::from_binary_base64("image/jpeg", image_base64, None),
        ])
    ]);

    let chat_response = client
        .exec_chat(vision_model, chat_req, None)
        .await
        .map_err(|e| Error::new_message(format!("Vision analysis failed: {}", e)))?;

    chat_response
        .first_text()
        .ok_or_else(|| Error::new_message("No description generated"))
        .map(|s| s.to_string())
}

/// Describe an image with a custom prompt
async fn describe_image_with_prompt(
    client: &GenAiClient,
    vision_model: &str,
    image_base64: &str,
    prompt: &str,
) -> Result<String> {
    let chat_req = ChatRequest::new(vec![
        ChatMessage::user(vec![
            ContentPart::from_text(prompt),
            ContentPart::from_binary_base64("image/jpeg", image_base64, None),
        ])
    ]);

    let chat_response = client
        .exec_chat(vision_model, chat_req, None)
        .await
        .map_err(|e| Error::new_message(format!("Vision analysis failed: {}", e)))?;

    chat_response
        .first_text()
        .ok_or_else(|| Error::new_message("No description generated"))
        .map(|s| s.to_string())
}

/// Configuration for multimodal client
pub struct MultimodalConfig {
    pub vision_model: String,
    pub embedding_model: String,
}

impl MultimodalConfig {
    /// Create config for Ollama (LLaVA + nomic)
    pub fn ollama() -> Self {
        Self {
            vision_model: "ollama::llava:7b".to_string(),
            embedding_model: "ollama::nomic-embed-text".to_string(),
        }
    }

    /// Create config for OpenAI (GPT-4V + embeddings)
    pub fn openai() -> Self {
        Self {
            vision_model: "openai::gpt-4-vision-preview".to_string(),
            embedding_model: "openai::text-embedding-3-small".to_string(),
        }
    }

    /// Create config for mixed providers
    pub fn mixed(vision: &str, embedding: &str) -> Self {
        Self {
            vision_model: vision.to_string(),
            embedding_model: embedding.to_string(),
        }
    }
}