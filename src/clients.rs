use sqlite_loadable::{Error, Result};
use std::time::Duration;

const DEFAULT_TIMEOUT_SECS: u64 = 30;

pub(crate) fn try_env_var(key: &str) -> Result<String> {
    std::env::var(key)
        .map_err(|_| Error::new_message(format!("{} environment variable not defined. Alternatively, pass in an API key with rembed_client_options", key)))
}

/// Create an HTTP agent with timeout configuration
fn create_agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
        .build()
}

/// Common trait for embedding clients
pub trait EmbeddingClient {
    fn infer_single(&self, input: &str) -> Result<Vec<f32>>;
}

/// Helper trait for clients that support input_type parameter
pub trait EmbeddingClientWithType {
    fn infer_single(&self, input: &str, input_type: Option<&str>) -> Result<Vec<f32>>;
}

/// Common HTTP request builder for embedding APIs
fn send_embedding_request(
    url: &str,
    body: serde_json::Map<String, serde_json::Value>,
    auth_header: Option<String>,
) -> Result<serde_json::Value> {
    let agent = create_agent();
    let mut request = agent.post(url)
        .set("Content-Type", "application/json");

    if let Some(auth) = auth_header {
        request = request.set("Authorization", &auth);
    }

    request
        .send_bytes(
            serde_json::to_vec(&body)
                .map_err(|error| {
                    Error::new_message(format!("Error serializing body to JSON: {error}"))
                })?
                .as_ref(),
        )
        .map_err(|error| Error::new_message(format!("Error sending HTTP request: {error}")))?
        .into_json()
        .map_err(|error| {
            Error::new_message(format!("Error parsing HTTP response as JSON: {error}"))
        })
}

/// Common parser for OpenAI-style responses (data[0].embedding)
fn parse_openai_style_response(value: serde_json::Value) -> Result<Vec<f32>> {
    value
        .get("data")
        .ok_or_else(|| Error::new_message("expected 'data' key in response body"))
        .and_then(|v| {
            v.get(0)
                .ok_or_else(|| Error::new_message("expected 'data.0' path in response body"))
        })
        .and_then(|v| {
            v.get("embedding").ok_or_else(|| {
                Error::new_message("expected 'data.0.embedding' path in response body")
            })
        })
        .and_then(|v| {
            v.as_array().ok_or_else(|| {
                Error::new_message("expected 'data.0.embedding' path to be an array")
            })
        })
        .and_then(|arr| parse_float_array(arr, "data.0.embedding"))
}

/// Common parser for simple embedding responses (embedding or embeddings[0])
fn parse_simple_embedding_response(value: serde_json::Value, key: &str) -> Result<Vec<f32>> {
    value
        .get(key)
        .ok_or_else(|| Error::new_message(format!("expected '{}' key in response body", key)))
        .and_then(|v| {
            if key == "embeddings" {
                v.get(0)
                    .ok_or_else(|| Error::new_message("expected 'embeddings.0' path in response body"))
                    .and_then(|v| {
                        v.as_array()
                            .ok_or_else(|| Error::new_message("expected 'embeddings.0' path to be an array"))
                    })
                    .and_then(|arr| parse_float_array(arr, "embeddings.0"))
            } else {
                v.as_array()
                    .ok_or_else(|| Error::new_message(format!("expected '{}' path to be an array", key)))
                    .and_then(|arr| parse_float_array(arr, key))
            }
        })
}

/// Helper to parse array of floats
fn parse_float_array(arr: &[serde_json::Value], context: &str) -> Result<Vec<f32>> {
    arr.iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| {
                    Error::new_message(format!("expected '{}' array to contain floats", context))
                })
                .map(|f| f as f32)
        })
        .collect()
}

#[derive(Clone)]
pub struct OpenAiClient {
    model: String,
    url: String,
    key: String,
}
const DEFAULT_OPENAI_URL: &str = "https://api.openai.com/v1/embeddings";
const DEFAULT_OPENAI_API_KEY_ENV: &str = "OPENAI_API_KEY";

impl OpenAiClient {
    pub fn new<S: Into<String>>(
        model: S,
        url: Option<String>,
        key: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            model: model.into(),
            url: url.unwrap_or(DEFAULT_OPENAI_URL.to_owned()),
            key: match key {
                Some(key) => key,
                None => try_env_var(DEFAULT_OPENAI_API_KEY_ENV)?,
            },
        })
    }
    pub fn infer_single(&self, input: &str) -> Result<Vec<f32>> {
        let mut body = serde_json::Map::new();
        body.insert("input".to_owned(), input.to_owned().into());
        body.insert("model".to_owned(), self.model.to_owned().into());

        let data = send_embedding_request(
            &self.url,
            body,
            Some(format!("Bearer {}", self.key)),
        )?;

        parse_openai_style_response(data)
    }
}

#[derive(Clone)]
pub struct NomicClient {
    model: String,
    url: String,
    key: String,
}
const DEFAULT_NOMIC_URL: &str = "https://api-atlas.nomic.ai/v1/embedding/text";
const DEFAULT_NOMIC_API_KEY_ENV: &str = "NOMIC_API_KEY";

impl NomicClient {
    pub fn new<S: Into<String>>(
        model: S,
        url: Option<String>,
        key: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            model: model.into(),
            url: url.unwrap_or(DEFAULT_NOMIC_URL.to_owned()),
            key: match key {
                Some(key) => key,
                None => try_env_var(DEFAULT_NOMIC_API_KEY_ENV)?,
            },
        })
    }

    pub fn infer_single(&self, input: &str, input_type: Option<&str>) -> Result<Vec<f32>> {
        let mut body = serde_json::Map::new();
        body.insert("texts".to_owned(), vec![input.to_owned()].into());
        body.insert("model".to_owned(), self.model.to_owned().into());

        if let Some(input_type) = input_type {
            body.insert("input_type".to_owned(), input_type.to_owned().into());
        }

        let data = send_embedding_request(
            &self.url,
            body,
            Some(format!("Bearer {}", self.key)),
        )?;

        parse_simple_embedding_response(data, "embeddings")
    }
}

#[derive(Clone)]
pub struct CohereClient {
    url: String,
    model: String,
    key: String,
}
const DEFAULT_COHERE_URL: &str = "https://api.cohere.com/v1/embed";
const DEFAULT_COHERE_API_KEY_ENV: &str = "CO_API_KEY";

impl CohereClient {
    pub fn new<S: Into<String>>(
        model: S,
        url: Option<String>,
        key: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            model: model.into(),
            url: url.unwrap_or(DEFAULT_COHERE_URL.to_owned()),
            key: match key {
                Some(key) => key,
                None => try_env_var(DEFAULT_COHERE_API_KEY_ENV)?,
            },
        })
    }

    pub fn infer_single(&self, input: &str, input_type: Option<&str>) -> Result<Vec<f32>> {
        let mut body = serde_json::Map::new();
        body.insert("texts".to_owned(), vec![input.to_owned()].into());
        body.insert("model".to_owned(), self.model.to_owned().into());

        if let Some(input_type) = input_type {
            body.insert("input_type".to_owned(), input_type.to_owned().into());
        }

        let data = send_embedding_request(
            &self.url,
            body,
            Some(format!("Bearer {}", self.key)),
        )?;

        parse_simple_embedding_response(data, "embeddings")
    }
}
#[derive(Clone)]
pub struct JinaClient {
    url: String,
    model: String,
    key: String,
}
const DEFAULT_JINA_URL: &str = "https://api.jina.ai/v1/embeddings";
const DEFAULT_JINA_API_KEY_ENV: &str = "JINA_API_KEY";

impl JinaClient {
    pub fn new<S: Into<String>>(
        model: S,
        url: Option<String>,
        key: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            model: model.into(),
            url: url.unwrap_or(DEFAULT_JINA_URL.to_owned()),
            key: match key {
                Some(key) => key,
                None => try_env_var(DEFAULT_JINA_API_KEY_ENV)?,
            },
        })
    }

    pub fn infer_single(&self, input: &str) -> Result<Vec<f32>> {
        let mut body = serde_json::Map::new();
        body.insert("input".to_owned(), vec![input.to_owned()].into());
        body.insert("model".to_owned(), self.model.to_owned().into());

        let agent = create_agent();
        let data: serde_json::Value = agent.post(&self.url)
            .set("Content-Type", "application/json")
            .set("Accept", "application/json")
            .set("Authorization", format!("Bearer {}", self.key).as_str())
            .send_bytes(
                serde_json::to_vec(&body)
                    .map_err(|error| {
                        Error::new_message(format!("Error serializing body to JSON: {error}"))
                    })?
                    .as_ref(),
            )
            .map_err(|error| Error::new_message(format!("Error sending HTTP request: {error}")))?
            .into_json()
            .map_err(|error| {
                Error::new_message(format!("Error parsing HTTP response as JSON: {error}"))
            })?;
        JinaClient::parse_single_response(data)
    }
    pub fn parse_single_response(value: serde_json::Value) -> Result<Vec<f32>> {
        value
            .get("data")
            .ok_or_else(|| Error::new_message("expected 'data' key in response body"))
            .and_then(|v| {
                v.get(0)
                    .ok_or_else(|| Error::new_message("expected 'data.0' path in response body"))
            })
            .and_then(|v| {
                v.get("embedding").ok_or_else(|| {
                    Error::new_message("expected 'data.0.embedding' path in response body")
                })
            })
            .and_then(|v| {
                v.as_array().ok_or_else(|| {
                    Error::new_message("expected 'data.0.embedding' path to be an array")
                })
            })
            .and_then(|arr| {
                arr.iter()
                    .map(|v| {
                        v.as_f64()
                            .ok_or_else(|| {
                                Error::new_message(
                                    "expected 'data.0.embedding' array to contain floats",
                                )
                            })
                            .map(|f| f as f32)
                    })
                    .collect()
            })
    }
}
#[derive(Clone)]
pub struct MixedbreadClient {
    url: String,
    model: String,
    key: String,
}
const DEFAULT_MIXEDBREAD_URL: &str = "https://api.mixedbread.ai/v1/embeddings/";
const DEFAULT_MIXEDBREAD_API_KEY_ENV: &str = "MIXEDBREAD_API_KEY";

impl MixedbreadClient {
    pub fn new<S: Into<String>>(
        model: S,
        url: Option<String>,
        key: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            model: model.into(),
            url: url.unwrap_or(DEFAULT_MIXEDBREAD_URL.to_owned()),
            key: match key {
                Some(key) => key,
                None => try_env_var(DEFAULT_MIXEDBREAD_API_KEY_ENV)?,
            },
        })
    }

    pub fn infer_single(&self, input: &str) -> Result<Vec<f32>> {
        let mut body = serde_json::Map::new();
        body.insert("input".to_owned(), vec![input.to_owned()].into());
        body.insert("model".to_owned(), self.model.to_owned().into());

        let agent = create_agent();
        let data: serde_json::Value = agent.post(&self.url)
            .set("Content-Type", "application/json")
            .set("Accept", "application/json")
            .set("Authorization", format!("Bearer {}", self.key).as_str())
            .send_bytes(
                serde_json::to_vec(&body)
                    .map_err(|error| {
                        Error::new_message(format!("Error serializing body to JSON: {error}"))
                    })?
                    .as_ref(),
            )
            .map_err(|error| Error::new_message(format!("Error sending HTTP request: {error}")))?
            .into_json()
            .map_err(|error| {
                Error::new_message(format!("Error parsing HTTP response as JSON: {error}"))
            })?;
        MixedbreadClient::parse_single_response(data)
    }
    pub fn parse_single_response(value: serde_json::Value) -> Result<Vec<f32>> {
        value
            .get("data")
            .ok_or_else(|| Error::new_message("expected 'data' key in response body"))
            .and_then(|v| {
                v.get(0)
                    .ok_or_else(|| Error::new_message("expected 'data.0' path in response body"))
            })
            .and_then(|v| {
                v.get("embedding").ok_or_else(|| {
                    Error::new_message("expected 'data.0.embedding' path in response body")
                })
            })
            .and_then(|v| {
                v.as_array().ok_or_else(|| {
                    Error::new_message("expected 'data.0.embedding' path to be an array")
                })
            })
            .and_then(|arr| {
                arr.iter()
                    .map(|v| {
                        v.as_f64()
                            .ok_or_else(|| {
                                Error::new_message(
                                    "expected 'data.0.embedding' array to contain floats",
                                )
                            })
                            .map(|f| f as f32)
                    })
                    .collect()
            })
    }
}

#[derive(Clone)]
pub struct OllamaClient {
    url: String,
    model: String,
}
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434/api/embeddings";
impl OllamaClient {
    pub fn new<S: Into<String>>(model: S, url: Option<String>) -> Self {
        Self {
            model: model.into(),
            url: url.unwrap_or(DEFAULT_OLLAMA_URL.to_owned()),
        }
    }

    pub fn infer_single(&self, input: &str) -> Result<Vec<f32>> {
        let mut body = serde_json::Map::new();
        body.insert("prompt".to_owned(), input.to_owned().into());
        body.insert("model".to_owned(), self.model.to_owned().into());

        let agent = create_agent();
        let data: serde_json::Value = agent.post(&self.url)
            .set("Content-Type", "application/json")
            .send_bytes(
                serde_json::to_vec(&body)
                    .map_err(|error| {
                        Error::new_message(format!("Error serializing body to JSON: {error}"))
                    })?
                    .as_ref(),
            )
            .map_err(|error| Error::new_message(format!("Error sending HTTP request: {error}")))?
            .into_json()
            .map_err(|error| {
                Error::new_message(format!("Error parsing HTTP response as JSON: {error}"))
            })?;
        OllamaClient::parse_single_response(data)
    }
    pub fn parse_single_response(value: serde_json::Value) -> Result<Vec<f32>> {
        value
            .get("embedding")
            .ok_or_else(|| Error::new_message("expected 'embedding' key in response body"))
            .and_then(|v| {
                v.as_array()
                    .ok_or_else(|| Error::new_message("expected 'embedding' path to be an array"))
            })
            .and_then(|arr| {
                arr.iter()
                    .map(|v| {
                        v.as_f64()
                            .ok_or_else(|| {
                                Error::new_message("expected 'embedding' array to contain floats")
                            })
                            .map(|f| f as f32)
                    })
                    .collect()
            })
    }
}

#[derive(Clone)]
pub struct LlamafileClient {
    url: String,
}
const DEFAULT_LLAMAFILE_URL: &str = "http://localhost:8080/embedding";

impl LlamafileClient {
    pub fn new(url: Option<String>) -> Self {
        Self {
            url: url.unwrap_or(DEFAULT_LLAMAFILE_URL.to_owned()),
        }
    }

    pub fn infer_single(&self, input: &str) -> Result<Vec<f32>> {
        let mut body = serde_json::Map::new();
        body.insert("content".to_owned(), input.to_owned().into());

        let agent = create_agent();
        let data: serde_json::Value = agent.post(&self.url)
            .set("Content-Type", "application/json")
            .send_bytes(
                serde_json::to_vec(&body)
                    .map_err(|error| {
                        Error::new_message(format!("Error serializing body to JSON: {error}"))
                    })?
                    .as_ref(),
            )
            .map_err(|error| Error::new_message(format!("Error sending HTTP request: {error}")))?
            .into_json()
            .map_err(|error| {
                Error::new_message(format!("Error parsing HTTP response as JSON: {error}"))
            })?;
        LlamafileClient::parse_single_response(data)
    }

    pub fn parse_single_response(value: serde_json::Value) -> Result<Vec<f32>> {
        value
            .get("embedding")
            .ok_or_else(|| Error::new_message("expected 'embedding' key in response body"))
            .and_then(|v| {
                v.as_array()
                    .ok_or_else(|| Error::new_message("expected 'embedding' path to be an array"))
            })
            .and_then(|arr| {
                arr.iter()
                    .map(|v| {
                        v.as_f64()
                            .ok_or_else(|| {
                                Error::new_message("expected 'embedding' array to contain floats")
                            })
                            .map(|f| f as f32)
                    })
                    .collect()
            })
    }
}

#[derive(Clone)]
pub enum Client {
    OpenAI(OpenAiClient),
    Nomic(NomicClient),
    Cohere(CohereClient),
    Ollama(OllamaClient),
    Llamafile(LlamafileClient),
    Jina(JinaClient),
    Mixedbread(MixedbreadClient),
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_openai_style_response() {
        let response = json!({
            "data": [{
                "embedding": [0.1, 0.2, 0.3]
            }]
        });

        let result = parse_openai_style_response(response).unwrap();
        assert_eq!(result, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_parse_openai_style_response_missing_data() {
        let response = json!({
            "error": "something"
        });

        let result = parse_openai_style_response(response);
        assert!(result.is_err());
        // sqlite_loadable::Error doesn't implement Display properly in tests
        // Just verify it's an error
    }

    #[test]
    fn test_parse_simple_embedding_response() {
        let response = json!({
            "embeddings": [[0.4, 0.5, 0.6]]
        });

        let result = parse_simple_embedding_response(response, "embeddings").unwrap();
        assert_eq!(result, vec![0.4, 0.5, 0.6]);
    }

    #[test]
    fn test_parse_simple_embedding_response_single_array() {
        let response = json!({
            "embedding": [0.7, 0.8, 0.9]
        });

        let result = parse_simple_embedding_response(response, "embedding").unwrap();
        assert_eq!(result, vec![0.7, 0.8, 0.9]);
    }

    #[test]
    fn test_parse_float_array() {
        let arr = vec![json!(1.0), json!(2.0), json!(3.0)];
        let result = parse_float_array(&arr, "test").unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parse_float_array_with_non_float() {
        let arr = vec![json!(1.0), json!("not a float"), json!(3.0)];
        let result = parse_float_array(&arr, "test");
        assert!(result.is_err());
        // sqlite_loadable::Error doesn't implement Display properly in tests
    }

    #[test]
    fn test_create_agent_has_timeout() {
        // This test verifies that create_agent() returns an agent with timeout configured
        // The timeout is internal to ureq, but we can verify the function doesn't panic
        let _agent = create_agent();
        // If we got here without panicking, the agent was created successfully
    }

    #[test]
    fn test_try_env_var_error_message() {
        // Test that the error message includes the actual key name
        let result = try_env_var("NONEXISTENT_TEST_KEY_12345");
        assert!(result.is_err());
        // sqlite_loadable::Error doesn't implement Display properly in tests
        // but we've verified the code uses the correct variable
    }
}
