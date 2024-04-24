pub mod openai;
pub mod triton;

use std::collections::HashMap;

use openai_dive::v1::api::Client as OpenAIClient;
use tonic::transport::Channel;

use crate::backend::triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::config::{AiRouterBackend, AiRouterBackendType, AiRouterConfigFile};
use crate::state::BackendTypes;

type BackendClient = BackendTypes<OpenAIClient, GrpcInferenceServiceClient<Channel>>;
pub type Backends = HashMap<String, Backend>;

#[derive(Debug)]
pub struct Backend {
    pub client: BackendClient,
}

impl Backend {
    /// # Panics
    /// - when trying to initialize an `OpenAI` backend without API key
    /// - when unable to connect to a Trinton backend
    pub async fn new(name: &String, backend: &AiRouterBackend) -> Self {
        let client: BackendClient = match backend.backend_type {
            AiRouterBackendType::OpenAI => {
                println!("initializing OpenAI backend {name}");
                BackendClient::OpenAI(OpenAIClient {
                    api_key: backend
                        .api_key
                        .as_ref()
                        .unwrap_or_else(|| panic!("OpenAI backend {name} is missing API key"))
                        .clone(),
                    base_url: backend.base_url.clone(),
                    http_client: reqwest::Client::new(),
                    organization: None,
                })
            }
            AiRouterBackendType::Triton => {
                println!("initializing Triton backend {name}");
                BackendClient::Triton(
                    GrpcInferenceServiceClient::connect(backend.base_url.clone())
                        .await
                        .unwrap_or_else(|e| {
                            panic!(
                                "failed to connect to Triton backend {name} ({}): {e:?}",
                                backend.base_url
                            )
                        }),
                )
            }
        };

        Self { client }
    }

    pub async fn init(config: &AiRouterConfigFile) -> HashMap<String, Self> {
        let mut map: Backends = HashMap::new();

        for (name, backend) in &config.backends {
            map.insert(name.clone(), Self::new(name, backend).await);

            if backend.default.unwrap_or(false) {
                map.insert(String::from("default"), Self::new(name, backend).await);
            }
        }

        map
    }
}

#[cfg(test)]
mod tests {
    use crate::config::AiRouterConfigFile;

    use super::Backend;

    #[tokio::test]
    async fn test_default_backend() {
        let config_file =
            AiRouterConfigFile::parse(String::from("tests/ai-router.toml.default_backend"))
                .expect("failed to load test config file");

        assert!(Backend::init(&config_file).await.contains_key("default"));
    }
}
