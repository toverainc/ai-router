pub mod openai;
pub mod triton;

use std::collections::HashMap;

use openai_dive::v1::api::Client as OpenAIClient;
use tonic::transport::Channel;

use crate::backend::triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::config::{AiRouterBackendType, AiRouterConfigFile};
use crate::startup::BackendTypes;

pub(crate) async fn init_backends(
    config: &AiRouterConfigFile,
) -> HashMap<String, BackendTypes<OpenAIClient, GrpcInferenceServiceClient<Channel>>> {
    let mut map = HashMap::new();

    for (name, backend) in &config.backends {
        match backend.backend_type {
            AiRouterBackendType::OpenAI => {
                println!("initializing OpenAI backend {name}");
                let backend_client = OpenAIClient {
                    api_key: backend
                        .api_key
                        .as_ref()
                        .unwrap_or_else(|| panic!("OpenAI backend {name} is missing API key"))
                        .clone(),
                    base_url: backend.base_url.clone(),
                    http_client: reqwest::Client::new(),
                    organization: None,
                };
                map.insert(name.clone(), BackendTypes::OpenAI(backend_client.clone()));

                if backend.default.unwrap_or(false) {
                    map.insert(
                        String::from("default"),
                        BackendTypes::OpenAI(backend_client.clone()),
                    );
                }
            }
            AiRouterBackendType::Triton => {
                println!("initializing Triton backend {name}");
                let backend_client = GrpcInferenceServiceClient::connect(backend.base_url.clone())
                    .await
                    .unwrap_or_else(|e| {
                        panic!(
                            "failed to connect to Triton backend {name} ({}): {e:?}",
                            backend.base_url
                        )
                    });
                map.insert(name.clone(), BackendTypes::Triton(backend_client.clone()));

                if backend.default.unwrap_or(false) {
                    map.insert(
                        String::from("default"),
                        BackendTypes::Triton(backend_client.clone()),
                    );
                }
            }
        }
    }

    map
}
