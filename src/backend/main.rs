use std::collections::HashMap;

use openai_dive::v1::api::Client as OpenAIClient;
use tonic::transport::Channel;

use crate::backend::triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::config::{AiRouterBackendType, AiRouterConfigFile};
use crate::startup::BackendTypes;

pub async fn init_backends(
    config: &AiRouterConfigFile,
) -> HashMap<String, BackendTypes<OpenAIClient, GrpcInferenceServiceClient<Channel>>> {
    let mut map = HashMap::new();

    for (name, backend) in &config.backends {
        match backend.backend_type {
            AiRouterBackendType::OpenAI => {
                println!("initializing OpenAI backend {name}");
                let backend_client = OpenAIClient {
                    api_key: backend.api_key.as_ref().unwrap().clone(),
                    base_url: backend.base_url.clone(),
                    http_client: reqwest::Client::new(),
                    organization: None,
                };
                map.insert(name.clone(), BackendTypes::OpenAI(backend_client));
            }
            AiRouterBackendType::Triton => {
                println!("initializing Triton backend {name}");
                let backend_client = GrpcInferenceServiceClient::connect(backend.base_url.clone())
                    .await
                    .unwrap();
                map.insert(name.clone(), BackendTypes::Triton(backend_client));
            }
        }
    }

    map
}
