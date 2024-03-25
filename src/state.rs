use std::collections::HashMap;

use openai_dive::v1::api::Client as OpenAIClient;
use tonic::transport::Channel;

use crate::{
    backend::triton::grpc_inference_service_client::GrpcInferenceServiceClient,
    config::AiRouterConfigFile, tokenizers::Tokenizers,
};

#[derive(Debug)]
pub enum BackendTypes<O, T> {
    OpenAI(O),
    Triton(T),
}

#[derive(Debug)]
pub struct State {
    pub backends: HashMap<String, BackendTypes<OpenAIClient, GrpcInferenceServiceClient<Channel>>>,
    pub config: AiRouterConfigFile,
    pub tokenizers: Tokenizers,
}
