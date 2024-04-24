use crate::{
    backend::{Backend, Backends},
    config::AiRouterConfigFile,
    tokenizers::Tokenizers,
};

#[derive(Debug)]
pub enum BackendTypes<O, T> {
    OpenAI(O),
    Triton(T),
}

#[derive(Debug)]
pub struct State {
    pub backends: Backends,
    pub config: AiRouterConfigFile,
    pub tokenizers: Tokenizers,
}

impl State {
    pub async fn new(config_file: &AiRouterConfigFile) -> Self {
        let backends = Backend::init(config_file).await;
        let tokenizers = Tokenizers::new(&config_file.models);

        Self {
            backends,
            config: config_file.clone(),
            tokenizers,
        }
    }
}
