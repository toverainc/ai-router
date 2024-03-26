use crate::{backend::Backends, config::AiRouterConfigFile, tokenizers::Tokenizers};

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
