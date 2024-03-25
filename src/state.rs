use crate::{
    backend::{Backend, Backends},
    config::AiRouterConfigFile,
    templater::Templater,
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
    pub templater: Templater,
    pub tokenizers: Tokenizers,
}

impl State {
    pub async fn new(config_file: &AiRouterConfigFile) -> Self {
        let backends = Backend::init(config_file).await;
        let templater = Templater::new(&config_file.daemon.template_dir)
            .expect("failed to initialize templater");
        let tokenizers = Tokenizers::new(&config_file.models);

        Self {
            backends,
            config: config_file.clone(),
            templater,
            tokenizers,
        }
    }
}
