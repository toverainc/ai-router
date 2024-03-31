use std::collections::HashMap;

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
    pub models_prompt: HashMap<String, String>,
    pub templater: Templater,
    pub tokenizers: Tokenizers,
}

impl State {
    pub async fn new(config_file: &AiRouterConfigFile) -> Self {
        let backends = Backend::init(config_file).await;
        let mut models_prompt = HashMap::new();
        let templater = Templater::new(&config_file.daemon.template_dir)
            .expect("failed to initialize templater");
        let tokenizers = Tokenizers::new(&config_file.models);

        if let Some(prompt_models) = &config_file.prompt_models {
            for (prompt_format, models) in prompt_models {
                for model in &models.models {
                    models_prompt.insert(String::from(model), String::from(prompt_format));
                }
            }
        }

        Self {
            backends,
            config: config_file.clone(),
            models_prompt,
            templater,
            tokenizers,
        }
    }
}
