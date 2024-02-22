use std::collections::HashMap;

use tokenizers::tokenizer::Tokenizer;

use crate::config::AiRouterModels;

#[derive(Debug)]
pub struct Tokenizers(HashMap<String, Tokenizer>);

impl Tokenizers {
    pub fn new(models: &AiRouterModels) -> Self {
        let mut tokenizers: HashMap<String, Tokenizer> = HashMap::new();

        for models in models.values() {
            for (model_name, model) in models {
                if let Some(hf_model_name) = &model.hf_model_name {
                    if !tokenizers.contains_key(hf_model_name) {
                        if let Ok(tokenizer) = Tokenizer::from_pretrained(hf_model_name, None) {
                            tokenizers.insert(String::from(hf_model_name), tokenizer);
                        } else {
                            tracing::error!("failed to initialize tokenizer '{hf_model_name}' for model '{model_name}'");
                        }
                    }
                }
            }
        }

        Tokenizers(tokenizers)
    }

    pub fn get(tokenizers: &Self, name: &str) -> Option<Tokenizer> {
        tokenizers.0.get(name).cloned()
    }
}
