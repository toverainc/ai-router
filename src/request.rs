use std::sync::Arc;

use tokenizers::Tokenizer;
use tracing::instrument;

use crate::{config::AiRouterModel, errors::AiRouterError, state::State, tokenizers::Tokenizers};

#[derive(Debug)]
pub struct AiRouterRequestData {
    pub max_input: Option<usize>,
    pub max_tokens: Option<u32>,
    pub original_model: Option<String>,
    pub prompt_tokens: usize,
    pub tokenizer: Option<Tokenizer>,
}

impl AiRouterRequestData {
    pub const fn new() -> Self {
        Self {
            max_input: None,
            max_tokens: None,
            original_model: None,
            prompt_tokens: 0,
            tokenizer: None,
        }
    }

    /// # Errors
    /// `AiRouterError::InternalServerError` when `max_input` is set for the model but `hf_model_name is not`
    #[instrument(level = "debug", skip(model, model_name, state))]
    pub fn build(
        model: &AiRouterModel,
        model_name: &str,
        state: &Arc<State>,
    ) -> Result<Self, AiRouterError<String>> {
        let mut request_data: Self = Self::new();

        request_data.original_model = Some(String::from(model_name));

        if let Some(max_input) = model.max_input {
            if let Some(hf_model_name) = model.hf_model_name.clone() {
                request_data.max_input = Some(max_input);
                request_data.tokenizer = Tokenizers::get(&state.tokenizers, &hf_model_name);
            } else {
                return Err(AiRouterError::InternalServerError::<String>(String::from(
                    "model parameter max_input requires hf_model_name",
                )));
            }
        }

        if let Some(max_tokens) = model.max_tokens {
            request_data.max_tokens = Some(max_tokens);
        }

        Ok(request_data)
    }
}

/// # Errors
/// `AiRouterError::InputExceededError` when number of tokens in request exceeds `max_input` for the model
/// `AiRouterError::InternalServerError` when `max_input` is set for the model but the tokenizer is not available or failed to encode the request input
pub fn check_input_cc(
    input: &str,
    model: &str,
    request_data: &mut AiRouterRequestData,
) -> Result<(), AiRouterError<String>> {
    let model = request_data
        .original_model
        .clone()
        .unwrap_or_else(|| String::from(model));
    if let Some(max_input) = request_data.max_input {
        if let Some(tokenizer) = &request_data.tokenizer {
            if let Ok(encoded) = tokenizer.encode(input, false) {
                request_data.prompt_tokens = encoded.get_tokens().len();
                if request_data.prompt_tokens > max_input {
                    return Err(AiRouterError::InputExceededError::<String>(
                        model,
                        max_input,
                        request_data.prompt_tokens,
                    ));
                }
                return Ok(());
            }
            return Err(AiRouterError::InternalServerError::<String>(format!(
                "max_input set for model {model} but tokenizer failed to encode the request input"
            )));
        }
        return Err(AiRouterError::InternalServerError::<String>(format!(
            "max_input set for model {model} but tokenizer is not available",
        )));
    }
    Ok(())
}
