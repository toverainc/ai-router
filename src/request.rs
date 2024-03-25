use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::{
    config::AiRouterModel, errors::AiRouterError, startup::AppState, tokenizers::Tokenizers,
};

#[derive(Debug)]
pub struct AiRouterRequestData {
    pub max_input: Option<usize>,
    pub original_model: Option<String>,
    pub tokenizer: Option<Tokenizer>,
}

impl AiRouterRequestData {
    pub fn new() -> Self {
        Self {
            max_input: None,
            original_model: None,
            tokenizer: None,
        }
    }

    /// # Errors
    /// `AiRouterError::InternalServerError` when `max_input` is set for the model but `hf_model_name is not`
    pub fn build(
        model: &AiRouterModel,
        model_name: &str,
        state: &Arc<AppState>,
    ) -> Result<Self, AiRouterError<String>> {
        let mut request_data: AiRouterRequestData = Self::new();

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

        Ok(request_data)
    }
}

/// # Errors
/// `AiRouterError::InputExceededError` when number of tokens in request exceeds `max_input` for the model
/// `AiRouterError::InternalServerError` when `max_input` is set for the model but the tokenizer is not available or failed to encode the request input
pub fn check_input_cc(
    input: &str,
    model: &str,
    request_data: &AiRouterRequestData,
) -> Result<(), AiRouterError<String>> {
    let model = request_data
        .original_model
        .clone()
        .unwrap_or(String::from(model));
    if let Some(max_input) = request_data.max_input {
        if let Some(tokenizer) = &request_data.tokenizer {
            if let Ok(encoded) = tokenizer.encode(input, false) {
                let num_tokens = encoded.get_tokens().len();
                if num_tokens > max_input {
                    return Err(AiRouterError::InputExceededError::<String>(
                        model, max_input, num_tokens,
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
