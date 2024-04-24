use std::sync::Arc;

use axum::extract::State as AxumState;
use axum::response::Response;
use axum::Json;
use openai_dive::v1::resources::audio::AudioSpeechParameters;

use crate::backend::openai::routes as openai_routes;
use crate::config::AiRouterModelType;
use crate::errors::AiRouterError;
use crate::state::{BackendTypes, State};

pub async fn speech(
    AxumState(state): AxumState<Arc<State>>,
    Json(mut parameters): Json<AudioSpeechParameters>,
) -> Result<Response, AiRouterError<String>> {
    if let Some(models) = state.config.models.get(&AiRouterModelType::AudioSpeech) {
        if let Some(model) = models.get(&parameters.model) {
            if let Some(backend_model) = model.backend_model.clone() {
                parameters.model = backend_model;
            }

            let model_backend = model.backend.as_ref().map_or("default", |m| m);

            let Some(backend) = state.backends.get(model_backend) else {
                return Err(AiRouterError::InternalServerError::<String>(format!(
                    "backend {model_backend} not found"
                )));
            };

            match &backend.client {
                BackendTypes::OpenAI(c) => {
                    return openai_routes::audio::speech(c, parameters).await;
                }
                BackendTypes::Triton(_c) => {
                    return Err(AiRouterError::InternalServerError::<String>(String::from(
                        "create speech to Triton backend not implemented yet",
                    )));
                }
            }
        }
    }

    Err(AiRouterError::ModelNotFound::<String>(
        parameters.model.clone(),
    ))
}
