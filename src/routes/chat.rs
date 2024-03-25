use std::sync::Arc;

use axum::extract::State as AxumState;
use axum::response::Response;
use axum::Json;
use openai_dive::v1::resources::chat::ChatCompletionParameters;
use tracing::instrument;

use crate::backend::openai::routes as openai_routes;
use crate::backend::triton::routes as triton_routes;
use crate::config::AiRouterModelType;
use crate::errors::AiRouterError;
use crate::request::AiRouterRequestData;
use crate::state::{BackendTypes, State};

#[instrument(skip(state, request))]
pub async fn completion(
    AxumState(state): AxumState<Arc<State>>,
    mut request: Json<ChatCompletionParameters>,
) -> Result<Response, AiRouterError<String>> {
    if let Some(models) = state.config.models.get(&AiRouterModelType::ChatCompletions) {
        if let Some(model) = models.get(&request.model) {
            let mut request_data = AiRouterRequestData::build(model, &request.model, &state)?;

            if let Some(backend_model) = model.backend_model.clone() {
                request.model = backend_model;
            }

            let model_backend = model.backend.as_ref().map_or("default", |m| m);

            let Some(backend) = state.backends.get(model_backend) else {
                return Err(AiRouterError::InternalServerError::<String>(format!(
                    "backend {model_backend} not found"
                )));
            };

            match &backend.client {
                BackendTypes::OpenAI(c) => {
                    return Ok(openai_routes::chat::wrap_chat_completion(
                        c.clone(),
                        request,
                        &request_data,
                    )
                    .await);
                }
                BackendTypes::Triton(c) => {
                    return Ok(triton_routes::chat::compat_chat_completions(
                        c.clone(),
                        request,
                        &mut request_data,
                        state.templater.clone(),
                    )
                    .await);
                }
            }
        }
    }

    Err(AiRouterError::ModelNotFound::<String>(
        request.model.clone(),
    ))
}
