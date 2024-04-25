use std::sync::Arc;

use axum::extract::State as AxumState;
use axum::response::Response;
use axum::Json;
use tracing::instrument;

use crate::backend::triton::routes as triton_routes;
use crate::backend::triton::routes::completions::CompletionCreateParams;
use crate::config::AiRouterModelType;
use crate::errors::AiRouterError;
use crate::request::AiRouterRequestData;
use crate::state::{BackendTypes, State};

#[instrument(skip(state, request))]
pub async fn completion(
    AxumState(state): AxumState<Arc<State>>,
    mut request: Json<CompletionCreateParams>,
) -> Result<Response, AiRouterError<String>> {
    if let Some(models) = state.config.models.get(&AiRouterModelType::ChatCompletions) {
        if let Some(model) = models.get(&request.model) {
            let mut request_data = AiRouterRequestData::build(model, &request.model, &state)?;

            if let Some(backend_model) = model.backend_model.clone() {
                request.model = backend_model;
            }

            let model_backend = model.backend.as_ref().map_or("default", |m| m);

            let Some(backend) = state.backends.get(model_backend) else {
                tracing::warn!("backend: {:#?}", state.backends);
                return Err(AiRouterError::InternalServerError::<String>(format!(
                    "backend {model_backend} not found"
                )));
            };

            match &backend.client {
                BackendTypes::OpenAI(_) => {
                    return Err(AiRouterError::BadRequestError(String::from(
                        "legacy completions to OpenAI backend not implemented yet",
                    )));
                }
                BackendTypes::Triton(c) => {
                    return Ok(triton_routes::completions::compat_completions(
                        c.clone(),
                        request,
                        &mut request_data,
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
