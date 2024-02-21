use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use openai_dive::v1::resources::chat::ChatCompletionParameters;
use tracing::instrument;

use crate::backend::openai::routes as openai_routes;
use crate::backend::triton::routes as triton_routes;
use crate::config::AiRouterModelType;
use crate::errors::AiRouterError;
use crate::startup::{AppState, BackendTypes};

#[instrument(name = "routes::chat::completion", skip(state, request))]
pub async fn completion(
    State(state): State<Arc<AppState>>,
    mut request: Json<ChatCompletionParameters>,
) -> Response {
    if let Some(model) = state.config.models.get(&AiRouterModelType::ChatCompletions) {
        if let Some(model) = model.get(&request.model) {
            if let Some(backend_model) = model.backend_model.clone() {
                request.model = backend_model;
            }

            let model_backend = match &model.backend {
                Some(m) => m,
                None => "default",
            };

            match state.backends.get(model_backend).unwrap() {
                BackendTypes::OpenAI(c) => {
                    return openai_routes::chat::wrap_chat_completion(c.clone(), request).await;
                }
                BackendTypes::Triton(c) => {
                    return triton_routes::chat::compat_chat_completions(c.clone(), request).await;
                }
            }
        }
    }

    return AiRouterError::ModelNotFound::<String>(request.model.clone()).into_response();
}
