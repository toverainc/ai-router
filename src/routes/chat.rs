use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use openai_dive::v1::resources::chat::ChatCompletionParameters;
use tracing::instrument;

use crate::backend::openai::routes as openai_routes;
use crate::backend::triton::routes as triton_routes;
use crate::config::AiRouterModelType;
use crate::startup::{AppState, BackendTypes};

#[instrument(name = "routes::chat::completion", skip(state, request))]
pub async fn completion(
    State(state): State<AppState>,
    request: Json<ChatCompletionParameters>,
) -> Response {
    if let Some(model) = state.config.models.get(&AiRouterModelType::ChatCompletions) {
        if let Some(model) = model.get(&request.model) {
            match state.backends.get(&model.backend).unwrap() {
                BackendTypes::OpenAI(c) => {
                    return openai_routes::chat::wrap_chat_completion(c.clone(), request).await;
                }
                BackendTypes::Triton(c) => {
                    return triton_routes::chat::compat_chat_completions(c.clone(), request).await;
                }
            }
        }
    }

    "failed to select backend for chat completion request".into_response()
}