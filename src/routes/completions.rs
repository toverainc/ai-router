use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use tracing::instrument;

use crate::backend::triton::routes as triton_routes;
use crate::backend::triton::routes::completions::CompletionCreateParams;
use crate::config::AiRouterModelType;
use crate::startup::{AppState, BackendTypes};

#[instrument(name = "routes::completion::completions", skip(state, request))]
pub async fn completion(
    State(state): State<AppState>,
    request: Json<CompletionCreateParams>,
) -> Response {
    if let Some(model) = state.config.models.get(&AiRouterModelType::ChatCompletions) {
        if let Some(model) = model.get(&request.model) {
            match state.backends.get(&model.backend).unwrap() {
                BackendTypes::OpenAI(_) => {
                    return "legacy completions not supported in OpenAI backend".into_response();
                }
                BackendTypes::Triton(c) => {
                    return triton_routes::completions::compat_completions(c.clone(), request)
                        .await;
                }
            }
        }
    }

    "failed to select backend for legacy completion request".into_response()
}
