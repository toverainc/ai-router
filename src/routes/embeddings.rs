use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use openai_dive::v1::resources::embedding::EmbeddingParameters;
use tracing::instrument;

use crate::backend::openai::routes as openai_routes;
use crate::backend::triton::routes as triton_routes;
use crate::config::AiRouterModelType;
use crate::startup::{AppState, BackendTypes};

#[instrument(name = "routes::embeddings::embed", skip(state, request))]
pub async fn embed(
    State(state): State<AppState>,
    mut request: Json<EmbeddingParameters>,
) -> Response {
    if let Some(model) = state.config.models.get(&AiRouterModelType::Embeddings) {
        if let Some(model) = model.get(&request.model) {
            if let Some(backend_model) = model.backend_model.clone() {
                request.model = backend_model;
            }
            match state.backends.get(&model.backend).unwrap() {
                BackendTypes::OpenAI(c) => {
                    return openai_routes::embeddings::embed(c.clone(), request)
                        .await
                        .unwrap()
                        .into_response();
                }
                BackendTypes::Triton(c) => {
                    return triton_routes::embeddings::embed(c.clone(), request)
                        .await
                        .unwrap()
                        .into_response();
                }
            }
        }
    }

    "failed to select backend for embeddings request".into_response()
}
