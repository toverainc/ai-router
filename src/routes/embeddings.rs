use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use openai_dive::v1::resources::embedding::EmbeddingParameters;
use tracing::instrument;

use crate::backend::openai::routes as openai_routes;
use crate::backend::triton::routes as triton_routes;
use crate::config::AiRouterModelType;
use crate::errors::AiRouterError;
use crate::startup::{AppState, BackendTypes};

#[instrument(name = "routes::embeddings::embed", skip(state, request))]
pub async fn embed(
    State(state): State<Arc<AppState>>,
    mut request: Json<EmbeddingParameters>,
) -> Response {
    if let Some(model) = state.config.models.get(&AiRouterModelType::Embeddings) {
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

    return AiRouterError::ModelNotFound::<String>(request.model.clone()).into_response();
}
