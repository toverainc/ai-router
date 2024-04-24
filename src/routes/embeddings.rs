use std::sync::Arc;

use axum::extract::State as AxumState;
use axum::response::{IntoResponse, Response};
use axum::Json;
use openai_dive::v1::resources::embedding::EmbeddingParameters;
use tracing::instrument;

use crate::backend::openai::routes as openai_routes;
use crate::backend::triton::routes as triton_routes;
use crate::config::AiRouterModelType;
use crate::errors::AiRouterError;
use crate::request::AiRouterRequestData;
use crate::state::{BackendTypes, State};

#[instrument(skip(state, request))]
pub async fn embed(
    AxumState(state): AxumState<Arc<State>>,
    mut request: Json<EmbeddingParameters>,
) -> Result<Response, AiRouterError<String>> {
    if let Some(models) = state.config.models.get(&AiRouterModelType::Embeddings) {
        if let Some(model) = models.get(&request.model) {
            let request_data = AiRouterRequestData::build(model, &request.model, &state)?;
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
                    return Ok(
                        openai_routes::embeddings::embed(c.clone(), request, &request_data)
                            .await
                            .into_response(),
                    );
                }
                BackendTypes::Triton(c) => {
                    return Ok(
                        triton_routes::embeddings::embed(c.clone(), request, &request_data)
                            .await
                            .into_response(),
                    );
                }
            }
        }
    }

    Err(AiRouterError::ModelNotFound::<String>(
        request.model.clone(),
    ))
}
