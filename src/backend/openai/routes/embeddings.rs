use axum::Json;
use openai_dive::v1::api::Client;
use openai_dive::v1::resources::embedding::{EmbeddingParameters, EmbeddingResponse};
use tracing::instrument;

use crate::errors::{transform_openai_dive_apierror, AiRouterError};
use crate::request::AiRouterRequestData;

#[instrument(skip(client, request))]
pub async fn embed(
    client: Client,
    Json(request): Json<EmbeddingParameters>,
    request_data: &AiRouterRequestData,
) -> Result<Json<EmbeddingResponse>, AiRouterError<String>> {
    let response_model = request_data
        .original_model
        .clone()
        .unwrap_or(request.model.clone());

    let mut response = client
        .embeddings()
        .create(request)
        .await
        .map_err(|e| transform_openai_dive_apierror(&e))?;

    response.model = response_model;

    Ok(Json(response))
}
