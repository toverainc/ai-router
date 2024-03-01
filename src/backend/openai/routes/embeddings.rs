use axum::Json;
use openai_dive::v1::api::Client;
use openai_dive::v1::resources::embedding::{EmbeddingParameters, EmbeddingResponse};
use tracing::instrument;

use crate::errors::{transform_openai_dive_apierror, AiRouterError};

#[instrument(name = "backend::openai::embeddings::embed", skip(client, request))]
pub async fn embed(
    client: Client,
    Json(request): Json<EmbeddingParameters>,
) -> Result<Json<EmbeddingResponse>, AiRouterError<String>> {
    let response = client
        .embeddings()
        .create(request)
        .await
        .map_err(|e| transform_openai_dive_apierror(&e))?;
    Ok(Json(response))
}
