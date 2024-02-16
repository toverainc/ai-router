use axum::Json;
use openai_dive::v1::api::Client;
use openai_dive::v1::resources::embedding::{EmbeddingParameters, EmbeddingResponse};
use tracing::instrument;

use crate::errors::AppError;

#[instrument(name = "backend::openai::embeddings::embed", skip(client, request))]
pub(crate) async fn embed(
    client: Client,
    Json(request): Json<EmbeddingParameters>,
) -> Result<Json<EmbeddingResponse>, AppError> {
    let response = client.embeddings().create(request).await.unwrap();
    Ok(Json(response))
}
