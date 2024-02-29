use async_stream::try_stream;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use openai_dive::v1::api::Client;
use openai_dive::v1::resources::chat::{ChatCompletionParameters, ChatCompletionResponse};
use tonic::codegen::tokio_stream::{Stream, StreamExt};
use tracing::instrument;

use crate::errors::{transform_openai_dive_apierror, AiRouterError};

#[instrument(
    name = "backend::openai::routes::chat::wrap_chat_completion",
    skip(client, request)
)]
pub(crate) async fn wrap_chat_completion(
    client: Client,
    request: Json<ChatCompletionParameters>,
) -> Response {
    if request.stream.unwrap_or(false) {
        chat_completion_stream(client, request)
            .await
            .into_response()
    } else {
        chat_completion(client, request).await.into_response()
    }
}

#[instrument(
    name = "backend::openai::routes::chat::chat_completion",
    skip(client, request)
)]
async fn chat_completion(
    client: Client,
    Json(request): Json<ChatCompletionParameters>,
) -> Result<Json<ChatCompletionResponse>, AiRouterError<String>> {
    let response = client
        .chat()
        .create(request)
        .await
        .map_err(|e| transform_openai_dive_apierror(&e))?;
    Ok(Json(response))
}

#[instrument(
    name = "backend::openai::routes::chat::chat_completion_stream",
    skip(client, request)
)]
async fn chat_completion_stream(
    client: Client,
    Json(request): Json<ChatCompletionParameters>,
) -> Result<Sse<impl Stream<Item = anyhow::Result<Event>>>, AiRouterError<String>> {
    let mut stream = client
        .chat()
        .create_stream(request.clone())
        .await
        .map_err(|e| transform_openai_dive_apierror(&e))?;

    let response_stream = try_stream! {
        while let Some(response) = stream.next().await {
            match response {
                Ok(response) => {
                    tracing::debug!("{response:?}");
                    yield Event::default().json_data(response)?;
                }
                Err(e) => tracing::error!("{e}"),
            }
        }
    };

    Ok(Sse::new(response_stream).keep_alive(KeepAlive::default()))
}
