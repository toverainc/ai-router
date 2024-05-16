use async_stream::try_stream;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use openai_dive::v1::api::Client;
use openai_dive::v1::resources::chat::{ChatCompletionParameters, ChatCompletionResponse};
use tonic::codegen::tokio_stream::{Stream, StreamExt};
use tracing::instrument;

use crate::errors::{transform_openai_dive_apierror, AiRouterError};
use crate::request::AiRouterRequestData;

#[instrument(skip(client, request))]
pub async fn wrap_chat_completion(
    client: Client,
    request: Json<ChatCompletionParameters>,
    request_data: &AiRouterRequestData,
) -> Response {
    if request.stream.unwrap_or(false) {
        chat_completion_stream(client, request, request_data)
            .await
            .into_response()
    } else {
        chat_completion(client, request, request_data)
            .await
            .into_response()
    }
}

#[instrument(skip(client, request))]
async fn chat_completion(
    client: Client,
    Json(request): Json<ChatCompletionParameters>,
    request_data: &AiRouterRequestData,
) -> Result<Json<ChatCompletionResponse>, AiRouterError<String>> {
    let mut response = client
        .chat()
        .create(request)
        .await
        .map_err(|e| transform_openai_dive_apierror(&e))?;
    response.model = request_data
        .original_model
        .clone()
        .unwrap_or(response.model);
    Ok(Json(response))
}

#[instrument(skip(client, request))]
async fn chat_completion_stream(
    client: Client,
    Json(request): Json<ChatCompletionParameters>,
    request_data: &AiRouterRequestData,
) -> Result<Sse<impl Stream<Item = anyhow::Result<Event>>>, AiRouterError<String>> {
    let mut stream = client
        .chat()
        .create_stream(request.clone())
        .await
        .map_err(|e| transform_openai_dive_apierror(&e))?;

    let response_model = request_data.original_model.clone().unwrap_or(request.model);

    let response_stream = try_stream! {
        while let Some(response) = stream.next().await {
            match response {
                Ok(mut response) => {
                    tracing::debug!("{response:?}");
                    response.model.clone_from(&response_model);
                    yield Event::default().json_data(response)?;
                }
                Err(e) => tracing::error!("{e}"),
            }
        }
    };

    Ok(Sse::new(response_stream).keep_alive(KeepAlive::default()))
}
