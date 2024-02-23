//! <https://platform.openai.com/docs/api-reference/chat/create>
use std::iter::IntoIterator;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use async_stream::{stream, try_stream};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use openai_dive::v1::resources::chat::{
    ChatCompletionChoice, ChatCompletionChunkChoice, ChatCompletionChunkResponse,
    ChatCompletionParameters, ChatCompletionResponse, ChatMessage, ChatMessageContent,
    DeltaChatMessage, Role,
};
use openai_dive::v1::resources::shared::{FinishReason, StopToken, Usage};
use serde_json::json;
use tonic::codegen::tokio_stream::Stream;
use tonic::transport::Channel;
use tracing;
use tracing::instrument;
use uuid::Uuid;

use crate::backend::triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::backend::triton::request::{Builder, InferTensorData};
use crate::backend::triton::ModelInferRequest;
use crate::errors::AppError;
use crate::utils::deserialize_bytes_tensor;

#[instrument(name = "chat_completions", skip(client, request))]
pub(crate) async fn compat_chat_completions(
    client: GrpcInferenceServiceClient<Channel>,
    request: Json<ChatCompletionParameters>,
) -> Response {
    tracing::info!("request: {:?}", request);

    if request.stream.unwrap_or(false) {
        chat_completions_stream(client, request)
            .await
            .into_response()
    } else {
        chat_completions(client, request).await.into_response()
    }
}

#[instrument(name = "streaming chat completions", skip(client, request))]
async fn chat_completions_stream(
    mut client: GrpcInferenceServiceClient<Channel>,
    Json(request): Json<ChatCompletionParameters>,
) -> Result<Sse<impl Stream<Item = anyhow::Result<Event>>>, AppError> {
    let id = format!("cmpl-{}", Uuid::new_v4());
    let created = u32::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())?;

    let model_name = request.model.clone();
    let request = build_triton_request(request)?;

    let response_stream = try_stream! {
        let request_stream = stream! { yield request };

        let mut stream = client
            .model_stream_infer(tonic::Request::new(request_stream))
            .await
            .context("failed to call triton grpc method model_stream_infer")?
            .into_inner();

        let mut content_prev = String::new();

        while let Some(response) = stream.message().await? {
            if !response.error_message.is_empty() {
                tracing::error!("received error message from triton: {}", response.error_message);

                // Corresponds to https://github.com/openai/openai-python/blob/17ac6779958b2b74999c634c4ea4c7b74906027a/src/openai/_streaming.py#L113
                yield Event::default().event("error").json_data(json!({
                    "error": {
                        "status_code": 500,
                        "message": "Internal Server Error"
                    }
                }))?;
                return;
            }
            let infer_response = response
                .infer_response
                .context("empty infer response received")?;
            tracing::debug!("triton infer response: {:?}", infer_response);

            let raw_content = infer_response.raw_output_contents[0].clone();
            let content = deserialize_bytes_tensor(raw_content)?
                .into_iter()
                .map(|s| s.replace("</s>", ""))
                .collect::<String>();

            if !content.is_empty() {
                let content_new = content.replace(&content_prev, "");
                if content_new.is_empty() {
                    continue;
                }
                content_prev = content.clone();
                let response = ChatCompletionChunkResponse {
                    id: id.clone(),
                    object: String::from("chat.completion.chunk"),
                    created,
                    model: model_name.clone(),
                    system_fingerprint: None,
                    choices: vec![ChatCompletionChunkChoice {
                        index: Some(0),
                        delta: DeltaChatMessage {
                            role: Some(Role::Assistant),
                            content: Some(content_new),
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                };
                yield Event::default().json_data(response)?;
            }
        }
        let response = ChatCompletionChunkResponse {
            id,
            object: String::from("chat.completion.chunk"),
            created,
            model: model_name,
            system_fingerprint: None,
            choices: vec![ChatCompletionChunkChoice {
                index: Some(0),
                delta: DeltaChatMessage {
                    role: None,
                    content: None,
                    tool_calls: None,
                },
                finish_reason: Some(FinishReason::StopSequenceReached),
            }],
        };
        yield Event::default().json_data(response)?;

        // OpenAI stream response terminated by a data: [DONE] message.
        yield Event::default().data("[DONE]");
    };

    Ok(Sse::new(response_stream).keep_alive(KeepAlive::default()))
}

#[instrument(
    name = "non-streaming chat completions",
    skip(client, request),
    err(Debug)
)]
async fn chat_completions(
    mut client: GrpcInferenceServiceClient<Channel>,
    Json(request): Json<ChatCompletionParameters>,
) -> Result<Json<ChatCompletionResponse>, AppError> {
    let model_name = request.model.clone();
    let request = build_triton_request(request)?;
    let request_stream = stream! { yield request };
    let mut stream = client
        .model_stream_infer(tonic::Request::new(request_stream))
        .await
        .context("failed to call triton grpc method model_stream_infer")?
        .into_inner();

    let mut contents: Vec<String> = Vec::new();
    while let Some(response) = stream.message().await? {
        if !response.error_message.is_empty() {
            return Err(anyhow::anyhow!(
                "error message received from triton: {}",
                response.error_message
            )
            .into());
        }
        let infer_response = response
            .infer_response
            .context("empty infer response received")?;
        tracing::debug!("triton infer response: {:?}", infer_response);

        let raw_content = infer_response.raw_output_contents[0].clone();
        let content = deserialize_bytes_tensor(raw_content)?
            .into_iter()
            .map(|s| s.replace("</s>", ""))
            .collect();
        contents.push(content);
    }

    Ok(Json(ChatCompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: String::from("chat.completion"),
        created: u32::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())?,
        model: model_name,
        system_fingerprint: None,
        choices: vec![ChatCompletionChoice {
            index: Some(0),
            message: ChatMessage {
                name: None,
                role: Role::Assistant,
                tool_calls: None,
                tool_call_id: None,
                content: ChatMessageContent::Text(contents.into_iter().collect()),
            },
            finish_reason: Some(FinishReason::StopSequenceReached),
        }],
        // Not supported yet, need triton to return usage stats
        // but add a fake one to make LangChain happy
        usage: Some(Usage {
            prompt_tokens: 0,
            completion_tokens: Some(0),
            total_tokens: 0,
        }),
    }))
}

fn build_triton_request(request: ChatCompletionParameters) -> anyhow::Result<ModelInferRequest> {
    let chat_history = build_chat_history(request.messages);
    tracing::debug!("chat history after formatting: {}", chat_history);

    let mut builder = Builder::new()
        .model_name(request.model)
        .input(
            "text_input",
            [1, 1],
            InferTensorData::Bytes(vec![chat_history.as_bytes().to_vec()]),
        )
        .input(
            "bad_words",
            [1, 1],
            InferTensorData::Bytes(vec!["".as_bytes().to_vec()]),
        )
        .input(
            "stream",
            [1, 1],
            InferTensorData::Bool(vec![request.stream.unwrap_or(false)]),
        )
        .output("text_output");

    if request.n.is_some() {
        builder = builder.input(
            "beam_width",
            [1, 1],
            InferTensorData::Int32(vec![i32::try_from(request.n.unwrap())?]),
        );
    }

    if request.max_tokens.is_some() {
        builder = builder.input(
            "max_tokens",
            [1, 1],
            InferTensorData::Int32(vec![i32::try_from(request.max_tokens.unwrap())?]),
        );
    }

    if request.presence_penalty.is_some() {
        builder = builder.input(
            "presence_penalty",
            [1, 1],
            InferTensorData::FP32(vec![request.presence_penalty.unwrap()]),
        );
    }

    if request.seed.is_some() {
        builder = builder.input(
            "random_seed",
            [1, 1],
            InferTensorData::UInt64(vec![u64::from(request.seed.unwrap())]),
        );
    }

    if request.stop.is_some() {
        let stop_words = match request.stop.unwrap() {
            StopToken::Array(a) => string_vec_to_byte_vecs(&a),
            StopToken::String(s) => vec![s.as_bytes().to_vec()],
        };
        builder = builder.input("stop_words", [1, 1], InferTensorData::Bytes(stop_words));
    }

    if request.temperature.is_some() {
        builder = builder.input(
            "temperature",
            [1, 1],
            InferTensorData::FP32(vec![request.temperature.unwrap()]),
        );
    }

    if request.top_p.is_some() {
        builder = builder.input(
            "top_p",
            [1, 1],
            InferTensorData::FP32(vec![request.top_p.unwrap()]),
        );
    }

    builder.build().context("failed to build triton request")
}

fn build_chat_history(messages: Vec<ChatMessage>) -> String {
    let mut history = String::new();
    for message in messages {
        let ChatMessageContent::Text(content) = message.content else {
            continue;
        };
        match message.role {
            Role::System => {
                if let Some(name) = message.name {
                    history.push_str(&format!("System {}: {}\n", name, content));
                } else {
                    history.push_str(&format!("System: {}\n", content));
                }
            }
            Role::User => {
                if let Some(name) = message.name {
                    history.push_str(&format!("User {}: {}\n", name, content));
                } else {
                    history.push_str(&format!("User: {}\n", content));
                }
            }
            Role::Assistant => {
                history.push_str(&format!("Assistant: {}\n", content));
            }
            Role::Tool => {
                history.push_str(&format!("Tool: {}\n", content));
            }
            Role::Function => {}
        }
    }
    history.push_str("ASSISTANT:");
    history
}

fn string_vec_to_byte_vecs(strings: &Vec<String>) -> Vec<Vec<u8>> {
    let mut byte_vecs: Vec<Vec<u8>> = Vec::new();

    for string in strings {
        let bytes: Vec<u8> = string.as_bytes().to_vec();
        byte_vecs.push(bytes);
    }

    byte_vecs
}
