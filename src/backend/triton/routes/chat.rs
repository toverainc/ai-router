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
use crate::backend::triton::utils::get_output_idx;
use crate::backend::triton::ModelInferRequest;
use crate::errors::AiRouterError;
use crate::request::{check_input_cc, AiRouterRequestData};
use crate::utils::deserialize_bytes_tensor;

const MAX_TOKENS: u32 = 131_072;
const MODEL_OUTPUT_NAME: &str = "text_output";

#[instrument(
    name = "backend::triton::chat::compat_chat_completions",
    skip(client, request, request_data)
)]
pub async fn compat_chat_completions(
    client: GrpcInferenceServiceClient<Channel>,
    request: Json<ChatCompletionParameters>,
    request_data: &mut AiRouterRequestData,
) -> Response {
    tracing::debug!("request: {:?}", request);

    if request.stream.unwrap_or(false) {
        chat_completions_stream(client, request, request_data)
            .await
            .into_response()
    } else {
        chat_completions(client, request, request_data)
            .await
            .into_response()
    }
}

#[instrument(
    name = "backend::triton::chat::chat_completions_stream",
    skip(client, request, request_data)
)]
async fn chat_completions_stream(
    mut client: GrpcInferenceServiceClient<Channel>,
    Json(request): Json<ChatCompletionParameters>,
    request_data: &mut AiRouterRequestData,
) -> Result<Sse<impl Stream<Item = anyhow::Result<Event>>>, AiRouterError<String>> {
    let id = format!("cmpl-{}", Uuid::new_v4());
    let created = u32::try_from(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs())?;

    let request = build_triton_request(request, request_data)?;
    let model_name = request_data
        .original_model
        .clone()
        .unwrap_or(request.model_name.clone());

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

            let Some(idx) = get_output_idx(&infer_response.outputs, MODEL_OUTPUT_NAME) else {
                let error = format!("{MODEL_OUTPUT_NAME} not found in Triton response");
                tracing::error!("{error:?}");
                yield Event::default().json_data(json!({
                    "error": error,
                }))?;
                return;
            };

            let raw_content = infer_response.raw_output_contents[idx].clone();
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
    name = "backend::triton::chat::chat_completions",
    skip(client, request, request_data),
    err(Debug)
)]
async fn chat_completions(
    mut client: GrpcInferenceServiceClient<Channel>,
    Json(request): Json<ChatCompletionParameters>,
    request_data: &mut AiRouterRequestData,
) -> Result<Json<ChatCompletionResponse>, AiRouterError<String>> {
    let request = build_triton_request(request, request_data)?;
    let model_name = request_data
        .original_model
        .clone()
        .unwrap_or(request.model_name.clone());
    let request_stream = stream! { yield request };
    let mut stream = client
        .model_stream_infer(tonic::Request::new(request_stream))
        .await
        .context("failed to call triton grpc method model_stream_infer")?
        .into_inner();

    let mut contents: Vec<String> = Vec::new();
    while let Some(response) = stream.message().await? {
        if !response.error_message.is_empty() {
            return Err(AiRouterError::InternalServerError(format!(
                "error message received from triton: {}",
                response.error_message
            )));
        }
        let infer_response = response
            .infer_response
            .context("empty infer response received")?;
        tracing::debug!("triton infer response: {:?}", infer_response);

        let Some(idx) = get_output_idx(&infer_response.outputs, MODEL_OUTPUT_NAME) else {
            return Err(AiRouterError::InternalServerError(format!(
                "{MODEL_OUTPUT_NAME} not found in Triton response"
            )));
        };

        let raw_content = infer_response.raw_output_contents[idx].clone();
        let content = deserialize_bytes_tensor(raw_content)?
            .into_iter()
            .map(|s| s.replace("</s>", ""))
            .collect();
        contents.push(content);
    }

    let prompt_tokens = request_data.prompt_tokens.try_into().unwrap_or(0);

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
        // Not fully supported yet, need Triton to return usage stats
        // but populate prompt_tokens and total_tokens for models configured with max_tokens
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens: Some(0),
            // add completion_tokens once we can get them from Triton
            total_tokens: prompt_tokens,
        }),
    }))
}

fn build_triton_request(
    request: ChatCompletionParameters,
    request_data: &mut AiRouterRequestData,
) -> Result<ModelInferRequest, AiRouterError<String>> {
    let chat_history = build_chat_history(request.messages);
    tracing::debug!("chat history after formatting: {}", chat_history);

    check_input_cc(&chat_history, &request.model, request_data)?;

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
            "max_tokens",
            [1, 1],
            InferTensorData::Int32(vec![i32::try_from(
                request.max_tokens.unwrap_or(MAX_TOKENS),
            )?]),
        )
        .input(
            "stream",
            [1, 1],
            InferTensorData::Bool(vec![request.stream.unwrap_or(false)]),
        )
        .output(MODEL_OUTPUT_NAME);

    if let Some(beam_width) = request.n {
        builder = builder.input(
            "beam_width",
            [1, 1],
            InferTensorData::Int32(vec![i32::try_from(beam_width)?]),
        );
    }

    if let Some(presence_penalty) = request.presence_penalty {
        builder = builder.input(
            "presence_penalty",
            [1, 1],
            InferTensorData::FP32(vec![presence_penalty]),
        );
    }

    if let Some(seed) = request.seed {
        builder = builder.input(
            "random_seed",
            [1, 1],
            InferTensorData::UInt64(vec![u64::from(seed)]),
        );
    }

    if let Some(stop) = request.stop {
        let stop_words = match stop {
            StopToken::Array(a) => string_vec_to_byte_vecs(&a),
            StopToken::String(s) => vec![s.as_bytes().to_vec()],
        };
        builder = builder.input("stop_words", [1, 1], InferTensorData::Bytes(stop_words));
    }

    if let Some(temperature) = request.temperature {
        builder = builder.input(
            "temperature",
            [1, 1],
            InferTensorData::FP32(vec![temperature]),
        );
    }

    if let Some(top_p) = request.top_p {
        builder = builder.input("top_p", [1, 1], InferTensorData::FP32(vec![top_p]));
    }

    Ok(builder.build().context("failed to build triton request")?)
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
                    history.push_str(&format!("System {name}: {content}\n"));
                } else {
                    history.push_str(&format!("System: {content}\n"));
                }
            }
            Role::User => {
                if let Some(name) = message.name {
                    history.push_str(&format!("User {name}: {content}\n"));
                } else {
                    history.push_str(&format!("User: {content}\n"));
                }
            }
            Role::Assistant => {
                history.push_str(&format!("Assistant: {content}\n"));
            }
            Role::Tool => {
                history.push_str(&format!("Tool: {content}\n"));
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
