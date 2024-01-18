//! <https://platform.openai.com/docs/api-reference/chat/create>
use std::collections::HashMap;
use std::iter::IntoIterator;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use async_stream::{stream, try_stream};
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use openai_dive::v1::resources::chat::{
    ChatCompletionChoice, ChatCompletionChunkChoice, ChatCompletionChunkResponse, ChatMessage,
    ChatMessageContent, DeltaChatMessage, Role,
};
use openai_dive::v1::resources::shared::{FinishReason, Usage};
use serde::{Deserialize, Serialize};
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
    client: State<GrpcInferenceServiceClient<Channel>>,
    request: Json<ChatCompletionCreateParams>,
) -> Response {
    tracing::info!("request: {:?}", request);

    if request.stream {
        chat_completions_stream(client, request)
            .await
            .into_response()
    } else {
        chat_completions(client, request).await.into_response()
    }
}

#[instrument(name = "streaming chat completions", skip(client, request))]
async fn chat_completions_stream(
    State(mut client): State<GrpcInferenceServiceClient<Channel>>,
    Json(request): Json<ChatCompletionCreateParams>,
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
                })).unwrap();
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
                    object: "text_completion".to_string(),
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
                yield Event::default().json_data(response).unwrap();
            }
        }
        let response = ChatCompletionChunkResponse {
            id,
            object: "text_completion".to_string(),
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
        yield Event::default().json_data(response).unwrap();

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
    State(mut client): State<GrpcInferenceServiceClient<Channel>>,
    Json(request): Json<ChatCompletionCreateParams>,
) -> Result<Json<ChatCompletion>, AppError> {
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

    Ok(Json(ChatCompletion {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
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
            finish_reason: FinishReason::StopSequenceReached,
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

fn build_triton_request(request: ChatCompletionCreateParams) -> anyhow::Result<ModelInferRequest> {
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
            "max_tokens",
            [1, 1],
            InferTensorData::Int32(vec![i32::try_from(request.max_tokens)?]),
        )
        .input(
            "bad_words",
            [1, 1],
            InferTensorData::Bytes(vec!["".as_bytes().to_vec()]),
        )
        .input(
            "stop_words",
            [1, 1],
            InferTensorData::Bytes(
                request
                    .stop
                    .unwrap_or_else(|| vec!["</s>".to_string()])
                    .into_iter()
                    .map(std::string::String::into_bytes)
                    .collect(),
            ),
        )
        .input("top_p", [1, 1], InferTensorData::FP32(vec![request.top_p]))
        .input(
            "temperature",
            [1, 1],
            InferTensorData::FP32(vec![request.temperature]),
        )
        .input(
            "presence_penalty",
            [1, 1],
            InferTensorData::FP32(vec![request.presence_penalty]),
        )
        .input(
            "beam_width",
            [1, 1],
            InferTensorData::Int32(vec![i32::try_from(request.n)?]),
        )
        .input(
            "stream",
            [1, 1],
            InferTensorData::Bool(vec![request.stream]),
        )
        .output("text_output");

    if request.seed.is_some() {
        builder = builder.input(
            "random_seed",
            [1, 1],
            InferTensorData::UInt64(vec![request.seed.unwrap() as u64]),
        );
    }

    builder.build().context("failed to build triton request")
}

fn build_chat_history(messages: Vec<ChatCompletionMessageParams>) -> String {
    let mut history = String::new();
    for message in messages {
        match message {
            ChatCompletionMessageParams::System { content, name } => {
                if let Some(name) = name {
                    history.push_str(&format!("System {}: {}\n", name, content));
                } else {
                    history.push_str(&format!("System: {}\n", content));
                }
            }
            ChatCompletionMessageParams::User { content, name } => {
                if let Some(name) = name {
                    history.push_str(&format!("User {}: {}\n", name, content));
                } else {
                    history.push_str(&format!("User: {}\n", content));
                }
            }
            ChatCompletionMessageParams::Assistant { content, .. } => {
                history.push_str(&format!("Assistant: {}\n", content));
            }
            ChatCompletionMessageParams::Tool { content, .. } => {
                history.push_str(&format!("Tool: {}\n", content));
            }
        }
    }
    history.push_str("ASSISTANT:");
    history
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub(crate) struct ChatCompletionCreateParams {
    /// A list of messages comprising the conversation so far.
    messages: Vec<ChatCompletionMessageParams>,
    /// ID of the model to use.
    model: String,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
    /// frequency in the text so far, decreasing the model's likelihood to repeat the same line
    /// verbatim.
    #[serde(default = "default_frequency_penalty")]
    frequency_penalty: f32,
    /// Modify the likelihood of specified tokens appearing in the completion.
    logit_bias: Option<HashMap<String, f32>>,
    /// The maximum number of tokens to generate in the completion.
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    /// How many completions to generate for each prompt.
    #[serde(default = "default_n")]
    n: usize,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they
    /// appear in the text so far, increasing the model's likelihood to talk about new topics.
    #[serde(default = "default_presence_penalty")]
    presence_penalty: f32,
    /// An object specifying the format that the model must output.
    /// Setting to { "type": "json_object" } enables JSON mode, which guarantees the message the
    /// model generates is valid JSON.
    response_format: Option<ResponseFormat>,
    /// If specified, our system will make a best effort to sample deterministically, such that
    /// repeated requests with the same seed and parameters should return the same result.
    seed: Option<usize>,
    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will
    /// not contain the stop sequence.
    stop: Option<Vec<String>>,
    /// Whether to stream back partial progress.
    #[serde(default = "default_stream")]
    stream: bool,
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the
    /// output more random, while lower values like 0.2 will make it more focused and deterministic.
    #[serde(default = "default_temperature")]
    temperature: f32,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model
    /// considers the results of the tokens with top_p probability mass. So 0.1 means only the
    /// tokens comprising the top 10% probability mass are considered.
    #[serde(default = "default_top_p")]
    top_p: f32,
    /// A unique identifier representing your end-user, which can help OpenAI to monitor and detect
    /// abuse.
    user: Option<String>,
    // Not supported yet:
    // tools
    // tool_choices
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
#[serde(tag = "role", rename_all = "lowercase")]
enum ChatCompletionMessageParams {
    System {
        content: String,
        name: Option<String>,
    },
    User {
        content: String,
        name: Option<String>,
    },
    Assistant {
        content: String,
        // Not supported yet:
        // tool_calls
    },
    Tool {
        content: String,
        tool_call_id: String,
    },
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ResponseFormat {
    Text,
    JsonObject,
}

#[derive(Serialize, Debug)]
struct ChatCompletion {
    /// A unique identifier for the completion.
    id: String,
    /// The object type, which is always "chat.completion"
    object: String,
    /// The Unix timestamp (in seconds) of when the completion was created.
    created: u64,
    /// The model used for completion.
    model: String,
    /// This fingerprint represents the backend configuration that the model runs with.
    system_fingerprint: Option<String>,
    /// The list of completion choices the model generated for the input prompt.
    choices: Vec<ChatCompletionChoice>,
    /// Usage statistics for the completion request.
    usage: Option<Usage>,
}

fn default_frequency_penalty() -> f32 {
    0.0
}

fn default_max_tokens() -> usize {
    16
}

fn default_n() -> usize {
    1
}

fn default_presence_penalty() -> f32 {
    0.0
}

fn default_stream() -> bool {
    false
}

fn default_temperature() -> f32 {
    1.0
}

fn default_top_p() -> f32 {
    1.0
}
