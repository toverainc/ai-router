//! <https://platform.openai.com/docs/api-reference/completions/create>
use std::collections::HashMap;
use std::iter::IntoIterator;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Context;
use async_stream::{stream, try_stream};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
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
use crate::backend::triton::utils::get_output_idx;
use crate::backend::triton::ModelInferRequest;
use crate::errors::AiRouterError;
use crate::request::{check_input_cc, AiRouterRequestData};
use crate::templater::{TemplateType, Templater};
use crate::utils::{deserialize_bytes_tensor, string_or_seq_string};

const MAX_TOKENS: u32 = 131_072;
const MODEL_OUTPUT_NAME: &str = "text_output";

#[instrument(skip(client, request, request_data, templater))]
pub async fn compat_completions(
    client: GrpcInferenceServiceClient<Channel>,
    request: Json<CompletionCreateParams>,
    request_data: &mut AiRouterRequestData,
    templater: Templater,
) -> Response {
    tracing::debug!("request: {:?}", request);

    if request.stream {
        completions_stream(client, request, request_data, templater)
            .await
            .into_response()
    } else {
        completions(client, request, request_data, templater)
            .await
            .into_response()
    }
}

#[instrument(skip(client, request, request_data, templater))]
async fn completions_stream(
    mut client: GrpcInferenceServiceClient<Channel>,
    Json(request): Json<CompletionCreateParams>,
    request_data: &mut AiRouterRequestData,
    templater: Templater,
) -> Result<Sse<impl Stream<Item = anyhow::Result<Event>>>, AiRouterError<String>> {
    let id = format!("cmpl-{}", Uuid::new_v4());
    let created = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();

    let request = build_triton_request(request, request_data, templater)?;
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
                content_prev.clone_from(&content);
                let response = Completion {
                    id: id.clone(),
                    object: "text_completion".to_string(),
                    created,
                    model: model_name.clone(),
                    choices: vec![CompletionChoice {
                        text: content_new,
                        index: 0,
                        logprobs: None,
                        finish_reason: None,
                    }],
                    usage: None,
                };
                yield Event::default().json_data(response)?;
            }
        }
        let response = Completion {
            id,
            object: "text_completion".to_string(),
            created,
            model: model_name,
            choices: vec![CompletionChoice {
                text: String::new(),
                index: 0,
                logprobs: None,
                finish_reason: Some(FinishReason::StopSequenceReached),
            }],
            usage: None,
        };
        yield Event::default().json_data(response)?;

        // OpenAI stream response terminated by a data: [DONE] message.
        yield Event::default().data("[DONE]");
    };

    Ok(Sse::new(response_stream).keep_alive(KeepAlive::default()))
}

#[instrument(skip(client, request, request_data, templater), err(Debug))]
async fn completions(
    mut client: GrpcInferenceServiceClient<Channel>,
    Json(request): Json<CompletionCreateParams>,
    request_data: &mut AiRouterRequestData,
    templater: Templater,
) -> Result<Json<Completion>, AiRouterError<String>> {
    let request = build_triton_request(request, request_data, templater)?;
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
            .map(|s| s.trim().replace("</s>", ""))
            .collect();
        contents.push(content);
    }

    let prompt_tokens = request_data.prompt_tokens.try_into().unwrap_or(0);

    Ok(Json(Completion {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion".to_string(),
        created: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        model: model_name,
        choices: vec![CompletionChoice {
            text: contents.into_iter().collect(),
            index: 0,
            logprobs: None,
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
    request: CompletionCreateParams,
    request_data: &mut AiRouterRequestData,
    templater: Templater,
) -> Result<ModelInferRequest, AiRouterError<String>> {
    let input = templater.apply_completions(
        &request.prompt,
        request_data.template.clone(),
        &TemplateType::LegacyCompletion,
    )?;
    check_input_cc(&input, &request.model, request_data)?;

    let mut builder = Builder::new()
        .model_name(request.model)
        .input(
            "text_input",
            [1, 1],
            InferTensorData::Bytes(vec![input.as_bytes().to_vec()]),
        )
        .input(
            "max_tokens",
            [1, 1],
            InferTensorData::Int32(vec![i32::try_from(
                request
                    .max_tokens
                    .unwrap_or(request_data.max_tokens.unwrap_or(MAX_TOKENS)),
            )?]),
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
        .output(MODEL_OUTPUT_NAME);

    if let Some(seed) = request.seed {
        builder = builder.input(
            "random_seed",
            [1, 1],
            InferTensorData::UInt64(vec![seed as u64]),
        );
    }

    Ok(builder.build().context("failed to build triton request")?)
}

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
pub struct CompletionCreateParams {
    /// ID of the model to use.
    pub model: String,
    /// The prompt(s) to generate completions for, encoded as a string, array of strings, array of
    /// tokens, or array of token arrays.
    #[serde(deserialize_with = "string_or_seq_string")]
    prompt: Vec<String>,
    /// Generates best_of completions server-side and returns the "best" (the one with the highest
    /// log probability per token). Results cannot be streamed.
    #[serde(default = "default_best_of")]
    best_of: usize,
    /// Echo back the prompt in addition to the completion
    #[serde(default = "default_echo")]
    echo: bool,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing
    /// frequency in the text so far, decreasing the model's likelihood to repeat the same line
    /// verbatim.
    #[serde(default = "default_frequency_penalty")]
    frequency_penalty: f32,
    /// Modify the likelihood of specified tokens appearing in the completion.
    logit_bias: Option<HashMap<String, f32>>,
    /// Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens.
    logprobs: Option<usize>,
    /// The maximum number of tokens to generate in the completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// How many completions to generate for each prompt.
    #[serde(default = "default_n")]
    n: usize,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they
    /// appear in the text so far, increasing the model's likelihood to talk about new topics.
    #[serde(default = "default_presence_penalty")]
    presence_penalty: f32,
    /// If specified, our system will make a best effort to sample deterministically, such that
    /// repeated requests with the same seed and parameters should return the same result.
    seed: Option<usize>,
    /// Up to 4 sequences where the API will stop generating further tokens. The returned text will
    /// not contain the stop sequence.
    stop: Option<Vec<String>>,
    /// Whether to stream back partial progress.
    #[serde(default = "default_stream")]
    stream: bool,
    /// The suffix that comes after a completion of inserted text.
    suffix: Option<String>,
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
}

#[derive(Serialize, Debug)]
struct Completion {
    /// A unique identifier for the completion.
    id: String,
    /// The object type, which is always "text_completion"
    object: String,
    /// The Unix timestamp (in seconds) of when the completion was created.
    created: u64,
    /// The model used for completion.
    model: String,
    /// The list of completion choices the model generated for the input prompt.
    choices: Vec<CompletionChoice>,
    /// Usage statistics for the completion request.
    usage: Option<Usage>,
}

#[derive(Serialize, Debug)]
struct CompletionChoice {
    text: String,
    index: usize,
    logprobs: Option<()>,
    finish_reason: Option<FinishReason>,
}

fn default_best_of() -> usize {
    1
}

fn default_echo() -> bool {
    false
}

fn default_frequency_penalty() -> f32 {
    0.0
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
