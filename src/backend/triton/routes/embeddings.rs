use anyhow::Context;
use async_stream::stream;
use axum::Json;
use openai_dive::v1::resources::embedding::{
    Embedding, EmbeddingInput, EmbeddingOutput, EmbeddingParameters, EmbeddingResponse,
};
use openai_dive::v1::resources::shared::Usage;
use tonic::transport::Channel;
use tracing;
use tracing::instrument;

use crate::backend::triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::backend::triton::request::{Builder, InferTensorData};
use crate::backend::triton::utils::get_output_idx;
use crate::backend::triton::ModelInferRequest;
use crate::errors::AiRouterError;
use crate::request::AiRouterRequestData;

const MODEL_OUTPUT_NAME: &str = "embedding";

#[instrument(skip(client, request, request_data))]
pub(crate) async fn embed(
    mut client: GrpcInferenceServiceClient<Channel>,
    Json(request): Json<EmbeddingParameters>,
    request_data: &AiRouterRequestData,
) -> Result<Json<EmbeddingResponse>, AiRouterError<String>> {
    tracing::debug!("triton embeddings request: {:?}", request);

    let batch_size: usize = match request.input {
        EmbeddingInput::StringArray(ref sa) => sa.len(),
        EmbeddingInput::IntegerArrayArray(ref iaa) => iaa.len(),
        EmbeddingInput::IntegerArray(_) | EmbeddingInput::String(_) => 1,
    };
    let mut dimensions: usize = 0;

    let request = build_triton_request(request)?;
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

    let mut data: Vec<u8> = Vec::new();
    while let Some(response) = stream.message().await? {
        if !response.error_message.is_empty() {
            return Err(AiRouterError::InternalServerError(format!(
                "error message received from triton: {}",
                response.error_message
            )));
        }
        let mut infer_response = response
            .infer_response
            .context("empty infer response received")?;

        let Some(idx) = get_output_idx(&infer_response.outputs, MODEL_OUTPUT_NAME) else {
            return Err(AiRouterError::InternalServerError(format!(
                "{MODEL_OUTPUT_NAME} not found in Triton response"
            )));
        };

        if usize::try_from(infer_response.outputs[idx].shape[0])? != batch_size {
            return Err(AiRouterError::InternalServerError(String::from(
                "batch sizes of request and response differ",
            )));
        }

        dimensions = usize::try_from(infer_response.outputs[idx].shape[1])?;
        data.append(&mut infer_response.raw_output_contents[idx]);
    }

    tracing::debug!("{data:?}");

    let data = transform_triton_f32_array(&data, batch_size, dimensions);

    Ok(Json(EmbeddingResponse {
        object: String::from("embedding"),
        data: build_embedding_response_data(&data)?,
        model: model_name,
        // Not supported yet, need triton to return usage stats
        // but add a fake one to make `openai_dive` `EmbeddingResponse` happy
        usage: Some(Usage {
            prompt_tokens: 0,
            completion_tokens: Some(0),
            total_tokens: 0,
        }),
    }))
}

#[instrument(skip(request))]
fn build_triton_request(request: EmbeddingParameters) -> anyhow::Result<ModelInferRequest> {
    let (batch_size, triton_input) = match request.input {
        EmbeddingInput::String(i) => {
            let batch_size: i64 = 1;
            let input = vec![i.as_bytes().to_vec()];
            tracing::debug!("EmbeddingInput::String: batch_size={batch_size} input={input:?}");
            (batch_size, input)
        }
        EmbeddingInput::StringArray(i) => {
            let batch_size: i64 = i64::try_from(i.len())?;
            let mut input = Vec::new();
            for s in i {
                input.push(s.as_bytes().to_vec());
            }
            tracing::debug!("EmbeddingInput::StringArray: batch_size={batch_size} input={input:?}");
            (batch_size, input)
        }
        EmbeddingInput::IntegerArray(_) => todo!("IntegerArray"),
        EmbeddingInput::IntegerArrayArray(_) => todo!("IntegerArrayArray"),
    };

    let builder = Builder::new()
        .model_name(request.model)
        .input(
            "text",
            [batch_size, 1],
            InferTensorData::Bytes(triton_input),
        )
        .output(MODEL_OUTPUT_NAME);

    builder.build().context("failed to build triton request")
}

/// # Errors
/// - when the loop counter cannot be converted from `usize` to `u32`
pub fn build_embedding_response_data(input: &[Vec<f32>]) -> anyhow::Result<Vec<Embedding>> {
    let mut embeddings = Vec::new();

    for (i, input) in input.iter().enumerate() {
        let data: Vec<f64> = input.iter().map(|f| f64::from(*f)).collect();
        let embedding = Embedding {
            index: u32::try_from(i)?,
            embedding: EmbeddingOutput::Float(data),
            object: String::from("embedding"),
        };
        embeddings.push(embedding);
    }

    Ok(embeddings)
}

#[must_use]
pub fn transform_triton_f32_array(
    input: &[u8],
    batch_size: usize,
    dimensions: usize,
) -> Vec<Vec<f32>> {
    let mut output: Vec<Vec<f32>> = Vec::new();

    let slice_f32: &[f32] = bytemuck::cast_slice::<u8, f32>(input);

    for i in 0..batch_size {
        let begin = i * dimensions;
        let end = (i + 1) * dimensions;
        let slice_f32: &[f32] = &slice_f32[begin..end];
        output.append(&mut vec![slice_f32.to_vec()]);
    }

    tracing::debug!("{output:?}");
    output
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Read;

    use serde::Deserialize;
    use serde_json;

    use super::*;

    #[derive(Deserialize)]
    struct TestTransformTritonF32ArrayData {
        input_batch_size_1: Vec<u8>,
        input_batch_size_4: Vec<u8>,
        output_batch_size_1: Vec<Vec<f32>>,
        output_batch_size_4: Vec<Vec<f32>>,
    }

    #[test]
    fn test_transform_triton_f32_array() {
        const TESTDATA_FILE: &str =
            "tests/backend.triton.routes.embeddings.test_transform_triton_f32_array";
        let mut test_data = String::new();

        File::open(TESTDATA_FILE)
            .unwrap_or_else(|e| panic!("failed to open testdata file '{TESTDATA_FILE}': {e}"))
            .read_to_string(&mut test_data)
            .unwrap_or_else(|e| panic!("failed to read testdata file '{TESTDATA_FILE}': {e}"));

        let test_data: TestTransformTritonF32ArrayData =
            serde_json::from_str(&test_data).expect("failed to convert testdata to JSON");
        let test_result_batch_size_1 =
            transform_triton_f32_array(&test_data.input_batch_size_1, 1, 1024);

        let test_result_batch_size_4 =
            transform_triton_f32_array(&test_data.input_batch_size_4, 4, 1024);

        // print the result so we can verify we're really doing something
        println!("test_result: {test_result_batch_size_1:?}");
        println!("test_result: {test_result_batch_size_4:?}");

        assert_eq!(test_data.output_batch_size_1, test_result_batch_size_1);
        assert_eq!(test_data.output_batch_size_4, test_result_batch_size_4);
    }
}
