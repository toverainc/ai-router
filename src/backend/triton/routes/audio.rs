use std::io::Cursor;

use anyhow::Context;
use async_stream::stream;
use axum::response::{IntoResponse, Response};
use axum::Json;
use bytes::Bytes;
use fon::chan::Ch32;
use fon::Audio;
use openai_dive::v1::resources::audio::{
    AudioOutputFormat, AudioTranscriptionFile, AudioTranscriptionParameters,
};
use serde::Serialize;
use tonic::transport::Channel;
use tracing::instrument;
use wavers::Wav;

use crate::utils::{deserialize_bytes_tensor, get_file_extension};
use crate::{
    backend::triton::{
        grpc_inference_service_client::GrpcInferenceServiceClient,
        request::{Builder, InferTensorData},
        utils::get_output_idx,
        ModelInferRequest,
    },
    errors::AiRouterError,
    request::AiRouterRequestData,
    templater::Templater,
};

const MODEL_OUTPUT_NAME: &str = "TRANSCRIPTS";

#[derive(Serialize)]
struct AudioTranscriptionResponse {
    text: String,
}

#[instrument(level = "debug", skip(client, parameters))]
pub(crate) async fn transcriptions(
    mut client: GrpcInferenceServiceClient<Channel>,
    parameters: AudioTranscriptionParameters,
    request_data: AiRouterRequestData,
    templater: Templater,
) -> Result<Response, AiRouterError<String>> {
    // this results in the audio bytes being written to the OLTP endpoint
    // tracing::debug!("triton audio transcriptions request: {:?}", parameters);
    let response_format = parameters.response_format.clone();

    let request = build_triton_request(parameters, request_data, templater)?;
    let request_stream = stream! { yield request };
    let mut stream = client
        .model_stream_infer(tonic::Request::new(request_stream))
        .await
        .context("failed to call triton grpc method model_stream_infer")?
        .into_inner();

    let mut data = String::new();
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

        let Some(idx) = get_output_idx(&infer_response.outputs, MODEL_OUTPUT_NAME) else {
            return Err(AiRouterError::InternalServerError(format!(
                "{MODEL_OUTPUT_NAME} not found in Triton response"
            )));
        };

        let raw_content = infer_response.raw_output_contents[idx].clone();
        let content = deserialize_bytes_tensor(raw_content)?
            .into_iter()
            .collect::<String>();

        data.push_str(&content);
    }

    let response = String::from(data.trim_start());

    match response_format {
        None | Some(AudioOutputFormat::Json | AudioOutputFormat::VerboseJson) => {
            Ok(Json(AudioTranscriptionResponse { text: response }).into_response())
        }
        Some(AudioOutputFormat::Srt | AudioOutputFormat::Text | AudioOutputFormat::Vtt) => {
            Ok(response.into_response())
        }
    }
}

fn get_audio_samples(cursor: Cursor<Bytes>) -> Result<(i64, Vec<f32>), AiRouterError<String>> {
    let mut wav: Wav<f32> = wavers::Wav::new(Box::new(cursor)).map_err(|e| {
        AiRouterError::InternalServerError(format!("failed to process audio file: {e}",))
    })?;

    let n_channels = wav.n_channels();
    let n_samples = wav.n_samples();
    let sample_rate = u32::try_from(wav.sample_rate()).unwrap();

    tracing::info!("n_channels='{n_channels}' n_samples='{n_samples}' sample_rate='{sample_rate}'");

    let samples: &[f32] = &wav.read().unwrap();

    if n_channels != 1 || sample_rate != 16000 {
        tracing::info!("transcoding required!");
    }

    let mut audio_for_triton = match n_channels {
        1 => {
            let audio_original: Audio<Ch32, 1> = Audio::with_f32_buffer(sample_rate, samples);
            audio_original
        }
        2 => {
            let audio_original: Audio<Ch32, 2> = Audio::with_f32_buffer(sample_rate, samples);
            Audio::<Ch32, 1>::with_audio(16_000, &audio_original)
        }
        3 => {
            let audio_original: Audio<Ch32, 3> = Audio::with_f32_buffer(sample_rate, samples);
            Audio::<Ch32, 1>::with_audio(16_000, &audio_original)
        }
        4 => {
            let audio_original: Audio<Ch32, 4> = Audio::with_f32_buffer(sample_rate, samples);
            Audio::<Ch32, 1>::with_audio(16_000, &audio_original)
        }
        5 => {
            let audio_original: Audio<Ch32, 5> = Audio::with_f32_buffer(sample_rate, samples);
            Audio::<Ch32, 1>::with_audio(16_000, &audio_original)
        }
        6 => {
            let audio_original: Audio<Ch32, 6> = Audio::with_f32_buffer(sample_rate, samples);
            Audio::<Ch32, 1>::with_audio(16_000, &audio_original)
        }
        7 => {
            let audio_original: Audio<Ch32, 7> = Audio::with_f32_buffer(sample_rate, samples);
            Audio::<Ch32, 1>::with_audio(16_000, &audio_original)
        }
        8 => {
            let audio_original: Audio<Ch32, 8> = Audio::with_f32_buffer(sample_rate, samples);
            Audio::<Ch32, 1>::with_audio(16_000, &audio_original)
        }
        _ => {
            return Err(AiRouterError::BadRequestError(String::from(
                "audio with > 8 channels not supported",
            )));
        }
    };

    let audio_for_triton = audio_for_triton.as_f32_slice();

    let n_samples = audio_for_triton.len();

    let padding = 160_000 - (n_samples % 160_000);

    let num_samples_padded = n_samples + padding;
    let num_samples_padded_i64 = i64::try_from(num_samples_padded)?;

    tracing::info!("num_samples: {n_samples} - num_samples_padded: {num_samples_padded}");

    let mut vec_wac: Vec<f32> = vec![0.0; num_samples_padded];

    vec_wac[0..n_samples].copy_from_slice(audio_for_triton);

    Ok((num_samples_padded_i64, vec_wac))
}

#[instrument(level = "debug", skip(request))]
fn build_triton_request(
    request: AudioTranscriptionParameters,
    request_data: AiRouterRequestData,
    templater: Templater,
) -> Result<ModelInferRequest, AiRouterError<String>> {
    let audio = match request.file {
        AudioTranscriptionFile::Bytes(b) => {
            let extension = get_file_extension(&b.filename)?;
            if extension != "wav" {
                return Err(AiRouterError::InternalServerError::<String>(format!(
                    "extension {extension:?} not supported to Triton backend",
                )));
            };
            Cursor::new(b.bytes)
        }
        AudioTranscriptionFile::File(_) => {
            return Err(AiRouterError::InternalServerError(String::from(
                "`AudioTranscriptionFile::File` variant not supported",
            )));
        }
    };

    let text_prefix = templater.apply_transcription(request.language, request_data.template)?;

    let (num_samples, audio) = get_audio_samples(audio)?;

    let builder = Builder::new()
        .model_name(request.model)
        .input(
            "TEXT_PREFIX",
            [1, 1],
            InferTensorData::Bytes(vec![text_prefix.as_bytes().to_vec()]),
        )
        .input("WAV", [1, num_samples], InferTensorData::FP32(audio))
        .output(MODEL_OUTPUT_NAME);

    Ok(builder.build()?)
}
