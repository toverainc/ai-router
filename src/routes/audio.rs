use std::io::Read;
use std::str::FromStr;
use std::sync::Arc;

use anyhow::Context;
use axum::extract::{Multipart, State as AxumState};
use axum::response::Response;
use axum::Json;
use bytes::Bytes;
use openai_dive::v1::resources::audio::{
    AudioOutputFormat, AudioSpeechParameters, AudioTranscriptionBytes, AudioTranscriptionFile,
    AudioTranscriptionParameters, TimestampGranularity,
};
use tracing::instrument;

use crate::backend::openai::routes as openai_routes;
use crate::backend::triton::routes as triton_routes;
use crate::config::AiRouterModelType;
use crate::errors::AiRouterError;
use crate::request::AiRouterRequestData;
use crate::state::{BackendTypes, State};
use crate::utils::get_file_extension;

pub async fn speech(
    AxumState(state): AxumState<Arc<State>>,
    Json(mut parameters): Json<AudioSpeechParameters>,
) -> Result<Response, AiRouterError<String>> {
    if let Some(models) = state.config.models.get(&AiRouterModelType::AudioSpeech) {
        if let Some(model) = models.get(&parameters.model) {
            if let Some(backend_model) = model.backend_model.clone() {
                parameters.model = backend_model;
            }

            let model_backend = model.backend.as_ref().map_or("default", |m| m);

            let Some(backend) = state.backends.get(model_backend) else {
                return Err(AiRouterError::InternalServerError::<String>(format!(
                    "backend {model_backend} not found"
                )));
            };

            match &backend.client {
                BackendTypes::OpenAI(c) => {
                    return openai_routes::audio::speech(c, parameters).await;
                }
                BackendTypes::Triton(_c) => {
                    return Err(AiRouterError::BadRequestError::<String>(String::from(
                        "create speech to Triton backend not implemented yet",
                    )));
                }
            }
        }
    }

    Err(AiRouterError::ModelNotFound::<String>(
        parameters.model.clone(),
    ))
}

#[instrument(level = "debug", skip(state, multipart))]
pub async fn transcriptions(
    AxumState(state): AxumState<Arc<State>>,
    // Multipart must be the last argument
    // <https://github.com/tokio-rs/axum/discussions/1600>
    multipart: Multipart,
) -> Result<Response, AiRouterError<String>> {
    let mut parameters = match build_transcription_parameters(multipart).await {
        Ok(o) => o,
        Err(e) => {
            return Err(e);
        }
    };

    if let Some(models) = state
        .config
        .models
        .get(&AiRouterModelType::AudioTranscriptions)
    {
        if let Some(model) = models.get(&parameters.model) {
            if let Some(backend_model) = model.backend_model.clone() {
                parameters.model = backend_model;
            }

            let model_backend = model.backend.as_ref().map_or("default", |m| m);

            let Some(backend) = state.backends.get(model_backend) else {
                return Err(AiRouterError::InternalServerError::<String>(format!(
                    "backend {model_backend} not found"
                )));
            };

            match &backend.client {
                BackendTypes::OpenAI(c) => {
                    return openai_routes::audio::transcriptions(c, parameters).await
                }
                BackendTypes::Triton(c) => {
                    let request_data = AiRouterRequestData::build(
                        model,
                        model.backend_model.clone(),
                        &parameters.model,
                        &state,
                    )?;

                    return triton_routes::audio::transcriptions(
                        c.clone(),
                        parameters,
                        request_data,
                        state.templater.clone(),
                    )
                    .await;
                }
            }
        }
    }

    return Err(AiRouterError::ModelNotFound::<String>(parameters.model));
}

#[instrument(level = "debug", skip(multipart))]
pub async fn build_transcription_parameters(
    mut multipart: Multipart,
) -> Result<AudioTranscriptionParameters, AiRouterError<String>> {
    let mut parameters = AudioTranscriptionParameters {
        file: AudioTranscriptionFile::File(String::new()),
        language: None,
        model: String::new(),
        prompt: None,
        response_format: None,
        temperature: None,
        timestamp_granularities: None,
    };

    let mut timestamp_granularities: Vec<TimestampGranularity> = Vec::new();

    while let Ok(Some(field)) = multipart.next_field().await {
        tracing::trace!("{field:#?}");
        let field_name = field
            .name()
            .ok_or(AiRouterError::InternalServerError::<String>(String::from(
                "failed to read field name",
            )))?
            .to_string();

        if field_name == "file" {
            let filename: String =
                String::from(field.file_name().context("failed to read field filename")?);

            is_audio_format_supported(&filename)?;

            let field_data_vec = get_field_data_vec(&field.bytes().await?, &field_name)?;
            let bytes = AudioTranscriptionBytes {
                bytes: Bytes::copy_from_slice(&field_data_vec),
                filename,
            };

            parameters.file = AudioTranscriptionFile::Bytes(bytes);
        } else if field_name == "language" {
            let field_data_vec = get_field_data_vec(&field.bytes().await?, &field_name)?;

            parameters.language = Some(String::from_utf8(field_data_vec)?);
        } else if field_name == "model" {
            let field_data_vec = get_field_data_vec(&field.bytes().await?, &field_name)?;

            parameters.model = String::from_utf8(field_data_vec)?;
        } else if field_name == "prompt" {
            let field_data_vec = get_field_data_vec(&field.bytes().await?, &field_name)?;

            parameters.prompt = Some(String::from_utf8(field_data_vec)?);
        } else if field_name == "response_format" {
            let field_data_vec = get_field_data_vec(&field.bytes().await?, &field_name)?;
            let response_format = String::from_utf8(field_data_vec)?;
            let response_format = AudioOutputFormat::from_str(&response_format)?;

            parameters.response_format = Some(response_format);
        } else if field_name == "temperature" {
            let field_data_vec = get_field_data_vec(&field.bytes().await?, &field_name)?;

            parameters.temperature = Some(String::from_utf8(field_data_vec)?.parse()?);
        } else if field_name == "timestamp_granularities[]" {
            let field_data_vec = get_field_data_vec(&field.bytes().await?, &field_name)?;
            let granularity = String::from_utf8(field_data_vec)?;
            let granularity = TimestampGranularity::from_str(&granularity)?;

            timestamp_granularities.push(granularity);
        }
    }

    if !timestamp_granularities.is_empty()
        && parameters.response_format == Some(AudioOutputFormat::VerboseJson)
    {
        parameters.timestamp_granularities = Some(timestamp_granularities);
    }

    Ok(parameters)
}

fn is_audio_format_supported(filename: &str) -> Result<(), AiRouterError<String>> {
    const SUPPORTED_EXTENSIONS: [&str; 10] = [
        "flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "ogg", "ogm", "wav", "webm",
    ];

    let extension = get_file_extension(filename)?;

    tracing::debug!("filename: {filename} - extension: {extension}");

    if !SUPPORTED_EXTENSIONS.contains(&extension) {
        return Err(AiRouterError::InternalServerError::<String>(format!(
            "extension {extension:?} not supported",
        )));
    }

    Ok(())
}

fn get_field_data_vec(data: &Bytes, name: &str) -> Result<Vec<u8>, AiRouterError<String>> {
    let field_data_vec: Vec<u8> = match data.bytes().collect() {
        Ok(o) => o,
        Err(e) => {
            return Err(AiRouterError::InternalServerError::<String>(format!(
                "failed to read {name} field: {e}",
            )));
        }
    };

    Ok(field_data_vec)
}
