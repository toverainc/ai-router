use axum::response::{IntoResponse, Response};
use openai_dive::v1::{
    api::Client,
    resources::audio::{AudioSpeechParameters, AudioSpeechResponseFormat},
};

use crate::errors::{transform_openai_dive_apierror, AiRouterError};

pub async fn speech(
    client: &Client,
    parameters: AudioSpeechParameters,
) -> Result<Response, AiRouterError<String>> {
    let content_type = match parameters
        .response_format
        .clone()
        .unwrap_or(AudioSpeechResponseFormat::Mp3)
    {
        AudioSpeechResponseFormat::Aac => "audio/aac",
        AudioSpeechResponseFormat::Flac => "audio/flac",
        AudioSpeechResponseFormat::Mp3 => "audio/mpeg",
        AudioSpeechResponseFormat::Opus => "audio/opus",
        AudioSpeechResponseFormat::Pcm => "audio/pcm",
        AudioSpeechResponseFormat::Wav => "audio/wav",
    };

    let response = client
        .audio()
        .create_speech(parameters)
        .await
        .map_err(|e| transform_openai_dive_apierror(&e))?;

    Ok(([("content-type", content_type)], response.bytes).into_response())
}
