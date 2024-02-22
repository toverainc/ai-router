use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "An error occurred while trying to fulfill your request.",
        )
            .into_response()
    }
}

impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIErrorCode {
    InvalidApiKey,
    ModelNotFound,
    UnknownUrl,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIErrorType {
    InvalidRequestError,
}

#[derive(Deserialize, Serialize)]
pub struct OpenAIError {
    pub error: OpenAIErrorData,
}

#[derive(Deserialize, Serialize)]
pub struct OpenAIErrorData {
    pub code: Option<OpenAIErrorCode>,
    pub message: String,
    pub param: Option<String>,
    pub r#type: OpenAIErrorType,
}
