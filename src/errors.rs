use axum::{
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum AiRouterError<T> {
    InternalServerError(String),
    ModelNotFound(String),
    UnknownUrl(Request<T>),
}

impl<E, T> From<E> for AiRouterError<T>
where
    E: Into<anyhow::Error> + std::fmt::Debug,
{
    fn from(err: E) -> Self {
        Self::InternalServerError(format!("{err:?}"))
    }
}

impl<T> IntoResponse for AiRouterError<T> {
    fn into_response(self) -> Response {
        match self {
            Self::InternalServerError(msg) => {
                let error = OpenAIError {
                    error: OpenAIErrorData {
                        code: None,
                        message: msg,
                        param: None,
                        r#type: OpenAIErrorType::InternalServerError,
                    },
                };
                (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
            }
            Self::ModelNotFound(model_name) => {
                let error = OpenAIError {
                    error: OpenAIErrorData {
                        code: Some(OpenAIErrorCode::ModelNotFound),
                        message: format!("The model `{}` does not exist.", model_name),
                        param: None,
                        r#type: OpenAIErrorType::InvalidRequestError,
                    },
                };
                (StatusCode::NOT_FOUND, Json(error)).into_response()
            }
            Self::UnknownUrl(request) => {
                let error = OpenAIError {
                    error: OpenAIErrorData {
                        code: Some(OpenAIErrorCode::UnknownUrl),
                        message: format!(
                            "Unknown request URL: {} {}. Please check the URL for typos.",
                            request.method(),
                            request.uri()
                        ),
                        param: None,
                        r#type: OpenAIErrorType::InvalidRequestError,
                    },
                };
                (StatusCode::NOT_FOUND, Json(error)).into_response()
            }
        }
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
    InternalServerError,
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
