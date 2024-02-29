use axum::{
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use openai_dive::v1::error::APIError;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DefaultOnError};

#[derive(Debug)]
pub enum AiRouterError<T> {
    InternalServerError(String),
    ModelNotFound(String),
    UnknownUrl(Request<T>),
    WrappedOpenAi(OpenAIError),
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
            Self::WrappedOpenAi(error) => {
                (StatusCode::INTERNAL_SERVER_ERROR, Json(error)).into_response()
            }
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIErrorCode {
    InvalidApiKey,
    ModelNotFound,
    UnknownUrl,
}

#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAIErrorType {
    InternalServerError,
    InvalidRequestError,
    #[default]
    Undefined,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpenAIError {
    pub error: OpenAIErrorData,
}

#[serde_as]
#[derive(Debug, Deserialize, Serialize)]
pub struct OpenAIErrorData {
    #[serde_as(deserialize_as = "DefaultOnError")]
    pub code: Option<OpenAIErrorCode>,
    pub message: String,
    #[serde_as(deserialize_as = "DefaultOnError")]
    pub param: Option<String>,
    #[serde_as(deserialize_as = "DefaultOnError")]
    pub r#type: OpenAIErrorType,
}

pub fn transform_openai_dive_apierror(input: &APIError) -> AiRouterError<String> {
    let default = OpenAIErrorData {
        code: None,
        message: String::from("failed to transform error from OpenAI backend client"),
        param: None,
        r#type: OpenAIErrorType::InternalServerError,
    };

    let error: OpenAIErrorData = match input {
        APIError::EndpointError(s)
        | APIError::FileError(s)
        | APIError::InvalidRequestError(s)
        | APIError::ParseError(s)
        | APIError::StreamError(s) => serde_json::from_str(s).unwrap_or_else(|e| {
            tracing::error!("failed to deserialize {s}: {e}");
            default
        }),
    };

    AiRouterError::WrappedOpenAi(OpenAIError { error })
}
