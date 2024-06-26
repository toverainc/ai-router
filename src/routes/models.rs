use std::sync::Arc;

use axum::extract::State as AxumState;
use axum::response::{IntoResponse, Response};
use axum::Json;
use openai_dive::v1::resources::model::{ListModelResponse, Model};
use tracing::instrument;

use crate::errors::AiRouterError;
use crate::state::State;

#[instrument(skip(state))]
pub async fn get(AxumState(state): AxumState<Arc<State>>) -> Response {
    let mut model_names: Vec<String> = Vec::new();

    let model_types = state.config.models.keys();
    for model_type in model_types {
        let Some(models) = state.config.models.get(model_type) else {
            return AiRouterError::InternalServerError::<String>(format!(
                "failed to get models of type {model_type:?}"
            ))
            .into_response();
        };
        for model in models.keys() {
            model_names.push(model.clone());
        }
    }

    let mut response: ListModelResponse = ListModelResponse {
        data: Vec::new(),
        object: String::from("list"),
    };

    for model_name in model_names {
        let model = Model {
            id: model_name,
            created: 1_700_000_000,
            object: String::from("model"),
            owned_by: String::from("original owners"),
        };
        response.data.push(model);
    }

    Json(response).into_response()
}
