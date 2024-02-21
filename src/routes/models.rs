use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use openai_dive::v1::resources::model::{ListModelResponse, Model};
use tracing::instrument;

use crate::startup::AppState;

#[instrument(name = "routes::models::get", skip(state))]
pub async fn get(State(state): State<Arc<AppState>>) -> Response {
    let mut model_names: Vec<String> = Vec::new();

    for model_type in state.config.models.keys() {
        for model in state.config.models.get(model_type).unwrap().keys() {
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
