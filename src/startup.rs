use std::collections::HashMap;
use std::sync::Arc;

use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::Router;
use axum_prometheus::PrometheusMetricLayer;
use axum_tracing_opentelemetry::middleware::{OtelAxumLayer, OtelInResponseLayer};
use openai_dive::v1::api::Client as OpenAIClient;
use tokio::net::TcpListener;
use tonic::transport::Channel;
use tower::ServiceBuilder;
use tower_http::request_id::MakeRequestUuid;
use tower_http::{
    trace::TraceLayer,
    trace::{DefaultMakeSpan, DefaultOnResponse},
    ServiceBuilderExt,
};
use tracing::Level;

use crate::backend::init_backends;
use crate::backend::triton::grpc_inference_service_client::GrpcInferenceServiceClient;
use crate::config::AiRouterConfigFile;
use crate::errors::AiRouterError;
use crate::routes;

#[derive(Debug)]
pub struct AppState {
    pub backends: HashMap<String, BackendTypes<OpenAIClient, GrpcInferenceServiceClient<Channel>>>,
    pub config: AiRouterConfigFile,
}

#[derive(Debug)]
pub enum BackendTypes<O, T> {
    OpenAI(O),
    Triton(T),
}

/// Start axum server
///
/// # Errors
/// - when we're unable to connect to the Triton endpoint
/// - when we're unable to bind the `TCPListener` for the axum server
/// - when we're unable to start the axum server
pub async fn run_server(config_file: &AiRouterConfigFile) -> anyhow::Result<()> {
    let (prometheus_layer, metric_handle) = PrometheusMetricLayer::pair();

    let backends = init_backends(config_file).await;
    let state = AppState {
        backends,
        config: config_file.clone(),
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(routes::chat::completion))
        .route("/v1/completions", post(routes::completions::completion))
        .route("/v1/embeddings", post(routes::embeddings::embed))
        .route("/v1/models", get(routes::get))
        .route("/health_check", get(routes::health_check))
        .route("/metrics", get(|| async move { metric_handle.render() }))
        .fallback(fallback)
        .with_state(Arc::new(state))
        .layer(prometheus_layer)
        .layer(OtelInResponseLayer)
        .layer(OtelAxumLayer::default())
        .layer(
            ServiceBuilder::new()
                .set_x_request_id(MakeRequestUuid)
                .layer(
                    TraceLayer::new_for_http()
                        .make_span_with(
                            DefaultMakeSpan::new()
                                .include_headers(true)
                                .level(Level::DEBUG),
                        )
                        .on_response(
                            DefaultOnResponse::new()
                                .include_headers(true)
                                .level(Level::DEBUG),
                        ),
                )
                .propagate_x_request_id(),
        );

    let address = format!(
        "{}:{}",
        config_file.daemon.listen_ip, config_file.daemon.listen_port
    );
    tracing::info!("Starting server at {}", address);

    let listener = TcpListener::bind(address).await?;

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn fallback<T>(request: axum::http::Request<T>) -> impl IntoResponse {
    AiRouterError::UnknownUrl(request)
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("failed to install CTRL+C signal handler");

    if let Err(e) =
        tokio::task::spawn_blocking(opentelemetry::global::shutdown_tracer_provider).await
    {
        tracing::error!("failed to shutdown OpenTelemetry tracer provider: {e}");
    }
}
