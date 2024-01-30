use opentelemetry::trace::TraceError;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::trace as sdktrace;
use opentelemetry_sdk::{runtime, Resource};
use tracing::subscriber::set_global_default;
use tracing_bunyan_formatter::{BunyanFormattingLayer, JsonStorageLayer};
use tracing_log::LogTracer;
use tracing_subscriber::fmt::MakeWriter;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Registry};

use crate::config::AiRouterDaemon;

fn init_tracer(airouter_daemon_config: &AiRouterDaemon) -> Result<sdktrace::Tracer, TraceError> {
    let Some(otlp_endpoint) = airouter_daemon_config.otlp_endpoint.clone() else {
        return Err(TraceError::Other("otlp_endpoint not set".into()));
    };
    opentelemetry::global::set_text_map_propagator(opentelemetry_jaeger::Propagator::new());
    opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint(otlp_endpoint),
        )
        .with_trace_config(sdktrace::config().with_resource(Resource::new(vec![
            opentelemetry::KeyValue::new("service.name", "ai_router"),
            opentelemetry::KeyValue::new(
                "service.instance.id",
                airouter_daemon_config.instance_id.clone(),
            ),
        ])))
        .install_batch(runtime::Tokio)
}

/// Compose multiple layers into a `tracing`'s subscriber.
///
/// # Implementation Notes
///
/// We are using `impl Subscriber` as return type to avoid having to spell out the actual
/// type of the returned subscriber, which is indeed quite complex.
///
/// # Panics
///
/// Panics if `LogTracer` cannot be initialized
///
pub fn init_subscriber<Sink>(
    name: &str,
    env_filter: &str,
    sink: Sink,
    airouter_daemon_config: &AiRouterDaemon,
) where
    Sink: for<'a> MakeWriter<'a> + Send + Sync + 'static,
{
    LogTracer::init().expect("Failed to set logger");

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(env_filter));
    let formatting_layer = BunyanFormattingLayer::new(name.into(), sink);

    let registry = Registry::default()
        .with(env_filter)
        .with(JsonStorageLayer)
        .with(formatting_layer);

    if let Ok(tracer) = init_tracer(airouter_daemon_config) {
        let tracer_layer = tracing_opentelemetry::layer().with_tracer(tracer);
        set_global_default(registry.with(tracer_layer)).expect("Failed to set subscriber");
    } else {
        set_global_default(registry).expect("Failed to set subscriber");
    }
}
