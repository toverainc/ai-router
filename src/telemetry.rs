use opentelemetry::trace::TraceError;
use opentelemetry::{global, KeyValue};
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace as sdktrace;
use opentelemetry_sdk::{runtime, Resource};
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{layer::SubscriberExt, EnvFilter, Layer};

use crate::config::AiRouterDaemon;

fn init_tracer(
    name: &str,
    airouter_daemon_config: AiRouterDaemon,
) -> Result<sdktrace::Tracer, TraceError> {
    opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(
            opentelemetry_otlp::new_exporter().tonic().with_endpoint(
                airouter_daemon_config
                    .otlp_endpoint
                    .expect("init_tracer with otlp_endpoint None"),
            ),
        )
        .with_trace_config(
            sdktrace::config()
                .with_resource(Resource::new(vec![
                    KeyValue::new("service.name", name.to_owned()),
                    KeyValue::new("service.instance.id", airouter_daemon_config.instance_id),
                ]))
                .with_sampler(sdktrace::Sampler::AlwaysOn),
        )
        .install_batch(runtime::Tokio)
}

/// Compose multiple layers into a `tracing`'s subscriber.
///
/// # Implementation Notes
///
/// We are using `impl Subscriber` as return type to avoid having to spell out the actual
/// type of the returned subscriber, which is indeed quite complex.
///
/// # Errors
/// - when `env_filter` directives cannot be parsed
/// - when `init_tracer` returns an error
pub fn init_subscriber(
    name: &str,
    env_filter: &str,
    airouter_daemon_config: &AiRouterDaemon,
) -> anyhow::Result<()> {
    global::set_text_map_propagator(TraceContextPropagator::new());

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(env_filter))
        .add_directive("otel::tracing=info".parse()?)
        .add_directive("otel=debug".parse()?);

    let telemetry_layer = if airouter_daemon_config.otlp_endpoint.is_some() {
        let tracer = init_tracer(name, airouter_daemon_config.clone())?;

        Some(
            tracing_opentelemetry::layer()
                .with_error_records_to_exceptions(true)
                .with_tracer(tracer),
        )
    } else {
        None
    };

    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true)
        .with_thread_names(true)
        .with_span_events(FmtSpan::CLOSE)
        .boxed();

    tracing_subscriber::registry()
        .with(env_filter)
        .with(telemetry_layer)
        .with(fmt_layer)
        .init();

    Ok(())
}
