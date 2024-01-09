use clap::Parser;
use figment::providers::{Env, Serialized};
use figment::Figment;

use ai_router::config::Config;
use ai_router::startup;
use ai_router::telemetry;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config: Config = Figment::new()
        .merge(Env::prefixed("AI_ROUTER_"))
        .merge(Serialized::defaults(Config::parse()))
        .extract()
        .unwrap();

    telemetry::init_subscriber(
        "ai_router",
        "info",
        std::io::stdout,
        config.otlp_endpoint.clone(),
    );

    startup::run_server(config).await
}
