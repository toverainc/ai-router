use clap::Parser;
use figment::providers::Serialized;
use figment::Figment;

use ai_router::config::{AiRouterArguments, AiRouterConfigFile};
use ai_router::startup;
use ai_router::telemetry;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: AiRouterArguments = Figment::new()
        .merge(Serialized::defaults(AiRouterArguments::parse()))
        .extract()
        .unwrap();

    let config_file = AiRouterConfigFile::parse(args.config_file.clone())?;

    telemetry::init_subscriber("ai_router", "info", std::io::stdout, &config_file.daemon);

    startup::run_server(&config_file).await
}
