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
        .unwrap_or_else(|e| {
            panic!("failed to merge command line arguments: {e}");
        });

    let config_file = AiRouterConfigFile::parse(args.config_file.clone())?;

    if args.dump_config {
        println!(
            "{}",
            toml::to_string(&config_file).unwrap_or_else(|e| {
                panic!("failed to dump config file: {e}");
            })
        );
        return Ok(());
    }

    telemetry::init_subscriber("ai_router", "info", &config_file.daemon)?;

    startup::run_server(&config_file).await
}
