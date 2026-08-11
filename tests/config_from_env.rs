use std::path::Path;
use std::process::Command;

use ai_router::config::AiRouterConfigFile;

#[test]
fn environment_overrides_file_config() {
    let config_file = Path::new(env!("CARGO_MANIFEST_DIR")).join("ai-router.toml.example");
    let output = Command::new(env!("CARGO_BIN_EXE_ai-router"))
        .arg("--config-file")
        .arg(config_file)
        .arg("--dump-config")
        .env_clear()
        .env("AI_ROUTER_BACKENDS_OPENAI", "{api_key=from_env}")
        .output()
        .expect("failed to run ai-router");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "ai-router failed:\n{stderr}");

    let stdout = String::from_utf8(output.stdout).expect("config output is not valid UTF-8");
    let config: AiRouterConfigFile =
        toml::from_str(&stdout).expect("config output is not valid TOML");
    let backend = config
        .backends
        .get("openai")
        .expect("OpenAI backend missing from config output");

    assert_eq!(backend.api_key.as_deref(), Some("from_env"));
    assert_eq!(backend.base_url, "https://api.openai.com/v1");
}
