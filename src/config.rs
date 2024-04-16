use std::collections::HashMap;

use anyhow::{anyhow, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_with::{formats::PreferMany, serde_as, skip_serializing_none, OneOrMany};
use uuid::Uuid;

const DEFAULT_CONFIG_FILE: &str = "/etc/ai-router.toml";

pub type AiRouterModels = HashMap<AiRouterModelType, HashMap<String, AiRouterModel>>;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AiRouterBackendType {
    OpenAI,
    Triton,
}

#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AiRouterModelType {
    AudioSpeech,
    AudioTranscriptions,
    ChatCompletions,
    Embeddings,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AiRouterBackend {
    pub api_key: Option<String>,
    #[serde(rename = "type")]
    pub backend_type: AiRouterBackendType,
    pub base_url: String,
    pub default: Option<bool>,
    pub mode: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AiRouterConfigFile {
    pub backends: HashMap<String, AiRouterBackend>,
    pub daemon: AiRouterDaemon,
    pub models: AiRouterModels,
    pub title: String,
}

impl AiRouterConfigFile {
    /// Parse config file
    ///
    /// # Errors
    /// - when file at path cannot be opened
    /// - when file content cannot be deserialized into `AiRouterConfigFile`
    pub fn parse(path: String) -> Result<Self> {
        let config = std::fs::read_to_string(path)?;
        let config: Self = toml::from_str(&config)?;
        if let Err(e) = config.validate() {
            return Err(anyhow!("config file validation failed: {e}"));
        }
        Ok(config)
    }

    fn check_backends(&self) -> Result<()> {
        if self.backends.is_empty() {
            return Err(anyhow!("no backends defined in config file"));
        }
        Ok(())
    }

    fn check_default_backends(&self) -> Result<()> {
        if self.num_default_backends() > 1 {
            return Err(anyhow!("multiple backends set as default"));
        }
        Ok(())
    }

    fn check_default_models(&self) -> Result<()> {
        for model_type in self.models.values() {
            let mut num_default_models = 0;
            for model in model_type.values() {
                if model.default.unwrap_or(false) {
                    num_default_models += 1;
                    if num_default_models > 1 {
                        return Err(anyhow!("multiple models set as default"));
                    }
                }
            }
        }

        Ok(())
    }

    fn check_model_backends(&self) -> Result<()> {
        for model_type in self.models.values() {
            for (model_name, model) in model_type {
                if let Some(model_backend) = &model.backend {
                    if !self.backends.contains_key(model_backend) {
                        return Err(anyhow!(
                            "backend `{}` configured for model `{model_name}` does not exist",
                            model_backend
                        ));
                    }
                } else if self.num_default_backends() < 1 {
                    return Err(anyhow!(
                        "model `{model_name}` has no backend configured but no default backend exists",
                    ));
                }
            }
        }
        Ok(())
    }

    fn check_models(&self) -> Result<()> {
        if self.models.is_empty() {
            return Err(anyhow!("no models defined in config file"));
        }
        Ok(())
    }

    fn num_default_backends(&self) -> usize {
        let mut num_default_backends = 0;

        for backend in self.backends.values() {
            if backend.default.unwrap_or(false) {
                num_default_backends += 1;
            }
        }

        num_default_backends
    }

    fn validate(&self) -> Result<()> {
        self.check_backends()?;
        self.check_models()?;
        self.check_default_backends()?;
        self.check_default_models()?;
        self.check_model_backends()?;

        Ok(())
    }
}

#[serde_as]
#[skip_serializing_none]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AiRouterDaemon {
    #[serde(default = "default_api_key")]
    #[serde_as(deserialize_as = "OneOrMany<_, PreferMany>")]
    pub api_key: Vec<String>,
    #[serde(default = "default_instance_id")]
    pub instance_id: String,
    pub listen_ip: String,
    pub listen_port: u16,
    /// HTTP request body limit in MiB
    #[serde(default = "default_max_body_size")]
    pub max_body_size: usize,
    pub otlp_endpoint: Option<String>,
}

#[skip_serializing_none]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AiRouterModel {
    pub backend: Option<String>,
    pub backend_model: Option<String>,
    pub default: Option<bool>,
    pub hf_model_name: Option<String>,
    pub max_input: Option<usize>,
    pub max_tokens: Option<u32>,
    pub prompt_format: Option<String>,
}

#[derive(Parser, Debug, Serialize, Deserialize)]
pub struct AiRouterArguments {
    #[arg(long, short = 'c', default_value_t = String::from(DEFAULT_CONFIG_FILE))]
    pub config_file: String,
    #[arg(long, short = 'd', default_value_t = false)]
    pub dump_config: bool,
}

const fn default_api_key() -> Vec<String> {
    Vec::new()
}

fn default_instance_id() -> String {
    String::from(Uuid::new_v4())
}

const fn default_max_body_size() -> usize {
    2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ok() {
        let config: Result<AiRouterConfigFile> =
            AiRouterConfigFile::parse(String::from("ai-router.toml.example"));

        match config {
            Ok(o) => println!(
                "{}",
                serde_json::to_string_pretty(&o).expect("failed to convert config file to JSON")
            ),
            Err(e) => panic!("{e:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "config file validation failed: multiple backends set as default")]
    fn test_multiple_default_backends() {
        let config: Result<AiRouterConfigFile> = AiRouterConfigFile::parse(String::from(
            "tests/ai-router.toml.multiple_default_backends",
        ));

        match config {
            Ok(o) => println!(
                "{}",
                serde_json::to_string_pretty(&o).expect("failed to convert config file to JSON")
            ),
            Err(e) => panic!("{e:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "config file validation failed: multiple models set as default")]
    fn test_multiple_default_models() {
        let config: Result<AiRouterConfigFile> =
            AiRouterConfigFile::parse(String::from("tests/ai-router.toml.multiple_default_models"));

        match config {
            Ok(o) => println!(
                "{}",
                serde_json::to_string_pretty(&o).expect("failed to convert config file to JSON")
            ),
            Err(e) => panic!("{e:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "config file validation failed: backend ")]
    fn test_model_backend_invalid() {
        let config: Result<AiRouterConfigFile> =
            AiRouterConfigFile::parse(String::from("tests/ai-router.toml.model_backend_invalid"));

        match config {
            Ok(o) => println!(
                "{}",
                serde_json::to_string_pretty(&o).expect("failed to convert config file to JSON")
            ),
            Err(e) => panic!("{e:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "config file validation failed: no backends defined in config file")]
    fn test_no_backends() {
        let config: Result<AiRouterConfigFile> =
            AiRouterConfigFile::parse(String::from("tests/ai-router.toml.no_backends"));

        match config {
            Ok(o) => println!(
                "{}",
                serde_json::to_string_pretty(&o).expect("failed to convert config file to JSON")
            ),
            Err(e) => panic!("{e:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "config file validation failed: model ")]
    fn test_no_default_backend() {
        let config: Result<AiRouterConfigFile> =
            AiRouterConfigFile::parse(String::from("tests/ai-router.toml.no_default_backend"));

        match config {
            Ok(o) => println!(
                "{}",
                serde_json::to_string_pretty(&o).expect("failed to convert config file to JSON")
            ),
            Err(e) => panic!("{e:?}"),
        }
    }

    #[test]
    #[should_panic(expected = "config file validation failed: no models defined in config file")]
    fn test_no_models() {
        let config: Result<AiRouterConfigFile> =
            AiRouterConfigFile::parse(String::from("tests/ai-router.toml.no_models"));

        match config {
            Ok(o) => println!(
                "{}",
                serde_json::to_string_pretty(&o).expect("failed to convert config file to JSON")
            ),
            Err(e) => panic!("{e:?}"),
        }
    }
}
