use minijinja::{context, path_loader, Environment, Error, ErrorKind, State};
use tracing::instrument;

use crate::errors::AiRouterError;

// default template that works for both chat completions and legacy completions
const TPL_COMPLETIONS_DEFAULT: &str = r"
{%- for msg in messages %}
{%- if msg.content %}{{ msg.content }}
{%- elif msg.prompt %}{{ msg.prompt }}{% endif -%}
{% endfor -%}
";

pub enum TemplateType {
    ChatCompletion,
    LegacyCompletion,
}

#[derive(Clone, Debug)]
pub struct Templater {
    env: Environment<'static>,
}

impl Templater {
    pub fn new(path: &str) -> Result<Self, AiRouterError<String>> {
        let mut env = Environment::new();

        env.add_function("raise_exception", raise_exception);
        env.add_template("default_completions", TPL_COMPLETIONS_DEFAULT)?;
        env.set_loader(path_loader(path));

        Ok(Self { env })
    }

    #[instrument(level = "debug", skip(self, input, template, template_type))]
    pub fn apply_completions<T: serde::Serialize>(
        self,
        input: &T,
        template: Option<String>,
        template_type: &TemplateType,
    ) -> Result<String, AiRouterError<String>> {
        // load default template first
        let mut tpl = self.env.get_template("default_completions").map_err(|e| {
            AiRouterError::InternalServerError(format!(
                "failed to load default completions template: {e}"
            ))
        })?;

        if let Some(template) = template {
            let template = match template_type {
                TemplateType::ChatCompletion => format!("chat/{template}.j2"),
                TemplateType::LegacyCompletion => format!("completions/{template}.j2"),
            };

            tpl = self.env.get_template(&template).map_err(|e| {
                AiRouterError::InternalServerError(format!(
                    "failed to load completions template {template}: {e}",
                ))
            })?;
        }

        let ctx = context! {messages => input};

        let mut rendered = tpl.render(ctx).map_err(|e| {
            AiRouterError::InternalServerError(format!(
                "failed to render completions template: {e}"
            ))
        })?;

        rendered.push_str("\nASSISTANT:");

        tracing::debug!("input after applying completions template: {rendered}");

        Ok(rendered)
    }

    pub fn apply_transcription(
        self,
        language: Option<String>,
        template: Option<String>,
    ) -> Result<String, AiRouterError<String>> {
        let mut rendered = String::new();

        if template.is_none() {
            return Ok(rendered);
        }

        if let Some(template) = template {
            let template = format!("transcription/{template}.j2");
            let tpl = self.env.get_template(&template).map_err(|e| {
                AiRouterError::InternalServerError(format!(
                    "failed to load transcription template {template}: {e}",
                ))
            })?;

            let language = language.unwrap_or_else(|| String::from("en"));

            let ctx = context! {language => language};

            rendered = tpl.render(ctx).map_err(|e| {
                AiRouterError::InternalServerError(format!(
                    "failed to render transcription template: {e}"
                ))
            })?;
        }

        tracing::debug!("prefix after applying transcription template: {rendered}");

        Ok(rendered)
    }
}

fn raise_exception(_state: &State, msg: String) -> Result<String, Error> {
    Err(Error::new(ErrorKind::SyntaxError, msg))
}
