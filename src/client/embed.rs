use std::time::Instant;

use clap::Parser;
use openai_dive::v1::{
    api::Client as OpenAIClient,
    resources::embedding::{EmbeddingInput, EmbeddingParameters},
};

#[derive(Debug, Parser)]
struct Args {
    #[arg(short, long, default_value_t = String::from("test"))]
    api_key: String,

    #[arg(short, long, required = true)]
    input: Vec<String>,

    #[arg(short, long, default_value_t = String::from("BAAI/bge-large-en-v1.5"))]
    model: String,

    #[arg(short, long, default_value_t = String::from("http://localhost:3000/v1"))]
    url: String,
}

fn transform_input(input: Vec<String>) -> EmbeddingInput {
    if input.len() == 1 {
        EmbeddingInput::String(input[0].clone())
    } else {
        EmbeddingInput::StringArray(input)
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let client = OpenAIClient {
        api_key: args.api_key,
        base_url: args.url,
        http_client: reqwest::Client::new(),
        organization: None,
        project: None,
    };

    let request = EmbeddingParameters {
        dimensions: None,
        encoding_format: None,
        input: transform_input(args.input),
        model: args.model,
        user: None,
    };

    let start = Instant::now();

    match client.embeddings().create(request).await {
        Err(e) => println!("{e}"),
        Ok(o) => println!("{o:#?}"),
    };

    let duration = start.elapsed();

    println!("request took {duration:?}");
}
