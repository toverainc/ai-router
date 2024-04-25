# AI Router - AI Model Serving Flexibility and Performance

AI Router presents an OpenAI API compatible HTTP interface to clients. AI Router can then take input from OpenAI API clients and route requests to any number of backends and models with translation across protocol, model naming, and prompt formatting. Currently supports OpenAI API compatible backends or [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server) via gRPC.

Written in 100% pure Rust.

AI Router could be thought of as an API Gateway for AI and/or a free and open source [Nividia Inference Microservice](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/). Or a SBC (Session Border Controller) if you're a friend from the VoIP days.

Point your clients at AI Router and use any combination of Triton Inference Server, vLLM, OpenAI, Mistral, llama.cpp, etc hosted models and AI Router will handle the rest.

## Feature Support

- API keys for clients and OpenAI compatible backends.
- Model name mapping and rewriting.
- Transparent prompt formatting.
- Mapping between exposed API paths to backends and models on different paths, ports, IPs, etc.
- Define default backends and model names so for example OpenAI API compatible clients can run with default model settings and AI Router will pass them to defined default or wildcard matching model/endpoint names. Or not (return error).
- Extremely high performance (thanks to Rust)!
- Low system resource utilization (Rust FTW).
- Streaming support with fixups for Triton Inference Server (required ATM).
- Support mix of client stream request/stream to backend.
- More to come!

### Supported Inference Types vs Backend Types

| Inference Type        | OpenAI backend     | Triton backend     |
| :-------------------- | :----------------: | :----------------: |
| Audio > Speech        | :white_check_mark: | :x:                |
| Audio > Transcription | :x:                | :x:                |
| Audio > Translation   | :x:                | :x:                |
| Chat                  | :white_check_mark: | :white_check_mark: |
| Embeddings            | :white_check_mark: | :white_check_mark: |
| Images                | :x:                | :x:                |
| Legacy Completions    | :x:                | :white_check_mark: |

## Usage Example

You have Triton Inference Server, vLLM, HF TEI/TGI, or any other OpenAI compatible local embeddings/LLM model(s) served. You may also have API keys for OpenAI, Mistral Le Platforme, Anyscale, etc. Or all of the above, or not.

For example, AI Router can:

- Define API key for clients.
- Expose your hosted models as "tovera-chat-v1" and "tovera-embed-v1" (any names you want). Clients can request these models and AI Router will pass them to your defined backend(s) while not leaking anything - responses to clients will be re-written on the fly with your model names. You can change any aspect of the backend/model configuration and clients can continue to request your model name(s) and not know the difference. Or you can update version and support any variant of this configuration.
- Intercept any model name and pass to any of those backends, in any mixture of backend and model configuration, with rewriting.
- Define a catch-all/default model name for embeddings or chat so OpenAI clients can work with vLLM/Triton/HF TEI etc served models without configuration changes.
- Wildcard match for things like different/default OpenAI model names/versions and map to any backend/model you define.
- Much more in any number or combination of these across any backend, model name, or currently supported function (embeddings or chat).

Check the [example configuration file in this repo](airouter.toml.example) to get an idea.

## Performance
Written in pure Rust we have done performance testing showing AI Router competitive with (far more limited) nginx for request rate and latency. AI Router will not be the limiting factor in terms of request rate and contributes minimally to latency for the all-important TTFT (time to first token).

## Get started

Clone this repo:

```bash
git clone --recurse-submodules https://github.com/toverainc/ai-router.git
```

### Build

```bash
cd ai-router
./utils.sh build
```

### Start

Edit the configuration file (airouter.toml) then start:

```bash
./utils.sh run-local
```

### Test

Your OpenAI endpoint is available at `http://[your_host]:3000/v1`

There are example clients in `client/`

Example with streaming:

`python3 client/openai_chatcompletion_client.py -S`

When using Triton Inference Server especially tokens stream so quickly python/TTY can't keep up so it will appear as though the stream outputs by sentence/paragraph. Specify `-n` for the client to insert newlines in between received tokens so you can verify it streams per token correctly. Of course other clients don't have this issue and will behave properly.

## Tracing
We are tracing performance metrics using tracing, tracing-opentelemetry and opentelemetry-otlp crates.

Let's say you are running a Jaeger instance locally, you can run the following command to start it:
```bash
docker run --rm --name jaeger \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 14250:14250 \
  -p 14268:14268 \
  -p 14269:14269 \
  -p 9411:9411 \
  jaegertracing/all-in-one:1.51

```

To enable tracing, set the `otlp_endpoint` key in the `[daemon]` section of the TOML config file to the endpoint of your OpenTelemetry collector.

## Roadmap
- Support for more OpenAI API compatible endpoints (speech, vision, etc). Triton Inference Server currently supports Whisper, Stable Diffusion, and others but AI Router needs to be extended to support them. Once implemented will also work for OpenAI and other providers/implementations.
