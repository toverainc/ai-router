# AI Router - AI Model Flexibility

Provide [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/tensorrtllm_backend)
with an OpenAI-compatible API. This allows you to integrate with [langchain](https://github.com/langchain-ai/langchain) and other OpenAI compatible clients.

Also supports OpenAI and other OpenAI API compliant backends.

## Overall Design/Functionality/Architecture/Goals
AI Router presents an OpenAI API compatible HTTP interface to clients. AI Router can then take input from OpenAI API clients across any supported [OpenAI API endpoint](https://platform.openai.com/docs/api-reference/) (functionality) and route to any number of backends with translation across protocol, model naming, and prompt formatting. Currently supports OpenAI API compatible backends or Nvidia Triton Inference server via gRPC.

- Support multiple API keys for clients (or none/anything).
- Support arbitrary model names for client requests with mapping to actual backend model name.
- Clients can send input as though they're talking to OpenAI - AI Router will handle prompt formatting for open models across backends via a prompt format repo (or in this repo) with popular open source prompt formats with the ability to add/define your own. Goal is to make the most use/flexibility with backend open models within the limits of the OpenAI API for clients that have no concept of this or currently require manual prompt formatting with every request. What a pain!
- Support defining different API keys for OpenAI compatible backends.
- Support mapping between API paths to backends on different paths, ports, IPs, etc.
- Support defining default backends/functions/models so for example OpenAI API compatible clients can run with default model names and automatically map to backend model name.
- Extremely high performance.
- Low system resource utilization.
- Streaming support with fixups with Triton Inference Server (required ATM).
- Support mix of client stream request/stream to backend. Just handle it and buffer/stream internally.
- More to come!

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

Make sure your [easy-triton](https://github.com/toverainc/easy-triton) instance is running. As long as you're using default settings continue with:

```bash
./utils.sh run-local
```

...or just use OpenAI/OpenAI compliant backend.

### Test

Your OpenAI endpoint is available at `http://[your_host]:3000/v1`

There are example clients in `client/`

Example with streaming:

`python3 client/openai_chatcompletion_client.py -S`

When using Triton especially tokens stream so quickly python/TTY can't keep up so it will appear as though the stream outputs by sentence/paragraph. Specify `-n` for the client to insert newlines in between received tokens so you can verify it streams per token correctly. Of course other clients don't have this issue and will behave properly.

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

To enable tracing, set the `OTLP_ENDPOINT` environment variable or `--otlp-endpoint` command line
argument to the endpoint of your OpenTelemetry collector.
```bash
OTLP_ENDPOINT=http://localhost:4317 cargo run --release
```

## References
- [cria](https://github.com/AmineDiro/cria)
