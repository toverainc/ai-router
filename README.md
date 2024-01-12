# AI Router - OpenAI-compatible API for Nvidia Triton Inference Server

Provide [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/tensorrtllm_backend)
with an OpenAI-compatible API. This allows you to integrate with [langchain](https://github.com/langchain-ai/langchain) and other OpenAI compatible clients.

## Get started

Clone this repo.

### Build

`./utils.sh build`

### Start

Make sure your [easy-triton](https://github.com/toverainc/easy-triton) instance is running. As long as you're using default settings continue with:

`./utils.sh run-local`

### Test

Your OpenAI endpoint is available at `http://[your_host]:3000/v1`

There are example clients in `client/`

Example with streaming:

`python3 client/openai_chatcompletion_client.py -S`

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
