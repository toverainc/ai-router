from sys import exit

from openai import OpenAI
from openai._exceptions import OpenAIError
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument(
    "-S",
    "--stream",
    action="store_true",
    required=False,
    default=False,
    help="Enable streaming mode. Default is False.",
)

parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="What fun things are there to do in Chicago?",
    required=False,
    help="Input",
)

parser.add_argument(
    "-t",
    "--tokens",
    type=int,
    default=1024,
    required=False,
    help="Max tokens to generate",
)

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="Meta-Llama-3-8B-Instruct",
    required=False,
    help="Model name",
)

# Triton is faster than Python/the TTY can draw so this inserts a newline per chunk to confirm per-chunk stream
parser.add_argument(
    "-n",
    "--newline",
    action="store_true",
    default=False,
    required=False,
    help="Insert newline after each received chunk to show per-token output more clearly",
)

parser.add_argument(
    "-u",
    "--url",
    type=str,
    default="http://localhost:3000/v1",
    required=False,
    help="URL including port and version",
)

parser.add_argument(
    "-k",
    "--openai-api-key",
    type=str,
    default="test",
    required=False,
    help="API KEY",
)

FLAGS = parser.parse_args()

client = OpenAI(
    api_key=FLAGS.openai_api_key,
    base_url=FLAGS.url,
)

# We don't do prompt formatting in the proxy - yet
# Mistral Instruct
input = f'[INST] {FLAGS.input} [/INST]'

start_time = time.time()

try:
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": input
        }],
        model=FLAGS.model,
        max_tokens=FLAGS.tokens,
        stream=FLAGS.stream
    )
except OpenAIError as e:
    print(f"chat completion error: {type(e).__name__}: {e}")
    exit(1)

output_start_time = None
tokens = 0
if FLAGS.stream:
    for chunk in chat_completion:
        if output_start_time is None:
            output_start_time = time.time()
        tokens += 1
        if chunk.choices[0].delta.content is not None and len(chunk.choices[0].delta.content) > 0:
            if FLAGS.newline:
                print(f'{chunk.choices[0].delta.content}\n', end="")
            else:
                print(chunk.choices[0].delta.content, end="")
    print()
    print("****************")
else:
    print("Chat completion results:")
    print(chat_completion.choices[0].message.content)
    print(f"Usage statistics: {chat_completion.usage}")

timing = time.time()
response_time = timing - start_time
response_time_ms = response_time * 1000

# Tokens per second
# This may seem impossible without getting direct access to token output and having the tokenizer locally
# BUT in this case Triton outputs incrementally on a per token basis so we get it handled for us
tps = tokens / response_time

# print the time delay
if FLAGS.stream:
    first_response_time = output_start_time - start_time
    first_response_time = first_response_time * 1000
    # TODO: Currently broken
    print(f"Start of response: {first_response_time:.2f} ms after request")
    print(f"Total tokens: {tokens}")
    print(f"Tokens per second: {tps:.2f}")
print(f"Full response received {response_time_ms:.2f} ms after request")
