from openai import OpenAI
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
    default="Mistral-7B-Instruct-v0.2",
    required=False,
    help="Model name",
)

parser.add_argument(
    "-u",
    "--url",
    type=str,
    default="http://localhost:3000/v1",
    required=False,
    help="URL including port and version",
)

FLAGS = parser.parse_args()

# We don't really use a key - yet
openai_api_key = "test"

client = OpenAI(
    api_key=openai_api_key,
    base_url=FLAGS.url,
)

# We don't do prompt formatting in the proxy - yet
# Mistral Instruct
input = f'<s>[INST] {FLAGS.input} [/INST]'

start_time = time.time()
chat_completion = client.chat.completions.create(
    messages=[{
        "role": "system",
        "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    },
    {
        "role": "user",
        "content": input
    }],
    model=FLAGS.model,
    max_tokens=FLAGS.tokens,
    stream=FLAGS.stream
)

if FLAGS.stream:
    for chunk in chat_completion:
        print(chunk.choices[0].delta.content)
        print("****************")
else:
    print("Chat completion results:")
    print(chat_completion.choices[0].message.content)

response_time = time.time() - start_time

# print the time delay
print(f"Full response received {response_time:.2f} seconds after request")