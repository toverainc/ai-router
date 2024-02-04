from openai import OpenAI
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="BAAI/bge-large-en-v1.5",
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

parser.add_argument(
    "-c",
    "--compare",
    type=str,
    default=None,
    required=True,
    help="Compare URL for comparison",
)

parser.add_argument(
    "-k",
    "--openai-api-key",
    type=str,
    default="test",
    required=False,
    help="API KEY",
)

parser.add_argument(
    "-i",
    "--input",
    type=str,
    default="This is the ai-router model comparison test",
    required=False,
    help="Model input text",
)

parser.add_argument(
    "-b",
    "--batch",
    type=int,
    default=1,
    required=False,
    help="Batch size",
)

parser.add_argument(
    "-v",
    "--verbose",
    type=bool,
    default=False,
    required=False,
    help="Show comparison outputs",
)

FLAGS = parser.parse_args()

ref_client = OpenAI(
    api_key=FLAGS.openai_api_key,
    base_url=FLAGS.url,
)

compare_client = OpenAI(
    api_key=FLAGS.openai_api_key,
    base_url=FLAGS.compare,
)

def get_embedding(client, text_to_embed):
	# Embed a line of text
	response = client.embeddings.create(
    	model= FLAGS.model,
    	input=[text_to_embed]
	)
	#print(str(response))
	# Extract the AI output embedding as a list of floats
	embedding = response.data[0].embedding
    
	return embedding

ref_embedding = get_embedding(ref_client,FLAGS.input)
compare_embedding = get_embedding(compare_client, FLAGS.input)

if ref_embedding == compare_embedding:
	print('Embeddings match')
else:
    print('Embeddings match fail')