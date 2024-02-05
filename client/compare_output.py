import argparse
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Triton
import tritonclient.grpc as grpcclient

# OpenAI
from openai import OpenAI

parser = argparse.ArgumentParser()

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

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="BAAI/bge-large-en-v1.5",
    required=False,
    help="Model name",
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
    "-t",
    "--triton",
    type=str,
    default="localhost",
    required=False,
    help="Triton host",
)

FLAGS = parser.parse_args()

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)
model = AutoModel.from_pretrained(FLAGS.model)
model.eval()

# Tokenize sentences
encoded_input = tokenizer(FLAGS.input, padding=True, truncation=True, return_tensors='pt')

def triton(text):
    triton_client = grpcclient.InferenceServerClient(url=f'{FLAGS.triton}:8001')
    triton_encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='np')

    input_ids_shape = triton_encoded_input.get('input_ids').shape
    attention_mask_shape = triton_encoded_input.get('attention_mask').shape
    token_type_ids_shape = triton_encoded_input.get('token_type_ids').shape

    # Infer
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput("input_ids", input_ids_shape, "INT64"))
    inputs.append(grpcclient.InferInput("attention_mask", attention_mask_shape, "INT64"))
    inputs.append(grpcclient.InferInput("token_type_ids", token_type_ids_shape, "INT64"))

    # Initialize the data
    inputs[0].set_data_from_numpy(triton_encoded_input.get('input_ids'))
    inputs[1].set_data_from_numpy(triton_encoded_input.get('attention_mask'))
    inputs[2].set_data_from_numpy(triton_encoded_input.get('token_type_ids'))

    outputs.append(grpcclient.InferRequestedOutput("embedding"))

    # Test with outputs
    results = triton_client.infer(
        model_name="embedding",
        inputs=inputs,
        outputs=outputs,
    )

    # Get the output arrays from the results
    embedding = results.as_numpy("embedding")
    return embedding

def triton_ensemble(text):
    body = {"text": text}
    response = requests.post(f"http://{FLAGS.triton}:8000/v2/models/bge-large-en-v1.5/generate", json=body)
    response = response.json()
    embedding = response.get("embedding")
    embedding = np.array([embedding])
    return embedding

def openai_embedding(text):
    client = OpenAI(
        api_key=FLAGS.openai_api_key,
        base_url=FLAGS.url,
    )

    response = client.embeddings.create(
    	model= FLAGS.model,
    	input=[text]
	)

    embedding = response.data[0].embedding
    embedding = np.array([embedding])
    return embedding

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    # Perform pooling. In this case, cls pooling.
    sentence_embeddings = model_output[0][:, 0]

# normalize embeddings
sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
sentence_embeddings = sentence_embeddings.numpy()

embeddings = triton(FLAGS.input)
ensemble_embeddings = triton_ensemble(FLAGS.input)

openai_embeddings = openai_embedding(FLAGS.input)

similarity = cosine_similarity(sentence_embeddings, embeddings)[0]
print(f'Triton similarity: {similarity[0]}')

similarity = cosine_similarity(sentence_embeddings, ensemble_embeddings)[0]
print(f'Triton ensemble similarity: {similarity[0]}')

similarity = cosine_similarity(sentence_embeddings, openai_embeddings)[0]
print(f'OpenAI similarity: {similarity[0]}')