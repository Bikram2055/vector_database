from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance
import tensorflow as tf
from tensorflow import keras
# from transformers import AutoImageProcessor, ResNetModel
from transformers import AutoTokenizer, DPRContextEncoder
import numpy as np
import torch

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Create the collection if it doesn't exist
collection_name = "my_text_collection1"
vector_size = 768  # Adjust this size to match the actual size of the embeddings from embedding_model
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

tokenizer = AutoTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Example chunk of text
chunk = "Climate change affects marine life by disrupting ecosystems and altering habitats."

inputs = tokenizer(chunk, return_tensors='pt')
chunk_embedding = context_encoder(**inputs).pooler_output.detach().numpy()[0]  # Convert to NumPy array

# Metadata including the actual text chunk
metadata = {
    "id": 1,  # Unique identifier for the chunk
    "source": "Environmental Article",  # Example metadata
    "date": "2023-09-06",
    "text": chunk  # Include the actual text chunk in the metadata
}

# Prepare the data point with the vector and metadata
point = PointStruct(
    id=metadata["id"],  # Unique ID for the chunk
    vector=chunk_embedding.tolist(),  # Convert NumPy array to list
    payload=metadata  # Add metadata with the actual text chunk included
)

# Insert the vector with metadata into the collection
client.upsert(collection_name=collection_name, points=[point])

print("Vector and metadata, including the text chunk, have been successfully added to the Qdrant database.")
