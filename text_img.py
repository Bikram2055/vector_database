from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import numpy as np
import base64
import io

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Collection name for storing image embeddings
collection_name = "image_search_collection"
vector_size = 512  # CLIP outputs 512-dimensional vectors

# Create a collection in Qdrant if it doesn't exist
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Function to encode images and insert them into Qdrant
def insert_image_to_qdrant(image_path, image_id, description):
    # Load the image and convert to RGB
    image = Image.open(image_path).convert("RGB")
    # Process the image with CLIP
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    # Flatten the embedding to store in Qdrant
    embedding = image_features.numpy().flatten()

    # Encode image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Insert into Qdrant
    client.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=image_id,
                vector=embedding.tolist(),
                payload={"description": description, "image_base64": img_str},
            )
        ]
    )


# Example: Insert an image of a dog
insert_image_to_qdrant("C:/Users/Msi/Desktop/child.png", 1, "A child in the park")


# Function to query an image based on text input
def query_image_by_text(query):
    # Encode the text query
    inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

    # Convert the query embedding to a list
    query_vector = text_features.numpy().flatten().tolist()

    # Search in Qdrant
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=1  # Retrieve the top result
    )

    # Display the retrieved image
    for result in search_results:
        print(f"Description: {result.payload['description']}")
        img_data = base64.b64decode(result.payload['image_base64'])
        retrieved_image = Image.open(io.BytesIO(img_data))
        retrieved_image.show()


# Example query: Searching for a dog image
query_image_by_text("how does a child look like")
