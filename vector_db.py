from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct, Distance
from tensorflow import keras
from transformers import AutoImageProcessor, ResNetModel
import numpy as np
import torch
from tqdm import tqdm  # Progress bar for monitoring the insertion process
import base64
import io
from PIL import Image

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Define the collection name and vector size based on ResNet output
collection_name = "mnist_28x28_image"
vector_size = 2048  # ResNet50 typically outputs 2048-dimensional embeddings

# Create the collection if it doesn't exist
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()


# Preprocess images: Normalize and ensure correct shape
def preprocess_images(imgs):
    # Normalize images to the range [0, 1] and ensure single-channel format
    imgs = imgs / 255.0
    if len(imgs.shape) == 3:  # If images are in (N, 28, 28) shape
        imgs = np.expand_dims(imgs, axis=-1)  # Add the channel dimension: (N, 28, 28, 1)
    return imgs


# Apply preprocessing to training images
train_images = preprocess_images(train_images)

# Load the pre-trained ResNet model and processor, adjusted for RGB input
processor = AutoImageProcessor.from_pretrained(
    "microsoft/resnet-50",
    image_mean=[0.485, 0.456, 0.406],  # Standard mean values for RGB images
    image_std=[0.229, 0.224, 0.225],  # Standard std values for RGB images
)

# Use ResNetModel to access feature embeddings
model = ResNetModel.from_pretrained("microsoft/resnet-50")
model.eval()  # Set the model to evaluation mode


# Function to extract embeddings
def get_image_embedding(image):
    # Ensure the image has the correct shape (28, 28) -> (28, 28, 1) -> (28, 28, 3)
    image = np.repeat(image, 3, axis=-1)  # Repeat the single channel 3 times to make it compatible with ResNet
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract the pooled output from the model as embeddings
    embeddings = outputs.pooler_output.numpy().flatten()
    return embeddings


# Function to convert an image to a base64 string
def encode_image_to_base64(image):
    pil_image = Image.fromarray((image * 255).astype(np.uint8).reshape(28, 28))
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


# Prepare points to insert into Qdrant
points = []

# Loop through all training images and extract embeddings
for idx, (image, label) in tqdm(enumerate(zip(train_images, train_labels)), total=len(train_images),
                                desc="Processing MNIST images"):
    embedding = get_image_embedding(image)
    img_base64 = encode_image_to_base64(image)  # Encode the image to base64

    # Create point with metadata including the base64 encoded image
    points.append(
        PointStruct(
            id=idx,  # Use a unique ID for each image
            vector=embedding.tolist(),  # Convert the embedding to a list format
            payload={"label": int(label), "image_base64": img_base64}  # Store the label and the image as metadata
        )
    )

    # Insert in batches to avoid memory issues
    if len(points) >= 1000:  # Adjust batch size as needed
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        points = []  # Clear the list after insertion

# Insert any remaining points
if points:
    client.upsert(
        collection_name=collection_name,
        points=points
    )

print(f"Inserted all {len(train_images)} training images into Qdrant collection '{collection_name}'.")


# for audio embedding and music recommendation system
# https://www.e2enetworks.com/blog/audio-driven-search-leveraging-vector-databases-for-audio-information-retrieval
