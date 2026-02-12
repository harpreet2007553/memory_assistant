from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from sklearn.metrics.pairwise import cosine_similarity
import os
import base64
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from relate import relate
# from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv
import requests
import random
import datetime
from config import TEST_IMG1, TEST_IMG2
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
load_dotenv()
import uuid

run_id = str(uuid.uuid4())[:8]

## set up the environment
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
        name="my_collection",
        embedding_function=None,
        metadata={"hnsw:space": "cosine"}
    )
def setup_environment():
    # Initialize ChromaDB client and collection
    

    # Initialize CLIP model + processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    return client, collection, clip_model, clip_processor

# Run only if executed directly

client, collection, clip_model, clip_processor = setup_environment()


def embed_image(image_data):
    """Embed image using CLIP"""
    if isinstance(image_data, str):  # If path
        image = Image.open(image_data).convert("RGB")
    else:  # If PIL Image
        image = image_data
    
    inputs=clip_processor(images=image,return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)

        # If outputs is already a tensor (newer transformers versions)
        if isinstance(outputs, torch.Tensor):
            features = outputs
        else:
            # If it's a BaseModelOutputWithPooling, extract the pooled output
            features = outputs.pooler_output

        # Normalize embeddings
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze().numpy()
    
def embed_text(text):
    """Embed text using CLIP."""
    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )
    with torch.no_grad():
        outputs = clip_model.get_text_features(**inputs)

        # If outputs is already a tensor (newer transformers versions)
        if isinstance(outputs, torch.Tensor):
            features = outputs
        else:
            # If it's a BaseModelOutputWithPooling, extract the pooled output
            features = outputs.pooler_output

        # Normalize embeddings
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.squeeze().numpy()
    

url = "https://example.com/api/get-images"
response = requests.get(url)
# data = response.json()   # Suppose API returns {"images": ["...base64...", "...base64..."]}
data = {
    "images" : [TEST_IMG1, TEST_IMG2],
    "text": "Shubh narang have a blue color lamborghini car",
    "user_id" : "123"
}
image_data_store = {}
all_embeddings = []
all_docs = []


for img_index, img in enumerate(data["images"], start=1):
    try:
        img_bytes = base64.b64decode(img)
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
            # Create unique identifier
        image_id = f"img_{img_index}_{random.randint(1,10000000000)}"
            
            # Store image as base64 for later use with GPT-4V
        image_data_store[image_id] = img
            
            # Embed image using CLIP
        embedding = embed_image(pil_image)
        # print(embedding)
        all_embeddings.append(embedding)

            # Create document for image
        image_doc = Document(
            page_content=f"Bytes : {img}",
            metadata={"date": f'{datetime.datetime.now()}', "type": "image", "image_id": image_id, "user_id" : data['user_id']}
            )
        all_docs.append(image_doc)
    except Exception as e:
        print(f"Error processing image {img_index}: {e}")
        continue

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
if (data["text"] == ""):
    data["text"] = "Nil"

text_doc = Document(
    page_content=data["text"],
    metadata={"date": f'{datetime.datetime.now()}', "type": "text", "user_id": data['user_id']}
)

text_embedding = embed_text(text_doc.page_content)
all_embeddings.append(text_embedding)
all_docs.append(text_doc)

embeddings_array = np.array(all_embeddings)


metadatas = [doc.metadata for doc in all_docs]


if __name__ == "__main__":
    collection.add(
        ids=[doc.metadata["type"] + '_' + f'{random.randint(1, 10000000)}' for doc in all_docs],
        embeddings=[emb.tolist() for emb in all_embeddings],
        documents=[doc.page_content for doc in all_docs],
        metadatas=metadatas
    )
    print(collection.count())



