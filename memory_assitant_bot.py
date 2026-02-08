import fitz  
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
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv
import requests
import random
import datetime
from config import TEST_IMG
import chromadb
load_dotenv()

## set up the environment
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")


client = chromadb.PersistentClient(path="chroma_db")

collection = client.get_or_create_collection("my_collection")


### initialize the Clip Model for unified embeddings
clip_model=CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

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
data = response.json()   # Suppose API returns {"images": ["...base64...", "...base64..."]}
image_data_store = {}
all_embeddings = []
all_docs = []

# print(TEST_IMG)

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
        all_embeddings.append(embedding)
            
            # Create document for image
        image_doc = Document(
            page_content=f"[Image: {image_id}]",
            metadata={"date": f'{datetime.datetime.now()}', "type": "image", "image_id": image_id, "user_id" : data['user_id']}
        )
        all_docs.append(image_doc)
    except Exception as e:
        print(f"Error processing image {img_index}: {e}")
        continue

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
if True :
    text_doc = Document(
        page_content=data["text"],
        metadata={"date": f'{datetime.datetime.now()}', "type": "text", "user_id": data['user_id']}
    )
    
    text_embedding = embed_text(text_doc.page_content)
    all_embeddings.append(text_embedding)
    all_docs.append(text_doc)

embeddings_array = np.array(all_embeddings)
# print(image_doc)




# texts = [doc.page_content for doc in all_docs]
# embeddings = [emb for emb in embeddings_array]
# metadatas = [doc.metadata for doc in all_docs]

# # Add them to the store
# vector_store.add_embeddings(
#     texts=texts,
#     embeddings=embeddings,
#     metadatas=metadatas
# )

metadatas = [doc.metadata for doc in all_docs]

collection.add(
    ids=[f"{doc.metadata["type"]}_{doc.metadata["date"]}_{doc.metadata["user_id"]}" for doc in all_docs],
    embeddings=[emb for emb in embeddings_array],
    documents=[doc.page_content for doc in all_docs],
    metadatas=metadatas
)
print("Number of vectors:", collection.count())

# print(embeddings_array, all_docs)
# print(all_docs)

