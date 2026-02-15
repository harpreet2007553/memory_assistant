from flask import Flask, render_template, jsonify, request
from langchain_core.documents import Document
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import base64
import io
import os
from dotenv import load_dotenv
import random
import datetime
import chromadb
import uuid
from langchain_core.runnables import RunnableLambda
from prompt import prompt_template
from google import genai

app = Flask(__name__)

load_dotenv()
## set up the environment
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

client = chromadb.PersistentClient(path="./memory_assistant_bot/chroma_db")
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
    



@app.route("/store-memory", methods=["POST"])
def storing_memory():
    images = request.form.getlist("images")
    text = request.form.get("text", "Nil")
    user_id = request.form.get("user_id")

    data = {
    "images": images, "text": text, "user_id": user_id
    }
    run_id = str(uuid.uuid4())[:8]
    image_data_store = {}
    all_embeddings = []
    all_docs = []

    for img_index, img in enumerate(data["images"], start=1):
        try:
            img_bytes = base64.b64decode(img)
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            image_id = f"img_{img_index}_{random.randint(1,10000000000)}"
            image_data_store[image_id] = img

            embedding = embed_image(pil_image)
            all_embeddings.append(embedding)

            image_doc = Document(
                page_content=f"Bytes : {img}",
                metadata={
                    "date": str(datetime.datetime.now()),
                    "type": "image",
                    "id": image_id,
                    "user_id": data["user_id"]
                }
            )
            all_docs.append(image_doc)
        except Exception as e:
            print(f"Error processing image {img_index}: {e}")
            continue

    if data.get("text", "") == "":
        data["text"] = "Nil"

    text_doc = Document(
        page_content=data["text"],
        metadata={
            "date": str(datetime.datetime.now()),
            "type": "text",
            "id": "text"+run_id,
            "user_id": data["user_id"]
        }
    )

    text_embedding = embed_text(text_doc.page_content)
    all_embeddings.append(text_embedding)
    all_docs.append(text_doc)

    metadatas = [doc.metadata for doc in all_docs]

    # Store in ChromaDB
    try:
        collection.add(
            ids=[doc.metadata["id"] for doc in all_docs],
            embeddings=[emb.tolist() for emb in all_embeddings],
            documents=[doc.page_content for doc in all_docs],
            metadatas=metadatas
        )
        return jsonify({"status": "success", "count": collection.count()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    

def search(query, k=1):
    """Unified retrieval using CLIP embeddings for both text and images."""
    # Embed query using CLIP
    query_embedding = embed_text(query)
    
    # Search in unified vector store
    results_img = collection.query(query_embeddings=[query_embedding], n_results=k, where= {'type':"image"})
    results_text = collection.query(query_embeddings=[query_embedding], n_results=k, where= {'type':"text"})
    # print(results)

    img_bytes = base64.b64decode(results_img["documents"][0][0][8:])
    return img_bytes, results_text["documents"]

    
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=GOOGLE_API_KEY)

def relate(img_bytes, text):
    response = client.models.generate_content(
    model="gemini-2.5-flash",   # or gemini-2.5-flash if available
    contents=[
        {
            "role": "user",
            "parts": [
                {"text": f"Is this image related to {text}? Answer Yes or No."},
                {"inline_data": {"mime_type": "image/png", "data": img_bytes}}
            ]
        }
    ]
)

    return response.text

def llm_answer(prompt):
    llm = client.models.generate_content(
    model='gemini-2.5-flash',
    contents = prompt
    )
    return llm.text

def generateMemory(inputs: dict) -> dict:
    query = inputs["query"]
    img_bytes, text = search(query)
    # print(img_bytes)
    res = relate(img_bytes = img_bytes, text= text)

    memory = {}
    if res == 'No':
        memory = {"role": "user",
            "parts": [
                {"image": {"mime_type": "image/png", "data": img_bytes}},
            ]}
    else:
        memory = {"role": "user",
            "parts": [
                {"text": f"{text}"},
                {"inline_data": {"mime_type": "image/png", "data": img_bytes}}
            ]}
    return {'user_query': query, 'memory':memory}

@app.route("/query-answer", methods = ["GET"])
def quesry_answer():
    query = request.get_json()["query"]
    chain = RunnableLambda(generateMemory) | prompt_template | RunnableLambda(llm_answer)

    result = chain.invoke({
        "query": f'{query}',
    })
    return result

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)