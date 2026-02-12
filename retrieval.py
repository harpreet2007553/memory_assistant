from config import TEST_IMG1, TEST_IMG2
import base64
from PIL import Image
import io
from memory_storing import embed_text, collection

def retrieve_multimodal(query, k=1):
    """Unified retrieval using CLIP embeddings for both text and images."""
    # Embed query using CLIP
    query_embedding = embed_text(query)
    
    # Search in unified vector store
    results_img = collection.query(query_embeddings=[query_embedding], n_results=k, where= {'type':"image"})
    results_text = collection.query(query_embeddings=[query_embedding], n_results=k, where= {'type':"text"})
    # print(results)

    img_bytes = base64.b64decode(results_img["documents"][0][0][8:])
    return img_bytes, results_text["documents"]

# print(retrieve_multimodal("mickey mouse in black and red"))
# print(collection.get(include=['embeddings']))
