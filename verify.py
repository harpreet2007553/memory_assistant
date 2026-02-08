from config import TEST_IMG
import base64
from PIL import Image
import io
from memory_assitant_bot import embed_image, embed_text, collection

# img_bytes = base64.b64decode(TEST_IMG)
# with open("test.png", "wb") as f:
#     f.write(img_bytes)

# pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
# pil_image.save("test2.png")
# embedding = embed_image(pil_image)


def retrieve_multimodal(query, k=2):
    """Unified retrieval using CLIP embeddings for both text and images."""
    # Embed query using CLIP
    query_embedding = embed_text(query)
    
    # Search in unified vector store
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    print(results)

    
    return results

print(retrieve_multimodal("mblack and red"))