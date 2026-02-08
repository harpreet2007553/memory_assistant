
# for img_index, img in enumerate(data["images"], start=1):
#     try:
#         img_bytes = base64.b64decode(img)
#         pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            
#             # Create unique identifier
#         image_id = f"img_{img_index}_{random.randint(1,10000000000)}"
            
#             # Store image as base64 for later use with GPT-4V
#         image_data_store[image_id] = img
            
#             # Embed image using CLIP
#         embedding = embed_image(pil_image)
#         all_embeddings.append(embedding)
            
#             # Create document for image
#         image_doc = Document(
#             page_content=f"[Image: {image_id}]",
#             metadata={"date": datetime.datetime.now(), "type": "image", "image_id": image_id, "user_id" : data['user_id']}
#         )
#         all_docs.append(image_doc)
#     except Exception as e:
#         print(f"Error processing image {img_index}: {e}")
#         continue

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# if True :
#     text_doc = Document(
#         page_content=data["text"],
#         metadata={"date": datetime.datetime.now(), "type": "text", "user_id": data['user_id']}
#     )
    
#     text_embedding = embed_text(text_doc.page_content)
#     all_embeddings.append(text_embedding)
#     all_docs.append(text_doc)

# embeddings_array = np.array(all_embeddings)
# print(image_doc)
# vector_store = FAISS.from_embeddings(
#     text_embeddings=[(doc.page_content, emb) for doc, emb in zip(all_docs, embeddings_array)],
#     embedding=None,  # We're using precomputed embeddings
#     metadatas=[doc.metadata for doc in all_docs]
# )
# print(embeddings_array, all_docs)