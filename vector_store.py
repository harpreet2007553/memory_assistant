from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

dimension = 512  # fixed for clip-vit-base-patch32
index = faiss.IndexFlatL2(dimension)
docstore = InMemoryDocstore({})
vector_store = FAISS(
    embedding_function=None,  # no need, since you precompute
    index=index,
    docstore=docstore,
    index_to_docstore_id={}
)
vector_store.save_local("faiss_index")