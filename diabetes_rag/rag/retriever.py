import numpy as np
from core.embeddings import embed
from rag.vector_store import index, documents


def retrieve(query, top_k=5):

    # Safety check
    if documents is None or len(documents) == 0:
        print("No documents found in vector DB")
        return []

    # 🔹 Embed query
    query_vec = embed(query)

    # 🔹 FAISS search
    D, I = index.search(query_vec, top_k)

    print("DEBUG indices:", I)
    print("DEBUG docs len:", len(documents))

    # 🔹 Safe filtering
    valid_docs = []

    for idx in I[0]:
        if idx == -1:
            continue
        if 0 <= idx < len(documents):
            valid_docs.append(documents[idx])

    return valid_docs
