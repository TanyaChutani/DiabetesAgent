import faiss
import numpy as np
from core.embeddings import embed_batch

# 🔹 Your document store (replace with real loader if needed)
documents = [
    "Diabetes is a chronic condition characterized by high blood sugar levels.",
    "Insulin helps regulate blood glucose levels.",
    "HbA1c reflects average blood sugar over 3 months.",
    "Exercise improves insulin sensitivity.",
    "Low glycemic index foods help control glucose."
]

# 🔹 Create embeddings
embeddings = embed_batch(documents)

# 🔹 Convert to numpy float32
embeddings = np.array(embeddings).astype("float32")

# 🔹 Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)

# 🔹 Add vectors
index.add(embeddings)

print("Documents loaded:", len(documents))
print("FAISS index size:", index.ntotal)