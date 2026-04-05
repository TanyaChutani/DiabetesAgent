from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def embed(text):
    return model.encode([text])


def embed_batch(texts):
    return model.encode(texts)
