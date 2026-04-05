from sentence_transformers import SentenceTransformer
from core.config import EMBED_MODEL

embed_model = SentenceTransformer(EMBED_MODEL)
