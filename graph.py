import torch
# import os
# from langgraph.graph import StateGraph
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI


# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# VECTOR_DIM = 768  # embedding size

# embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
# llm_client = OpenAI(api_key=OPENAI_API_KEY)

# class SimpleVectorStore:
#     def __init__(self):
#         self.vectors = []
#         self.payloads = []

#     def add_documents(self, docs):
#         for doc in docs:
#             emb = embedding_model.encode(doc).tolist()
#             self.vectors.append(emb)
#             self.payloads.append({"text": doc})

#     def search(self, query, top_k=3):
#         import numpy as np
#         q_emb = embedding_model.encode(query)
#         vectors = np.array(self.vectors)
#         # cosine similarity
#         sims = vectors @ q_emb / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_emb) + 1e-10)
#         top_idx = sims.argsort()[-top_k:][::-1]
#         return [self.payloads[i] for i in top_idx]

# # Initialize store and add example docs
# vector_store = SimpleVectorStore()
# example_docs = [
#     "Diabetes treatment guideline 2025.",
#     "Blood sugar management overview.",
#     "Lifestyle modifications for type 2 diabetes."
# ]
# vector_store.add_documents(example_docs)

# def planner(state):
#     query = state.get("query")
#     if not query:
#         raise ValueError("Missing 'query'")
#     plan_type = "treatment" if "treat" in query.lower() else "general"
#     return {**state, "plan": plan_type}

# def retriever(state):
#     query = state.get("query")
#     if not query:
#         return {**state, "documents": []}
#     hits = vector_store.search(query, top_k=3)
#     docs = [h["text"] for h in hits]
#     return {**state, "documents": docs}

# def reranker(state):
#     docs = state.get("documents", [])
#     # simple heuristic: longer documents first
#     ranked_docs = sorted(docs, key=lambda x: len(x), reverse=True)
#     return {**state, "documents": ranked_docs}

# def reasoner(state):
#     query = state.get("query")
#     docs = state.get("documents", [])
#     context = "\n".join(docs)

#     prompt = f"""
# You are a medical assistant specialized in diabetes.
# Answer the user question based on the following evidence:


#     response = llm_client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2
#     )

#     answer = response.choices[0].message.content
#     confidence = 0.9  # placeholder

#     return {**state, "answer": answer, "confidence": confidence, "final": {"answer": answer, "confidence": confidence}}

# builder = StateGraph(dict)
# builder.add_node("planner", planner)
# builder.add_node("retriever", retriever)
# builder.add_node("reranker", reranker)
# builder.add_node("reasoner", reasoner)

# builder.add_edge("planner", "retriever")
# builder.add_edge("retriever", "reranker")
# builder.add_edge("reranker", "reasoner")

# builder.set_entry_point("planner")
# graph = builder.compile()

# if __name__ == "__main__":
#     state = {"query": "What is the recommended treatment for type 2 diabetes?"}
#     result = graph.run(state)
#     print(result)

import numpy as np
from langgraph.graph import StateGraph
from sentence_transformers import SentenceTransformer
from transformers import pipeline

embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=0
)

class SimpleVectorStore:
    def __init__(self):
        self.vectors = []
        self.texts = []

    def add_documents(self, docs):
        for doc in docs:
            emb = embedding_model.encode(doc, normalize_embeddings=True)
            self.vectors.append(emb)
            self.texts.append(doc)

    def search(self, query, top_k=3):
        q_emb = embedding_model.encode(query, normalize_embeddings=True)

        sims = np.dot(self.vectors, q_emb)
        top_idx = np.argsort(sims)[-top_k:][::-1]

        return [self.texts[i] for i in top_idx]

vector_store = SimpleVectorStore()

docs = [
    "Type 2 diabetes is treated with lifestyle modification and metformin.",
    "Insulin therapy is used when blood glucose is not controlled.",
    "Diet and exercise are essential in diabetes management.",
    "HbA1c is used to monitor long-term glucose control.",
    "GLP-1 receptor agonists help reduce blood sugar and weight."
]

vector_store.add_documents(docs)

def planner(state):
    query = state["query"]
    plan = "treatment" if "treat" in query.lower() else "general"
    return {**state, "plan": plan}

def retriever(state):
    docs = vector_store.search(state["query"])
    return {**state, "documents": docs}

def reranker(state):
    docs = state["documents"]
    return {**state, "documents": docs}

def reasoner(state):
    query = state["query"]
    context = "\n".join(state["documents"])

    prompt = f"""
You are a medical assistant.

Context:
{context}

Question: {query}

Give a concise, medically accurate answer:
"""

    output = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False,
        temperature=0.3
    )[0]["generated_text"]

    # Remove prompt from output
    answer = output.replace(prompt, "").strip()

    return {
        **state,
        "answer": answer,
        "confidence": 0.85,
        "final": {"answer": answer, "confidence": 0.85}
    }

builder = StateGraph(dict)

builder.add_node("planner", planner)
builder.add_node("retriever", retriever)
builder.add_node("reranker", reranker)
builder.add_node("reasoner", reasoner)

builder.add_edge("planner", "retriever")
builder.add_edge("retriever", "reranker")
builder.add_edge("reranker", "reasoner")

builder.set_entry_point("planner")
graph = builder.compile()

if __name__ == "__main__":
    query = "How to treat type 2 diabetes?"
    result = graph.run({"query": query})

    print("\nFINAL OUTPUT:\n")
    print(result["final"]["answer"])
