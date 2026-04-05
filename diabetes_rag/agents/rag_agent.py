from rag.retriever import retrieve
from rag.pubmed import fetch_pubmed_abstracts
from core.llm import generate


def rag_agent(query, history):
    return rag_agent_batch([query], [history])[0]


def rag_agent_batch(queries, histories):

    responses = []

    for q, hist in zip(queries, histories):

        # Retrieve from vector DB
        docs = retrieve(q)

        # Handle empty retrieval
        if len(docs) == 0:
            responses.append("No relevant medical context found. Try rephrasing your question.")
            continue

        # Fetch PubMed context
        pubmed_context = fetch_pubmed_abstracts(q)

        # Combine context
        context = "\n".join(docs) + "\n" + pubmed_context

        # Prompt
        prompt = f"""
You are a medical assistant for diabetes care.

Context:
{context}

Question:
{q}

Answer clearly and avoid repetition.
"""

        # Generate response
        output = generate(prompt)[0]

        responses.append(output)

    return responses