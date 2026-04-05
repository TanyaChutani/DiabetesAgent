import time

queue = []

def add_request(query, history):
    queue.append((query, history))

def process_batch(rag_agent_batch):

    global queue

    if not queue:
        return []

    queries, histories = zip(*queue)
    responses = rag_agent_batch(list(queries), list(histories))

    queue = []
    return responses
