def format_history(history):
    return "\n".join([f"User: {q}\nBot: {r}" for q, r in history])
