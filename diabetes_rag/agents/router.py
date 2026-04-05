def route(query):

    query = query.lower()

    # glucose detection
    if any(word in query for word in ["glucose", "sugar", "reading", "levels"]):
        return "glucose"

    # diet detection
    elif any(word in query for word in ["diet", "food", "eat", "meal", "nutrition"]):
        return "diet"

    #  plan ONLY if explicitly asked
    elif any(word in query for word in ["plan", "routine", "schedule"]):
        return "plan"

    # default → RAG (MOST IMPORTANT)
    else:
        return "rag"