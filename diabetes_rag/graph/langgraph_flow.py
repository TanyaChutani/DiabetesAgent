from langgraph.graph import StateGraph
from typing import TypedDict

from agents.router import route
from agents.rag_agent import rag_agent
from agents.glucose_agent import glucose_agent
from agents.diet_agent import diet_agent
from agents.plan_agent import plan_agent


class State(TypedDict):
    query: str
    history: list
    response: str


def run_agent(state: State):

    query = state["query"]
    history = state.get("history", [])

    intent = route(query)

    if intent == "glucose":
        response = glucose_agent(query)

    elif intent == "diet":
        response = diet_agent(query)

    elif intent == "plan":
        response = plan_agent(query)

    else:
        response = rag_agent(query, history)

    return {
        "response": response,
        "history": history + [(query, response)]
    }


workflow = StateGraph(State)
workflow.add_node("main", run_agent)
workflow.set_entry_point("main")

app = workflow.compile()