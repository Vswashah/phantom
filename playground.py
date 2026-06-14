from langgraph.graph import StateGraph, START, END

# ── 1. STATE ─────────────────────────────────────────────────
# This is the "notebook" passed between every node.
# Each node can read any key and write any key.

from typing import TypedDict

class MyState(TypedDict):
    name: str
    message: str
# ── 2. NODES ─────────────────────────────────────────────────
# A node is just a function.
# It receives the full state and returns a dict of changes.

def greet(state: MyState) -> dict:
    name = state["name"]
    return {"message": f"Hello, {name}!"}

def shout(state: MyState) -> dict:
    message = state["message"]
    return {"message": message.upper()}

# ── 3. BUILD THE GRAPH ────────────────────────────────────────
graph = StateGraph(MyState)

# Add nodes — (name, function)
graph.add_node("greet", greet)
graph.add_node("shout", shout)

# Add edges — tell it the order
graph.add_edge(START, "greet")
graph.add_edge("greet", "shout")
graph.add_edge("shout", END)

# Compile — locks the graph, makes it runnable
app = graph.compile()

# ── 4. RUN IT ─────────────────────────────────────────────────
result = app.invoke({"name": "Vishwaa"})
print(result)