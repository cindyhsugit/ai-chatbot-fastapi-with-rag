from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Define the state structure
class State(TypedDict):
    message: str

# Define nodes
def node_1(state: State):
    print("Executing Node 1")
    return {"message": state["message"] + " -> Node 1"}

def node_2(state: State):
    print("Executing Node 2")
    return {"message": state["message"] + " -> Node 2"}

# Build the graph
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

# Define edges
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

# Compile the graph
graph = builder.compile()

# Execute the graph
result = graph.invoke({"message": "Start"})
print("Final Result:", result)