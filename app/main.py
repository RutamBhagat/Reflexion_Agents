from dotenv import load_dotenv, find_dotenv
from typing import List

from langchain.schema import BaseMessage
from langchain_core.messages import ToolMessage
from langgraph.graph import END, MessageGraph
from app.chains import first_responder, revisor
from app.tool_executor import execute_tools


_ = load_dotenv(find_dotenv())

MAX_ITERATIONS = 2

builder = MessageGraph()
builder.add_node("draft", first_responder)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revisor)

builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")


def event_loop(state: List[BaseMessage]) -> str:
    count_tool_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tool_visits
    if num_iterations > MAX_ITERATIONS:
        return END
    return "execute_tools"


builder.add_conditional_edges("revise", event_loop)
builder.set_entry_point("draft")
graph = builder.compile()
print(graph.get_graph().draw_ascii())
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")

if __name__ == "__main__":
    print("Hello Reflexion")
    res = graph.invoke(
        """Write about AI-Powered SOC / autonomous SOC problem domain, 
        list startups that do that and raised capital"""
    )
    print(res[-1].tool_calls[0]["args"]["answer"])
