import json
from dotenv import load_dotenv, find_dotenv
from collections import defaultdict
from typing import List

from langchain.schema import AIMessage, BaseMessage
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolInvocation
from langchain.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolExecutor
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.schema import HumanMessage
from schemas import AnswerQuestion, Reflection
from langchain.schema import AIMessage
from app.chains import parser_json

# Load environment variables
_ = load_dotenv(find_dotenv())

# Create an instance of the Tavily Search API wrapper
search = TavilySearchAPIWrapper()

# Create an instance of the Tavily Search Results tool with a maximum of 5 results
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)

# Create an instance of the ToolExecutor with the Tavily Search Results tool
tool_executor = ToolExecutor([tavily_tool])


def execute_tools(state: List[BaseMessage]) -> List[ToolMessage]:
    # Get the last message in the state, which should be an AIMessage with tool invocations
    tool_invocation: AIMessage = state[-1]

    # Parse the tool invocations from the AIMessage using the parser_json module
    parsed_tool_calls = parser_json.invoke(tool_invocation)

    # Initialize empty lists to store IDs and ToolInvocation objects
    ids = []
    tool_invocations = []

    # Loop through each parsed tool call
    for parsed_call in parsed_tool_calls:
        # Loop through each search query in the parsed tool call
        for query in parsed_call["args"]["search_queries"]:
            # Create a ToolInvocation object for the Tavily Search Results tool with the query as input
            tool_invocations.append(
                ToolInvocation(tool="tavily_search_results_json", tool_input=query)
            )
            # Append the ID of the tool call to the ids list
            ids.append(parsed_call["id"])

    # Execute the tool invocations in parallel using the ToolExecutor
    outputs = tool_executor.batch(tool_invocations)

    # Map each output to its corresponding ID and tool input
    outputs_map = defaultdict(dict)
    for id_, output, invocation in zip(ids, outputs, tool_invocations):
        outputs_map[id_][invocation.tool_input] = output

    # Convert the mapped outputs to ToolMessage objects
    tool_messages = []
    for id_, mapped_output in outputs_map.items():
        tool_messages.append(
            ToolMessage(content=json.dumps(mapped_output), tool_call_id=id_)
        )

    # Return the list of ToolMessage objects
    return tool_messages


if __name__ == "__main__":
    # Create a dummy HumanMessage
    human_message = HumanMessage(
        content="""Write about AI-Powered SOC / autonomous SOC problem domain, 
        list startups that do that and raised capital"""
    )

    # Create a dummy AnswerQuestion object with search queries
    answer = AnswerQuestion(
        answer="",
        reflection=Reflection(missing="", superfluous=""),
        search_queries=[
            "AI-powered SOC startups funding",
            "AI SOC problem domain specifics",
            "Technologies used by AI-powered SOC startups",
        ],
        id="call_KpYHichFFEmLitHFvEhKy1Ra",
    )

    # Call the execute_tools function with a dummy state
    raw_res = execute_tools(
        state=[
            human_message,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": AnswerQuestion.__name__,
                        "args": answer.dict(),
                        "id": "call_KpYHichFFEmLitHFvEhKy1Ra",
                    }
                ],
            ),
        ]
    )

    # Print the result
    print(raw_res)
