import datetime
from langchain.output_parsers import JsonOutputToolsParser, PydanticToolsParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from app.schemas import AnswerQuestion, ReviseAnswer

_ = load_dotenv(find_dotenv())

actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert researcher.
            Current Time: {time}
            1. {first_instruction}
            2. Reflect and critique your answer. Be severe to maximize improvement.
            3. Recommend search queries to research information and improve your answer
            (NOTE: You MUST provide the search queries at all costs)""",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

actor_prompt_template = actor_prompt_template.partial(
    time=lambda: datetime.datetime.now().isoformat()
)
llm = ChatOpenAI()
parser_json = JsonOutputToolsParser(return_id=True)
parser_pydantic = PydanticToolsParser(tools=[AnswerQuestion])

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

first_responder = first_responder_prompt_template | llm.bind_tools(
    tools=[AnswerQuestion], tool_choice="AnswerQuestion"
)

human_message = HumanMessage(
    content="""Write about AI-Powered SOC / autonomous SOC problem domain
    List startups that do that and raised capital"""
)

chain = (
    first_responder_prompt_template
    | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
    | parser_pydantic
)

revise_instructions = """Revise your previous answer using the new information.
You should use the previous critique to add important information to your answer.
You MUST include numerical citations in your revised answer to ensure it can be verified.
Add a "References" section to the bottom of your answer (which does not count towards the word limit).
In the form of:
- [1] https://example.com
- [2] https://example.com
You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words"""


revisor_prompt_template = actor_prompt_template.partial(
    first_instruction=revise_instructions
)
revisor = revisor_prompt_template | llm.bind_tools(
    tools=[ReviseAnswer], tool_choice="ReviseAnswer"
)


if __name__ == "__main__":
    res = chain.invoke(input={"messages": [human_message]})

    for inst in res[0]:
        print(inst[0], end=": ")
        print(inst[1])
        print()
        print()
        print()
