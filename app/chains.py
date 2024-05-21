from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a viral twitter influencer grading a tweet. 
            (Note: Do not mention the original tweet and you are not allowed to actually revise the tweet yourself)
            Generate a critique and recommendations for the user's tweet
            Always provide detailed recommendations, including requests for length, virality, style, etc. 
            """
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a twitter techie influencer assistant tasked with writing excellent twitter posts.
            Generate the best twitter post possible for the user's request
            If the user provides critique, respond with a revised version of your previous attempts."""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI()

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm