import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_HUB_TOKEN"],
)

def call_llm(messages):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content
