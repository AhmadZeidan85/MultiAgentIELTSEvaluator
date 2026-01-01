import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai"

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_HUB_TOKEN"]
)
