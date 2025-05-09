import os
from dotenv import load_dotenv

load_dotenv()

AZURE_AI_SERVICES_ENDPOINT = os.getenv("AZURE_AI_SERVICES_ENDPOINT")
AZURE_AI_SERVICES_API_VERSION = os.getenv("AZURE_AI_SERVICES_API_VERSION")
AZURE_AI_SERVICES_API_KEY = os.getenv("AZURE_AI_SERVICES_API_KEY")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_CHAT_API_VERSION = os.getenv("AZURE_OPENAI_CHAT_API_VERSION")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDINGS_API_VERSION = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION")