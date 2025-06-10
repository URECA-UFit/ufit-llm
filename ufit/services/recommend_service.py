import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import PGVector

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

PGVECTOR_CONNECTIONS_STRING = os.getenv("PGVECTOR_CONNECTIONS_STRING")
collection_name = "plans"

vectorstore = PGVector(
    embedding_function = embedding_model,
    collection_name = collection_name,
    connection_string = PGVECTOR_CONNECTIONS_STRING
)

def search_similar_plans(query: str, k: int = 2):
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

