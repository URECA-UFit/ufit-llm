import os
from dotenv import load_dotenv

from functools import lru_cache
from pymongo import MongoClient
from pymongo.database import Database
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from fastapi import Depends
from typing import Optional, List, Dict

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

Base = declarative_base()

POSTGRES_URL = os.getenv("PGVECTOR_CONNECTIONS_STRING")

engine = create_engine(POSTGRES_URL, echo=True)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

def get_db() -> Session: # type: ignore
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

MONGO_URL = os.getenv("MONGO_URI")

@lru_cache()
def get_mongo_client() -> MongoClient:
    return MongoClient(MONGO_URL)

def get_mongo_db(
    client: MongoClient = Depends(get_mongo_client),
) -> Database:
    return client["ufit"]

def save_chat_bot_message(collection, content: str, owner: bool, chat_room_id: int, recommend_plan: Optional[List[Dict[str, str]]] = None):
    doc = {
        "content": content,
        "owner": owner,
        "chat_room_id": chat_room_id,
    }
    if recommend_plan:
        doc["recommend_plan"] = recommend_plan

    result = collection.insert_one(doc)
    return result.inserted_id
