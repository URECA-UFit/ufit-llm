import os
from dotenv import load_dotenv
from langchain_mongodb import MongoDBChatMessageHistory
from pymongo.database import Database

def get_history(session_id: str, mongo_db: Database):
    return MongoDBChatMessageHistory(
            connection_string=os.getenv("MONGO_URI"),
            session_id=session_id,
            database_name=mongo_db.name,
            collection_name="message_histories",
    )

