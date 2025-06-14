import os
from dotenv import load_dotenv

from functools import lru_cache
from pymongo import MongoClient
from pymongo.database import Database
from sqlalchemy import create_engine, Table, MetaData
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from fastapi import Depends

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# SQLAlchemy 기본 설정

Base = declarative_base()

# PostgreSQL 연결 URL
POSTGRES_URL = os.getenv("PGVECTOR_CONNECTIONS_STRING")
# echo=True 로 SQL 로그 보기
engine = create_engine(POSTGRES_URL, echo=True)

# 세션 팩토리
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# PostgreSQL 세션
def get_db() -> Session: # type: ignore
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# MongoDB 기본 설정

MONGO_URL = os.getenv("MONGO_URL")

# 싱글톤 클라이언트 관리
@lru_cache()
def get_mongo_client() -> MongoClient:
    return MongoClient(MONGO_URL)

# Dependency: 실제 Database 객체
def get_mongo_db(
    client: MongoClient = Depends(get_mongo_client),
) -> Database:
    return client["ufit_db"]
