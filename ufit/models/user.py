from sqlalchemy import Column, Integer, String, Enum
from ufit.models.time_base import TimeBase
from ufit.database.database import Base

import enum

class Gender(enum.Enum):
    MAN = "MAN"
    WOMAN = "WOMAN"

class Role(enum.Enum):
    USER = "USER"
    ADMIN = "ADMIN"

class User(Base,TimeBase):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    email = Column(String(50), unique=True, nullable=False)
    password = Column(String, nullable=False)
    age = Column(Integer, nullable=False)
    family = Column(Integer, nullable=False)
    gender = Column(Enum(Gender), nullable=False)
    role = Column(Enum(Role), nullable=False)
    rate_plan_id = Column(String, nullable=False)
    