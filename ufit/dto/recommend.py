from pydantic import BaseModel
from typing import List
from enum import Enum


class AnswerType(str, Enum):
    RECOMMEND = "RECOMMEND"
    GENERAL = "GENERAL"


class RecommendRequest(BaseModel):
    userId: int
    content: str
    chatRoomId: int


class PlanDTO(BaseModel):
    planId: str
    name: str


class RecommendResponse(BaseModel):
    messageId: str
    answer: str
    recommendPlans: List[PlanDTO]
    answerType: AnswerType
