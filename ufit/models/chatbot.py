from typing import Dict, Any, Optional
from datetime import datetime,timezone
from pydantic import BaseModel, Field
from bson import ObjectId


class ChatBotMessage(BaseModel):
    id: ObjectId = Field(default_factory=ObjectId, alias="_id")

    content: str
    owner: bool
    a_plan_id: Optional[int] = None
    b_plan_id: Optional[int] = None
    chat_room_id: int

    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc))

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        validate_by_name = True