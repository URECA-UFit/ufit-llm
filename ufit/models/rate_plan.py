from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from bson import ObjectId


class RatePlan(BaseModel):
    id: ObjectId = Field(default_factory=ObjectId, alias="_id")

    plan_name: str
    summary: str
    monthly_fee: int
    discount_fee: int
    data_allowance: str
    voice_allowance: str
    sms_allowance: str
    basic_benefit: Dict[str, Any]
    special_benefit: Optional[Dict[str, Any]] = None
    discount_benefit: Optional[Dict[str, Any]] = None
    is_enabled: bool
    is_deleted: bool

    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc))

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        validate_by_name = True