from typing import Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from bson import ObjectId


class RatePlan(BaseModel):

    plan_name: str
    summary: str
    monthly_fee: int
    discount_fee: int
    data_allowance: Optional[str] = None
    voice_allowance: Optional[str] = None
    sms_allowance: Optional[str] = None
    basic_benefit: Dict[str, Any]
    is_enabled: bool
    is_deleted: bool

    class Config:
        populate_by_name = True
        json_encoders = {ObjectId: str}
        arbitrary_types_allowed = True
        extra = "ignore"
