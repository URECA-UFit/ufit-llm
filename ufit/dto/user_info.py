from pydantic import BaseModel
from typing import List
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from bson import ObjectId
from ufit.models.mobile_device import DataType

class RatePlanDTO(BaseModel):
    id: str = Field(..., alias="_id")
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
    created_at: datetime
    updated_at: datetime

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}

class UsageDTO(BaseModel):
    usage_amount: int
    usage_month: datetime

class MobileDeviceDTO(BaseModel):
    device_name: str
    data_type: DataType

class UserFullInfoDTO(BaseModel):
    email: str
    age: int
    gender: str

    rate_plan: RatePlanDTO
    call_usages: List[UsageDTO]
    data_usages: List[UsageDTO]
    sms_usages: List[UsageDTO]
    devices: List[MobileDeviceDTO]

