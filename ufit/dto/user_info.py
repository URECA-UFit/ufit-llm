from pydantic import BaseModel
from typing import List
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from bson import ObjectId
from ufit.models.mobile_device import DataType
from ufit.models.rate_plan import RatePlan


class UsageDTO(BaseModel):
    usage_amount: int
    usage_month: datetime

class MobileDeviceDTO(BaseModel):
    device_name: str
    data_type: DataType

class UserInfoDTO(BaseModel):
    email: str
    age: int
    gender: str

    rate_plan: RatePlan
    call_usages: List[UsageDTO]
    data_usages: List[UsageDTO]
    sms_usages: List[UsageDTO]
    devices: List[MobileDeviceDTO]

