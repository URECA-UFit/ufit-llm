from pydantic import BaseModel
from typing import Dict

class callRatePlanRequest(BaseModel):
    ratePlanId: str
    planName: str
    summary: str
    monthlyFee: int
    discountFee: int
    extraData: str
    dataAllowance: str
    dataCategory: str
    voiceAllowance: str
    smsAllowance: str
    basicBenefit: Dict[str, object]
    discountBenefit: Dict[str, object]
    specialBenefit: Dict[str, object]
    deviceType: str
    dataSharing: str
    socialCategory: str 