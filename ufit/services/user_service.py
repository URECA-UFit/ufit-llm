from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from pymongo.database import Database
from ufit.models.user import User
from ufit.models.mobile_device import MobileDevice
from ufit.models.usages import DataUsage, SmsUsage, CallUsage
from ufit.dto.user_info import MobileDeviceDTO, UsageDTO, UserFullInfoDTO


def get_user_full_info(user_id: int, postgre_db: Session, mongo_db: Database ) -> UserFullInfoDTO:
    
    if(user_id==-1): return None

    # 1) PostgreSQL에서 사용자 조회
    user = postgre_db.query(User).filter(User.user_id == user_id).first()
    if user is None:
        # 사용자가 없으면 404 에러 반환
        raise HTTPException(
            status_code=404,
            detail=f"User with id {user_id} not found."
        )

    # 2) MongoDB에서 해당 사용자의 요금제 조회
    rate_plan = mongo_db.rate_plans.find_one({"_id": user.rate_plan_id})
    if rate_plan is None:
        # 요금제가 없으면 404 에러 반환
        raise HTTPException(
            status_code=404,
            detail=f"Rate plan {user.rate_plan_id} not found."
        )

    # 3) PostgreSQL에서 사용량과 디바이스 정보 조회
    call_usages = postgre_db.query(CallUsage).filter(CallUsage.user_id == user_id).all()
    data_usages = postgre_db.query(DataUsage).filter(DataUsage.user_id == user_id).all()
    sms_usages  = postgre_db.query(SmsUsage).filter(SmsUsage.user_id == user_id).all()
    devices     = postgre_db.query(MobileDevice).filter(MobileDevice.user_id == user_id).all()

    # 4) DTO 변환 함수 호출 및 반환
    return to_user_full_info_dto(
        user=user,
        rate_plan=rate_plan,
        call_usages=call_usages,
        data_usages=data_usages,
        sms_usages=sms_usages,
        devices=devices,
    )


def to_user_full_info_dto(
    user: User,
    rate_plan: dict,
    call_usages: list[CallUsage],
    data_usages: list[DataUsage],
    sms_usages: list[SmsUsage],
    devices: list[MobileDevice],
) -> UserFullInfoDTO:
    # UserFullInfoDTO를 만들어 FastAPI 응답 모델로 사용
    return UserFullInfoDTO(
        email=user.email,  
        age=user.age,      
        gender=user.gender.value,  
        rate_plan=rate_plan, # MongoDB에서 온 요금제 데이터
        call_usages=[
            UsageDTO(usage_amount=u.usage_amount, usage_month=u.usage_month)
            for u in call_usages
        ],
        data_usages=[
            UsageDTO(usage_amount=u.usage_amount, usage_month=u.usage_month)
            for u in data_usages
        ],
        sms_usages=[
            UsageDTO(usage_amount=u.usage_amount, usage_month=u.usage_month)
            for u in sms_usages
        ],
        devices=[
            MobileDeviceDTO(
                device_name=d.device_name,
                data_type=d.data_type.value
            ) for d in devices
        ],
    )
