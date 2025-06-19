from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from pymongo.database import Database
from ufit.models.user import User
from ufit.models.mobile_device import MobileDevice
from ufit.models.usages import DataUsage, SmsUsage, CallUsage
from ufit.models.rate_plan import RatePlan
from ufit.dto.user_info import MobileDeviceDTO, UsageDTO, UserFullInfoDTO
from bson import ObjectId


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
    raw_plan = mongo_db.rate_plans.find_one({"_id": ObjectId(user.rate_plan_id)})

    if raw_plan is None:
        # 요금제가 없으면 404 에러 반환
        raise HTTPException(
            status_code=404,
            detail=f"Rate plan {user.rate_plan_id} not found."
        )
    
    rate_plan = RatePlan.model_validate(raw_plan)

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
    rate_plan: RatePlan,
    call_usages: list[CallUsage],
    data_usages: list[DataUsage],
    sms_usages: list[SmsUsage],
    devices: list[MobileDevice],
) -> UserFullInfoDTO:
    # Handle None gender case
    gender_value = user.gender.value if user.gender else "unknown"
    
    # UserFullInfoDTO를 만들어 FastAPI 응답 모델로 사용
    return UserFullInfoDTO(
        email=user.email,  
        age=user.age,      
        gender=gender_value,  
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


def stringify_user_full_info(user: UserFullInfoDTO) -> str:
    if user is None:
        return "사용자 정보가 없습니다."
    
    # Handle None gender case with default fallback
    gender_value = user.gender if user.gender else "unknown"
    gender_kor = {"male": "남성", "female": "여성", "MAN": "남성", "WOMAN": "여성"}.get(gender_value.lower(), "정보 없음")

    device_str = ", ".join([f"{d.device_name} ({d.data_type})" for d in user.devices]) or "없음"
    call_str = ", ".join([f"{u.usage_month.strftime('%Y-%m')}에 {u.usage_amount}분" for u in user.call_usages]) or "없음"
    data_str = ", ".join([f"{u.usage_month.strftime('%Y-%m')}에 {u.usage_amount}MB" for u in user.data_usages]) or "없음"
    sms_str = ", ".join([f"{u.usage_month.strftime('%Y-%m')}에 {u.usage_amount}건" for u in user.sms_usages]) or "없음"
    current_plan = user.rate_plan.plan_name

    return (
        f"사용자 정보:\n"
        f"- 이메일: {user.email}\n"
        f"- 나이: {user.age}세\n"
        f"- 성별: {gender_kor}\n"
        f"- 사용 기기: {device_str}\n"
        f"- 최근 통화 사용량: {call_str}\n"
        f"- 최근 데이터 사용량: {data_str}\n"
        f"- 최근 문자 사용량: {sms_str}\n\n"
        f"현재 사용 중인 요금제 정보:\n{current_plan}"
    )