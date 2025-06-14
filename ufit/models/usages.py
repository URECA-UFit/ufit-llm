from sqlalchemy import Column, Integer, ForeignKey, DateTime
from ufit.models.time_base import TimeBase
from sqlalchemy.orm import relationship
from ufit.database.database import Base

class UsageBase:
    usage_amount = Column(Integer, nullable=False)
    usage_month = Column(DateTime, nullable=False)

class CallUsage(Base, UsageBase,TimeBase):
    __tablename__ = "call_usages"

    call_usage_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    user = relationship("User")

class DataUsage(Base, UsageBase, TimeBase):
    __tablename__ = "data_usages"

    data_usage_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    user = relationship("User")

class SmsUsage(Base, UsageBase, TimeBase):
    __tablename__ = "sms_usages"
    
    sms_usage_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    user = relationship("User")