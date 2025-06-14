from sqlalchemy import Column, Integer, String, Enum, ForeignKey
from sqlalchemy.orm import relationship
from ufit.models.time_base import TimeBase
from ufit.database.database import Base

import enum

class DataType(enum.Enum):
    LTE = "LTE",
    FIVE_G = "FIVE_G"

class MobileDevice(Base, TimeBase):
    __tablename__ = "mobile_devices"

    mobile_device_id = Column("mobile_device_id", Integer, primary_key=True, autoincrement=True)
    device_name = Column(String(50), nullable=False)
    data_type = Column(Enum(DataType), nullable=False)

    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    user = relationship("User", backref="mobile_devices")