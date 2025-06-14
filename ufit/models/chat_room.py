from sqlalchemy import Column, Integer, ForeignKey
from ufit.models.time_base import TimeBase
from sqlalchemy.orm import relationship
from ufit.database.database import Base

class ChatRoom(Base,TimeBase):
    __tablename__ = "chat_rooms"

    chat_room_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    user = relationship("User", back_populates="chat_room")