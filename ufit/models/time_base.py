from datetime import datetime
from sqlalchemy import Column, DateTime
from sqlalchemy.ext.declarative import declared_attr
from datetime import datetime, timezone

class TimeBase:
    @declared_attr
    def created_at(cls):
        return Column(DateTime, default=datetime.now(timezone.utc))

    @declared_attr
    def updated_at(cls):
        return Column(DateTime, default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    