from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String

from database import Base

class User(Base):

    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    name = Column(String)

    email = Column(
        String,
        unique=True
    )

    password = Column(String)

    role = Column(String)