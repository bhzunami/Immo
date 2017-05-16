from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from .utils import Base

class ObjectType(Base):
    """
    """
    __tablename__ = 'object_types'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    grouping = Column(String)


