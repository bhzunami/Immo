from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from .utils import Base

class Municipality(Base):
    """
    """
    __tablename__ = 'municipalities'
    id = Column(Integer, primary_key=True)
    zip = Column(Integer)
    name = Column(String)

    # TODO: add all fileds