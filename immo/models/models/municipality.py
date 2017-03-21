from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON

from .utils import Base

class Municipality(Base):
    """
    """
    __tablename__ = 'municipalities'
    id = Column(Integer, primary_key=True)
    zip = Column(Integer)
    name = Column(String)
    alternate_names = Column(JSON)

    # TODO: add all fileds