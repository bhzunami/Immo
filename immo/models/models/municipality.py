from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON

from .utils import Base

class Municipality(Base):
    """
    """
    __tablename__ = 'municipalities'
    id = Column(Integer, primary_key=True)
    bfsnr = Column(Integer)
    zip = Column(Integer)
    name = Column(String)
    canton_id = Column(Integer)
    district_id = Column(Integer)
    planning_region_id = Column(Integer)
    mountain_region_id = Column(Integer)
    ase = Column(Integer)
    greater_region_id = Column(Integer)
    language_region_id = Column(Integer)
    ms_region_id = Column(Integer)
    job_market_region_id = Column(Integer)
    agglomeration_id = Column(Integer)
    metropole_region_id = Column(Integer)
    tourism_region_id = Column(Integer)
    municipal_size_class_id = Column(Integer)
    urban_character_id = Column(Integer)
    agglomeration_size_class_id = Column(Integer)
    is_town = Column(Integer)
    degurba_id = Column(Integer)
    municipal_type22_id = Column(Integer)
    municipal_type9_id = Column(Integer)
    ms_region_typology_id = Column(Integer)
    lv03_easting = Column(Float)
    lv03_northing = Column(Float)
    alternate_names = Column(JSON)
    lat = Column(Float)
    long = Column(Float)
    noise_level = Column(Float)
    steuerfuss_gde = Column(Float)
    steuerfuss_kanton = Column(Float)
