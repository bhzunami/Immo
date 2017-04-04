# -*- coding: utf-8 -*-
"""
"""
from sqlalchemy import Column, Integer, String, Float, Boolean
from .utils import Base


class AD_View(Base):
    """View class to select from view
    """
    __tablename__ = 'ad_view'

    id = Column(Integer, primary_key=True)
    living_area	= Column(Float)
    floor = Column(Integer)
    price_brutto = Column(Float)
    build_year = Column(Integer)
    num_rooms = Column(Integer)
    last_renovation_year = Column(Integer)
    otype = Column(String)
    municipality = Column(String)
    canton_id = Column(String)
    district_id	= Column(String)
    mountain_region_id = Column(String)
    language_region_id = Column(String)
    job_market_region_id = Column(String)
    agglomeration_id = Column(String)
    metropole_region_id = Column(String)
    tourism_region_id = Column(String)
    is_town	= Column(Integer)
    lat	= Column(Float)
    long = Column(Float)
    ogroup = Column(String)
    c_view = Column(Boolean)
    c_parking = Column(Boolean)
    c_children = Column(Boolean)
    c_elevator = Column(Boolean)
    c_balcony = Column(Boolean)
    c_patio = Column(Boolean)
    c_newbuilding = Column(Boolean)
