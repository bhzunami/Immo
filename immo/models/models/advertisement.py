# -*- coding: utf-8 -*-
"""
"""
import json
from datetime import date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey
from .utils import get_int, get_float, get_date
from sqlalchemy.orm import relationship
from . import Municipality, ObjectType
from .utils import Base

class Advertisement(Base):
    """Advertisment class to store in the database
    """
    __tablename__ = 'advertisements'

    id = Column(Integer, primary_key=True)
    object_id = Column(String)         # The internal id of the company
    reference_no = Column(String)      # The offical ref number
    raw_data = Column(String)          # The whole html site
    crawler = Column(String)           # which company was crawled
    url = Column(String)               # The url of the site
    available = Column(Date)
    street = Column(String)
    price_brutto = Column(Integer)
    price_netto = Column(Integer)
    additional_costs = Column(Integer)
    description = Column(String)
    living_area = Column(Integer)
    floor = Column(Integer)             # on which floor
    num_rooms = Column(Float)
    num_floors = Column(Integer)        # if you have multiple floors
    build_year = Column(Integer)
    last_renovation_year = Column(Integer)
    cubature = Column(Float)
    room_height = Column(Float)
    effective_area = Column(Float)    # Nutzfläche
    plot_area = Column(Float)         # Grundstückfläche
    floors_house = Column(Integer)    # how many floors the whole building have
    characteristics = Column(String)  # Additional information Seesicht, Lift/ Balkon/Sitzplatz
    additional_data = Column(String)  # Data at the moment we do not know where to put
    owner = Column(String)            # The name of the copany which insert the ad (if exists)
    crawled_at = Column(Date)
    last_seen = Column(Date)
    longitude = Column(Float)
    latitude = Column(Float)
    municipality_unparsed = Column(String)

    # Relationship
    object_types_id = Column(Integer, ForeignKey('object_types.id'))
    object_type = relationship(ObjectType)

    municipalities_id = Column(Integer, ForeignKey('municipalities.id'))
    municipalities = relationship(Municipality)

    def __init__(self, data):
        # Set the easy values
        self.object_id = data.get('object_id', '')
        self.reference_no = data.get('reference_no', '')
        self.raw_data = data.get('raw_data', '')
        self.crawler = data.get('crawler', '')
        self.url = data.get('url', '')
        self.street = data.get('street', '')
        self.municipality_unparsed = data.get('place', '')
        self.description = data.get('description', '')
        self.owner = data.get('owner', '')
        self.crawled_at = date.today()
        self.last_seen = date.today()
        self.longitude = data.get('longitude', 0)
        self.latitude = data.get('latitude', 0)

        # Set integers
        self.price_brutto = get_int(data.get('price_brutto', '0'))
        self.price_netto = get_int(data.get('price_netto', '0'))
        self.additional_costs = get_int(data.get('additional_costs', '0'))

        self.living_area = get_int(data.get('living_area', '0'))
        self.floor = get_int(data.get('floor', '0'))
        self.num_floors = get_int(data.get('num_floors', '0'))
        self.build_year = get_int(data.get('build_year', '0'))
        self.last_renovation_year = get_int(data.get('last_renovation_year', '0'))
        self.floors_house = get_int(data.get('floors_house', '0'))

        # Set dates
        self.available = get_date(data.get('available', ''))

        # Set floats
        self.num_rooms = get_float(data.get('num_rooms', '0'))
        self.cubature = get_float(data.get('cubature', '0'))
        self.room_height = get_float(data.get('room_height', '0'))
        self.effective_area = get_float(data.get('effective_area', '0'))
        self.plot_area = get_float(data.get('plot_area', '0'))
        # Set jsons
        self.characteristics = json.dumps(data.get('characteristics', ''))
        self.additional_data = json.dumps(data.get('additional_data', ''))
