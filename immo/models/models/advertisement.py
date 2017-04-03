# -*- coding: utf-8 -*-
"""
"""
import json
from datetime import date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey
from .utils import get_int, get_float, get_date, Base, maybe_street
from sqlalchemy.orm import relationship
from . import Municipality, ObjectType

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
    price_netto = Column(Integer)       # If we also want crawl rent
    additional_costs = Column(Integer)  # Additional costs like: Garage or something
    description = Column(String)
    living_area = Column(Integer)
    floor = Column(Integer)             # on which floor
    num_rooms = Column(Float)           # Ow many rooms does this ad have
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
    quality_label = Column(String)

    # Relationship
    object_types_id = Column(Integer, ForeignKey('object_types.id'))
    object_type = relationship(ObjectType)

    municipalities_id = Column(Integer, ForeignKey('municipalities.id'))
    municipalities = relationship(Municipality)

    def __init__(self, data):
        # Set the easy values
        self.object_id = data.get('object_id', None)
        self.reference_no = data.get('reference_no', None)
        self.raw_data = data.get('raw_data', None)
        self.crawler = data.get('crawler', None)
        self.url = data.get('url', None)
        self.street = maybe_street(data.get('street', None))
        self.municipality_unparsed = data.get('place', None)
        self.description = data.get('description', None)
        self.owner = data.get('owner', None)
        self.crawled_at = date.today()
        self.last_seen = date.today()
        self.longitude = data.get('longitude', None)
        self.latitude = data.get('latitude', None)
        self.quality_label = data.get('quality_label', None)

        # Set integers
        self.price_brutto = get_int(data.get('price_brutto', None))
        self.price_netto = get_int(data.get('price_netto', None))
        self.additional_costs = get_int(data.get('additional_costs', None))
        self.num_floors = get_int(data.get('num_floors', None))
        self.build_year = get_int(data.get('build_year', None))
        self.last_renovation_year = get_int(data.get('last_renovation_year', None))
        self.floors_house = get_int(data.get('floors_house', None))

        # Set dates not important!
        # self.available = get_date(data.get('available', None))

        # Set floats
        self.living_area = get_float(data.get('living_area', None))
        self.floor = get_float(data.get('floor', None))
        self.num_rooms = get_float(data.get('num_rooms', None))
        self.cubature = get_float(data.get('cubature', None))
        self.room_height = get_float(data.get('room_height', None))
        self.effective_area = get_float(data.get('effective_area', None))
        self.plot_area = get_float(data.get('plot_area', None))
        # Set jsons
        self.characteristics = json.dumps(data.get('characteristics', None))
        self.additional_data = json.dumps(data.get('additional_data', None))
