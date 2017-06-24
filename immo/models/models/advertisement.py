# -*- coding: utf-8 -*-
"""
"""
import json
import datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, ForeignKey, JSON
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
    crawled_at = Column(DateTime)
    last_seen = Column(DateTime)
    longitude = Column(Float)
    latitude = Column(Float)
    municipality_unparsed = Column(String)
    quality_label = Column(String)
    lv03_easting = Column(Float)
    lv03_northing = Column(Float)
    noise_level = Column(Float)

    tags = Column(JSON) # JSON list of tags

    # Relationship
    object_types_id = Column(Integer, ForeignKey('object_types.id'))
    object_type = relationship(ObjectType, load_on_pending=True)

    municipalities_id = Column(Integer, ForeignKey('municipalities.id'))
    municipalities = relationship(Municipality, load_on_pending=True)

    def __init__(self, data):
        self.merge(data)

    def assign(self, data, key, default=None, func=lambda x: x):
        setattr(self, key, func(data.get(key, default)) or getattr(self, key))
        #self[key] = func(data.get(key, default)) or self[key]

    def merge(self, data):
        # Set the easy values
        self.assign(data, 'object_id')
        self.assign(data, 'reference_no')
        self.assign(data, 'raw_data')
        self.assign(data, 'crawler')
        self.assign(data, 'url')
        self.assign(data, 'description')
        self.assign(data, 'owner')
        self.assign(data, 'longitude')
        self.assign(data, 'latitude')
        self.assign(data, 'quality_label')
        self.assign(data, 'lv03_easting')
        self.assign(data, 'lv03_northing')
        self.assign(data, 'noise_level')

        self.crawled_at = self.crawled_at or datetime.datetime.now()
        self.last_seen = self.last_seen or datetime.datetime.now()

        self.assign(data, 'street', func=maybe_street)

        self.municipality_unparsed = data.get('place', None) or self.municipality_unparsed
        self.municipalities_id = data.get('municipality_id', None)
        self.object_types_id = data.get('obtype_id', None)

        # Set integers
        self.assign(data, 'price_brutto', func=get_int)
        self.assign(data, 'price_netto', func=get_int)
        self.assign(data, 'additional_costs', func=get_int)
        self.assign(data, 'num_floors', func=get_int)
        self.assign(data, 'build_year', func=get_int)
        self.assign(data, 'last_renovation_year', func=get_int)
        self.assign(data, 'floors_house', func=get_int)

        # Set floats
        self.assign(data, 'living_area', func=get_float)
        self.assign(data, 'floor', func=get_float)
        self.assign(data, 'num_rooms', func=get_float)
        self.assign(data, 'cubature', func=get_float)
        self.assign(data, 'room_height', func=get_float)
        self.assign(data, 'effective_area', func=get_float)
        self.assign(data, 'plot_area', func=get_float)

        # Set jsons
        self.characteristics = json.dumps(data.get('characteristics', None)) or self.characteristics
        self.additional_data = json.dumps(data.get('additional_data', None)) or self.additional_data
        self.tags = json.loads(data.get('tags', None)) or self.tags or []
