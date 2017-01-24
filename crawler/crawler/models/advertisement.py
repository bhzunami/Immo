# -*- coding: utf-8 -*-
"""
"""
import json
from datetime import date, datetime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, Date
from ..settings import DATABASE_URL

Base = declarative_base()


def convert_to_int(num):
    if type(num) == int:
        return num
    num = num.replace(",", ".").replace(".–", "").replace("'", "")
    try:
        return int(num)
    except ValueError:
        return None

def convert_to_float(num):
    if type(num) == float:
        return num
    num = num.replace(",", ".").replace(".–", "").replace("'", "")
    try:
        return float(num)
    except ValueError:
        return None

def convert_to_date(data):
    if data == "sofort":
        return date.today()
    try:
        return datetime.strptime(data, '%d.%m.%Y')
    except ValueError:
        pass

    return None

class Advertisement(Base):
    """
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
    floors_house = Column(Integer)      # how many floors the whole building have
    characteristics = Column(String)   # Additional information Seesicht, Lift/ Balkon/Sitzplatz
    additional_data = Column(String)   # Data at the moment we do not know where to put
    owner = Column(String)           # The name of the copany which insert the ad (if exists)
    crawled_at = Column(Date)
    longitude = Column(Float)
    latitude = Column(Float)


    def build_advertisement(self, data):
        # Set the easy values
        self.object_id = data.get('object_id', '')
        self.reference_no = data.get('reference_no', '')
        self.raw_data = data.get('raw_data', '')
        self.crawler = data.get('crawler', '')
        self.url = data.get('url', '')
        self.street = data.get('street', '')
        self.description = data.get('description', '')
        self.owner = data.get('owner', '')

        self.price_brutto = convert_to_int(data.get('price_brutto', ''))
        self.price_netto = convert_to_int(data.get('price_netto', ''))
        self.additional_costs = convert_to_int(data.get('additional_costs', ''))

        self.living_area = convert_to_int(data.get('living_area', ''))
        self.floor = convert_to_int(data.get('floor', ''))
        self.num_floors = convert_to_int(data.get('num_floors', ''))
        self.build_year = convert_to_int(data.get('build_year', ''))
        self.last_renovation_year = convert_to_int(data.get('last_renovation_year', ''))
        self.floors_house = convert_to_int(data.get('floors_house', ''))

        self.available = convert_to_date(data.get('available', ''))

        self.num_rooms = convert_to_float(data.get('num_rooms', ''))
        self.cubature = convert_to_float(data.get('cubature', ''))
        self.room_height = convert_to_float(data.get('room_height', ''))
        self.effective_area = convert_to_float(data.get('effective_area', ''))
        self.plot_area = convert_to_float(data.get('plot_area', ''))

        self.characteristics = json.dumps(data.get('characteristics', ''))
        self.additional_data = json.dumps(data.get('additional_data', ''))

        self.crawled_at = date.today()
