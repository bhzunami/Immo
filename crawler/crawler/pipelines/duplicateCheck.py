# -*- coding: utf-8 -*-
"""
Store advertisement in database

"""
import json
import logging
from datetime import date, datetime
from scrapy.exceptions import DropItem
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, and_
from ..settings import DATABASE_URL
from ..models import Advertisement
from ..models import ObjectType
from ..models import Municipality
from ..models.utils import convert_to_int, convert_to_float, convert_to_date

class DatabaseWriterPipline(object):

    def open_spider(self, spider):
        """
        Initializes database connection and sessionmaker.
        Creates deals table.
        """
        engine = create_engine(DATABASE_URL)
        self.Session = sessionmaker(bind=engine)


    def process_item(self, item, spider):
        """after item is processed
        """
        session = self.Session()
        type = item.get('objecttype')
        objectType = session.query(ObjectType).filter(ObjectType.name == type).first()
        if not objectType:
            logging.warn("Object type %s not found in database", type)
            # Check for duplicate is not usefull
            session.close()
            return item

        zip_code, *name = item.get('place').split(' ')
        municipality = session.query(Municipality).filter(Municipality.zip == int(zip_code)).filter(Municipality.name == ' '.join(name)).first()

        if not municipality:
            logging.warning("Municipality %s %s not found", zip_code, name)
            session.close()
            return item
        # zimmer
        # wohnfläche
        # preis
        # ort
        num_rooms = convert_to_int(item.get('num_rooms'))
        living_area = convert_to_int(item.get('living_area'))
        price_netto = convert_to_int(item.get('price_netto'))

        ad = session.query(Advertisement).filter(Advertisement.num_rooms == num_rooms).filter(Advertisement.living_area == living_area).filter(Advertisement.price_netto == price_netto).filter(Advertisement.object_types_id == objectType.id).fitler(Advertisement.municipalities_id == municipality.id).all()

        session.close()
        if len(ad) > 1:
            logging.warning("Found possible dublicate: %s", ad[0].id)
            raise DropItem

        return item

