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
from models import Advertisement, ObjectType, Municipality, utils

from ..settings import DATABASE_URL

logger = logging.getLogger(__name__)

class DuplicateCheckPipeline(object):

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
        # zimmer
        # wohnfläche
        # preis
        # ort
        municipality_id = item.get('municipality_id')
        objectType_id = item.get('obtype_id')
        num_rooms = utils.get_int(item.get('num_rooms'))
        living_area = utils.get_float(item.get('living_area'))
        price_brutto = utils.get_int(item.get('price_brutto'))
        crawler = item.get('crawler', '')

        ad = session.query(Advertisement) \
                    .filter(Advertisement.num_rooms == num_rooms) \
                    .filter(Advertisement.living_area == living_area) \
                    .filter(Advertisement.price_brutto == price_brutto) \
                    .filter(Advertisement.object_types_id == objectType_id) \
                    .filter(Advertisement.municipalities_id == municipality_id) \
                    .filter(Advertisement.crawler != crawler) \
                    .all()

        session.close()
        if len(ad) > 1:
            logger.info("Found possible duplicate: url in database: {}, duplicate url: {}".format(ad[0].url, item.get('url', '')))
            raise DropItem

        return item

