# -*- coding: utf-8 -*-
"""
Store advertisement in database

"""
import json
from datetime import date, datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from ..settings import DATABASE_URL
from ..models.advertisement import Advertisement

class DatabasePipline(object):

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
        ad = Advertisement()
        ad.build_advertisement(item)
        try:
            session.add(ad)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
        return item

