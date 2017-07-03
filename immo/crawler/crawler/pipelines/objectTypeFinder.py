# -*- coding: utf-8 -*-
"""
Find the correct municipality

"""
import logging
import json
from datetime import date, datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import Advertisement, ObjectType, Municipality
from models.utils import extract_number, get_place, extract_municipality
from ..settings import DATABASE_URL

logger = logging.getLogger(__name__)

class ObjectTypeFinderPipeline(object):
    def open_spider(self, spider):
        """
        Initializes database connection and sessionmaker.
        Creates deals table.
        """
        engine = create_engine(DATABASE_URL)
        self.Session = sessionmaker(bind=engine)

    def process_item(self, item, spider):
        session = self.Session()

        object_str = item.get('objecttype', '').lower()
        # Check if the type of the object is already in the Database
        # If we do not have the type we store a new type.
        logger.debug("Search for type %s", object_str)
        obtype = session.query(ObjectType).filter(ObjectType.name == object_str).first()
        if not obtype:
            logger.warn("This object type {} was not found in the database -> Store it".format(object_str))
            # Store new ObjectType
            obtype = ObjectType(name=object_str)
            session.add(obtype)
            # To get the new id
            session.commit()
            logger.debug("Objecttype stored: %i", obtype.id)

        logger.debug("Objecttype: {}".format(obtype.name))
        item['obtype_id'] = obtype.id

        session.close()
        return item
