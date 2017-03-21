# -*- coding: utf-8 -*-
"""
Store advertisement in database

"""
import logging
import json
from datetime import date, datetime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from ..settings import DATABASE_URL
from models import Advertisement, ObjectType, Municipality

# from ..models import Advertisement
# from ..models import ObjectType
# from ..models import Municipality

logger = logging.getLogger(__name__)

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
        ad = Advertisement(item)

        # Check if the type of the object is already in the Database
        # If we do not have the type we store a new type.
        obtype_name = item.get('objecttype')
        logger.debug("Search for type %s", obtype_name)
        obtype = session.query(ObjectType).filter(ObjectType.name == item.get('objecttype')).first()
        if not obtype:
            logger.debug("This object type was not found in the database -> Store it")
            # Store new ObjectType
            obtype = ObjectType(name=item.get('objecttype'))
            session.add(obtype)
            # To get the new id
            session.commit()
            logger.debug("Objecttype stored: %i", obtype.id)

        # Now we have for shure a correspondence type
        logger.debug("Objecttype id: %i", obtype.id)

        # Next we have to find our place from the zip and name from the database
        # Get zip
        zip_code, *name = item.get('place').split(' ')
        logger.debug("Search place %s %s", int(zip_code), ' '.join(name))
        # Search in database
        municipalities = session.query(Municipality).filter(Municipality.zip == int(zip_code)).all()

        # It is possible to get more than one municipality so if this happens
        # we search through all
        municipality = None

        # Only one was found
        if len(municipalities) == 1:
            municipality = municipalities[0]
            logger.debug("Found exact one %s ", municipality.name)

        if len(municipalities) > 1:
            logger.debug("Found more than one %i search for %s", len(municipalities), name[0])
            for m in municipalities:
                if m.name.startswith(name[0]) or name[0] in m.alternate_names:
                    municipality = m
                    logger.debug("Found the municipality '%s' for input: %s", municipality.name, item.get('place'))


        if municipality:
            ad.municipalities_id = municipality.id
        else:
            logger.warn("Could not find zip_code %s %s in database", zip_code, ' '.join(name))

        ad.object_types_id = obtype.id

        # Store the add in the database
        try:
            session.add(ad)
            session.commit()
            logger.debug("Advertisement stored: %i", ad.id)
        except Exception as exception:
            logger.error("Could not save advertisement %s cause %s", ad.object_id, exception)
            session.rollback()
            raise
        finally:
            session.close()
        return item

