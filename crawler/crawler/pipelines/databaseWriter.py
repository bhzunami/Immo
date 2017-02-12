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
from ..models import Advertisement
from ..models import ObjectType
from ..models import Municipality

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
        logging.debug("Search for type %s", obtype_name)
        obtype = session.query(ObjectType).filter(ObjectType.name == item.get('objecttype')).first()
        if not obtype:
            logging.info("This object type was not found in the database -> Store it")
            # Store new ObjectType
            obtype = ObjectType(name=item.get('objecttype'))
            session.add(obtype)
            # To get the new id
            session.commit()
            logging.debug("Objecttype stored: %i", obtype.id)

        # Now we have for shure a correspondence type
        logging.debug("Objecttype id: %i", obtype.id)

        # Next we have to find our place from the zip and name from the database
        # Get zip
        zip_code, *name = item.get('place').split(' ')
        logging.debug("Search place %s %s", int(zip_code), ' '.join(name))
        # Search in database
        municipalities = session.query(Municipality).filter(Municipality.zip == int(zip_code)).all()

        # It is possible to get more than one municipality so if this happens
        # we search through all 
        municipality = None

        # Only one was found
        if len(municipalities) == 1:
            municipality = municipalities[0]
            logging.debug("Found exact one %s ", municipality.name)

        if len(municipalities) > 1:
            logging.debug("Found more as one %i search for %s", len(municipalities), name[0])
            for m in municipalities:
                if m.name.startswith(name[0]):
                    municipality = m
                    logging.debug("Found the correct municipality %s", municipality.name)


        if not municipality:
            logging.error("Could not find zip_code %s %s in database", zip_code, ' '.join(name))

        ad.municipalities_id = municipality.id
        ad.object_types_id = obtype.id

        # Store the add in the database
        try:
            session.add(ad)
            session.commit()
            logging.info("Advertisement stored: %i", ad.id)
        except Exception as exception:
            logging.error("Could not save advertisement %s cause %s", ad.object_id, exception)
            session.rollback()
            raise
        finally:
            session.close()
        return item

