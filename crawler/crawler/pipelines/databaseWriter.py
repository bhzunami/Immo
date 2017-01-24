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

        # get type
        obtype_name = item.get('objecttype')
        logging.info("Search for type %s", obtype_name)
        obtype = session.query(ObjectType).filter(ObjectType.name == item.get('objecttype')).first()
        if not obtype:
            logging.info("This object type was not found in the database -> Store it")
            # Store new ObjectType
            obtype = ObjectType(name=item.get('objecttype'))
            session.add(obtype)
            # To get the new id
            session.commit()
            logging.info("Objecttype stored: %i", obtype.id)

        logging.info("Objecttype id: %i", obtype.id)
        # Get zip
        zip_code, *name = item.get('place').split(' ')

        logging.info("Search place %s %s", int(zip_code), ' '.join(name))

        municipalities = session.query(Municipality).filter(Municipality.zip == int(zip_code)).all()
        if len(municipalities) == 1:
            municipality = municipalities[0]
            logging.info("Found exact one %s ", municipality.name)

        if len(municipalities) > 1:
            logging.info("Found more as one %i search for %s", len(municipalities), name[0])
            for m in municipalities:
                if m.name.startswith(name[0]):
                    municipality = m
                    logging.info("Found the correct municipality %s", municipality.name)


        if not municipality:
            logging.error("Could not find zip_code %s %s in database", zip_code, ' '.join(name))

        ad.municipalities_id = municipality.id
        ad.object_types_id = obtype.id

        try:
            logging.info("Now we save the whole shit: SHOW THE FUCKING ID: muni %i, obj %i", ad.municipalities_id, ad.object_types_id)
            session.add(ad)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
        return item

