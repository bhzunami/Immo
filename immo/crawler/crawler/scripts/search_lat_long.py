import sys
import os
import json
import requests
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import time

from models import Municipality

sys.path.insert(0, '../')
from settings import OPENSTREETMAP_BASE_URL, GOOGLE_MAP_API_BASE_URL

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def process_item_osm(zipcode, city):
    payload = {'country': 'ch', 'format': 'json', 'addressdetails': 1, 'postcode': zipcode, 'city': city}

    # Do GET request and check if answer is ok
    response = requests.get(OPENSTREETMAP_BASE_URL, params=payload)
    resp = response.json()

    if len(resp) > 0:
        return resp[0]['lon'], resp[0]['lat']

    logger.error("ERROR: No results found for: {} {}".format(zipcode, city))
    return None, None

def process_item_google(zipcode, city):
    response = requests.get("{}{}%20{}&key={}".format(GOOGLE_MAP_API_BASE_URL, zipcode, city, os.environ.get('GOOGLE_MAP_API_KEY', None)))

    res = response.json()
    if len(res['results']) > 0:
        long = res['results'][0]['geometry']['location']['lng']
        lat = res['results'][0]['geometry']['location']['lat']
        return long, lat

    logger.error("ERROR: No results found for: {} {}".format(zipcode, city))
    return None, None



engine = create_engine(os.environ.get('DATABASE_URL', None), echo=False)
Session = sessionmaker(bind=engine)

# start transaction
session = Session()

try:
    results = session.query(Municipality).filter(Municipality.lat == None).all()
    i = 0
    count = len(results)
    for mun in results:
        i += 1
        logger.warn("Progress: {} {} {}/{}".format(mun.zip, mun.name, i, count))

        mun.long, mun.lat = process_item_osm(mun.zip, mun.name)
        # mun.long, mun.lat = process_item_google(mun.zip, mun.name)
        session.commit()

        time.sleep(1) # don't spam the OSM server

    session.commit()
except:
    session.rollback()
    raise
