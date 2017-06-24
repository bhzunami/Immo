# -*- coding: utf-8 -*-
"""
Get longitude and latitude from the openstreetmap

"""
import os
import logging
import requests
from models import utils
from ..settings import OPENSTREETMAP_BASE_URL, GOOGLE_MAP_API_BASE_URL, ADMIN_GEO_BASE_URL


class CoordinatesPipeline(object):
    """get longitutde and latitute for specific address
    """
    def process_item(self, item, spider):
        """after item is processed
        """
        logging.debug("Get coordnates for item")
        # Initalize default values to return at any time item
        item['longitude'] = None
        item['latitude'] = None
        item['lv03_easting'] = None
        item['lv03_northing'] = None

        # Check if we have a streetname
        if item.get('street', None) is None or item.get('place', None) is None:
            logging.warn("Address is incomplete to get coordinates")
            return item

        street = item.get('street')
        zip, *city = utils.get_place(item.get('place'))
        city = ' '.join(city)
        logging.debug("Extract street {}, zip {} and city {}".format(street, zip, city))
        # 1. First check openstreetmap
        long, lat = self.get_long_lat_openstreetmap(street, zip, city)

        # If openstreetmap does not find coordinates ask google
        if long is None or lat is None:
            logging.warn("Could not find long lat from openstreetmap try google now")
            long, lat = self.get_long_lat_google(street, zip, city)

            # If google does not have the address it is a wrong address
            if long is None or lat is None:
                logging.warn("Longitude and latitude could not be extracted form {}, {} {}".format(street, zip, city))
                return item

        logging.debug("Found long {}, lat {} for address".format(long, lat))
        item['longitude'] = long
        item['latitude'] = lat
        lv03_easting, lv03_northing = self.get_lv03(long, lat)
        logging.debug("Get lv03 easting {} and lv03 northing {}".format(lv03_easting, lv03_northing))
        item['lv03_easting'] = lv03_easting
        item['lv03_northing'] = lv03_northing

        return item


    def get_long_lat_openstreetmap(self, street=None, zip=None, city=None):
        url = "{}".format(os.environ.get('OPENSTREETMAP_BASE_URL', OPENSTREETMAP_BASE_URL))
        payload = {'country': 'ch', 'format': 'json', 'addressdetails': 1}
        payload['street'] = street
        payload['postcode'] = zip
        payload['city'] = city
        try:
            response = requests.get(url, params=payload)
            res = response.json()
            if len(res) > 0:
                return (res[0]['lon'], res[0]['lat'])

        except Exception:
            return (None, None)
        return (None, None)

    def get_long_lat_google(self, street=None, zip=None, city=None):
        if (not os.environ.get('GOOGLE_MAP_API_KEY', None)) or (not street and not zip):
            logging.warning("Missing API Key or address")
            return (None, None)

        address = "{},{} {}".format(street, zip, city)
        url = "{}{}&key={}".format(GOOGLE_MAP_API_BASE_URL,
                                   address,
                                   os.environ.get('GOOGLE_MAP_API_KEY'))
        try:
            response = requests.get(url)
            if response.status_code != 200:
                return (None, None)
            res = response.json()
            if res['status'] == "OVER_QUERY_LIMIT":
                return ("OVER_QUERY_LIMIT", None)

            if res['status'] == "ZERO_RESULTS":
                return (None, None)

            if len(res['results']) > 0:
                long = res['results'][0]['geometry']['location']['lng']
                lat = res['results'][0]['geometry']['location']['lat']
                return long, lat

            logging.debug(res['status'])

        except Exception:
            return (None, None)
        return (None, None)

    def get_lv03(self, long, lat):
        url = "{}?easting={}&northing={}&format=json".format(ADMIN_GEO_BASE_URL, long, lat)
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print("Could not get X and Y for address {}, {}".format(long, lat))
                return (None, None)

            answer = response.json()
            if len(answer) > 0:
                return (answer['easting'], answer['northing'])
        except Exception:
            return (None, None)
        return (None, None)
