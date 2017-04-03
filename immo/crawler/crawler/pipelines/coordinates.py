# -*- coding: utf-8 -*-
"""
Get longitude and latitude from the openstreetmap

"""
import os
import logging
import requests
from models import utils
from ..settings import OPENSTREETMAP_BASE_URL, GOOGLE_MAP_API_BASE_URL


class CoordinatesPipline(object):
    """get longitutde and latitute for specific address
    """
    def process_item(self, item, spider):
        """after item is processed
        """
        url = "{}".format(OPENSTREETMAP_BASE_URL)
        payload = {'country': 'ch', 'format': 'json', 'addressdetails': 1}

        # Check if we have a streetname
        if item.get('street', None) is not None:
            payload['street'] = item.get('street', None)

        # check if we have a city -> this should be set
        if item.get('place', None) is not None:
            payload['postcode'], *city = utils.get_place(item.get('place'))
            payload['city'] = ' '.join(city)

        # Do GET request and check if answer is ok
        response = requests.get(url, params=payload)
        if response.status_code != 200:
            logging.error("Could not get long and lat for addres %s, %s",
                          item.get('street', None),
                          item.get('place', None))
            return item

        # At the moment always get frist element
        res = response.json()
        logging.debug("Answer %s", res)
        if len(res) > 0:
            item['longitude'] = res[0]['lon']
            item['latitude'] = res[0]['lat']
            logging.debug("Nice found long lat for item %s, %s",
                          item.get('longitude'),
                          item.get('latitude'))
        else:
            # Check google:
            item['longitude'], item['latitude'] = self.askGoogle(item.get('street', None), item.get('place', None) )
            if not item.get('longitude', None):
                logging.warning("Could not get long or lat for address {}, {} cause answer was 0 [{}]".format(item.get('street', None), item.get('place', None), item.get('url', None)))
        return item


    def askGoogle(self, street=None, place=None):
        if not os.environ.get('GOOGLE_MAP_API_KEY', None):
            logging.error("Missing Google map api key")
            return (None, None)

        if not street and not place:
            logging.warning("Google is good but can not lookup addresses with no streetname and no place")
            return (None, None)

        logging.debug("Ask google for coordinates")
        if not street:
            address = "{}".format(place)
        else:
            address = "{},{}".format(street, place)

        url = "{}{}&key={}".format(GOOGLE_MAP_API_BASE_URL, address, os.environ.get('GOOGLE_MAP_API_KEY', None))

        response = requests.get(url)
        if response.status_code != 200:
            return (None, None)

        res = response.json()
        if len(res['results']) > 0:
            long = res['results'][0]['geometry']['location']['lng']
            lat = res['results'][0]['geometry']['location']['lat']
            return long, lat

        return (None, None)




