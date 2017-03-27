# -*- coding: utf-8 -*-
"""
Get longitude and latitude from the openstreetmap

"""
import logging
import requests
from ..settings import OPENSTREETMAP_BASE_URL


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
            payload['city'] = item.get('place', None)

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
            logging.warning("Could not get long or lat for address {}, {} cause answer was 0 [{}]".format(item.get('street', None), item.get('place', None), item.get('url', None)))
        return item

