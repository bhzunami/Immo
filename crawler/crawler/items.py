# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy
import json

class Ad(scrapy.Item):
    """Ad for all real estate advertisement
    rent and buy
    """
    object_id = scrapy.Field()         # The internal id of the company
    reference_no = scrapy.Field()      # The offical ref number
    raw_data = scrapy.Field(serializer=str)          # The whole html site
    crawler = scrapy.Field()           # which company was crawled
    url = scrapy.Field()               # The url of the site
    available = scrapy.Field()
    street = scrapy.Field()
    place = scrapy.Field()
    price_brutto = scrapy.Field()
    price_netto = scrapy.Field()
    additional_costs = scrapy.Field()
    description = scrapy.Field(serializer=str)
    living_area = scrapy.Field()
    floor = scrapy.Field()             # on which floor
    num_rooms = scrapy.Field()
    num_floors = scrapy.Field()
    build_year = scrapy.Field(serializer=str)
    last_renovation_year = scrapy.Field(serializer=str)
    cubature = scrapy.Field()
    room_height = scrapy.Field()
    effective_area = scrapy.Field()    # Nutzfläche
    plot_area = scrapy.Field()         # Grundstückfläche
    floors_house = scrapy.Field()      # how many floors the whole building have
    characteristics = scrapy.Field(serializer=json.dumps)   # Additional information Seesicht, Lift/ Balkon/Sitzplatz
    additional_data = scrapy.Field(serializer=json.dumps)   # Data at the moment we do not know where to put
    owner = scrapy.Field()             # The name of the copany which insert the ad (if exists)
    objecttype = scrapy.Field()

