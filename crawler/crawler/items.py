# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class HomegateAd(scrapy.Item):
    # define the fields for your item here like:
    ad_id = scrapy.Field()
    ref_no = scrapy.Field()
    street = scrapy.Field()
    place = scrapy.Field()
    price_total = scrapy.Field()
    price_netto = scrapy.Field()
    additional_costs = scrapy.Field()
    configuration = scrapy.Field()
    description = scrapy.Field()
    info = scrapy.Field()
    last_updated = scrapy.Field(serializer=str)
