# -*- coding: utf-8 -*-
"""
Define your item pipelines here

See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
"""
import json
import logging

class JSONWriterPipeline(object):
    """Write the item as Json
    """

    logger = logging.getLogger('jsonwriter')

    def __init__(self, *args, **kwargs):
        self.logger.setLevel(logging.INFO)
        super().__init__(*args, **kwargs)

    def open_spider(self, spider):
        """once when spider is started
        """
        self.file = open('items.jl', 'wb')

    def close_spider(self, spider):
        """called when spider is closed
        """
        self.file.close()

    def process_item(self, item, spider):
        """after item is processed
        """
        del item['raw_data']
        line = json.dumps(dict(item)) + ",\n"
        self.file.write(line.encode())
        self.logger.debug("Crawled {}".format(item.get('object_id')))
        return item

