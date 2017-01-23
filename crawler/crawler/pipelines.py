# -*- coding: utf-8 -*-
"""
Define your item pipelines here

See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
"""
import json

class JSONWriterPipeline(object):
    """Write the item as Json
    """

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
        line = json.dumps(dict(item)) + ",\n"
        self.file.write(line.encode())
        print("Craweled {}".format(item.get('object_id')))
        return item

