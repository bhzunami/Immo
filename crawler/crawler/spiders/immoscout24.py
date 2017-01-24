import logging
import re

import scrapy
from ..items import Ad
from ..utils import FIELDS as fields

class Immoscout24(scrapy.Spider):
    name = "immoscout24"

    def __init__(self, *args, **kwargs):
        logger = logging.getLogger('scrapy.core.scraper')
        logger.setLevel(logging.INFO)
        super().__init__(*args, **kwargs)


    def start_requests(self):
        # the l parameter describes the canton id!
        urls = ['http://www.immoscout24.ch/de/suche/wohnung-haus-kaufen?s=1&t=2&l=2&se=16&pn=1&ps=120']
        # for i in range(1, 27):
        #     urls.append('http://www.immoscout24.ch/de/suche/wohnung-haus-kaufen?s=1&t=2&l={}&se=16&pn=1&ps=120'.format(i))
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse_list)


    def parse_list(self, response):
        """ Parse the ad list """

        # find ads
        ad_link_path = '//a[@class="item-title"]/@href'

        for link in response.xpath(ad_link_path).extract():
            next_ad = response.urljoin(link)
            yield scrapy.Request(next_ad, callback=self.parse_ad)

        # find next page
        next_page_link = '//a[contains(@class, "next")]/@href'

        next_page_url = response.xpath(next_page_link).extract_first()
        if next_page_url:
            self.logger.debug("Found next page")
            next_page = response.urljoin(next_page_url)
            yield scrapy.Request(next_page, callback=self.parse_list)

    @staticmethod
    def parse_price(price_str):
        print(type(price_str))
        print("called parse_price with: {}".format(price_str))
        m = re.search('CHF ([0-9\\\']+)\\.', price_str)
        print(m)
        if m is not None:
            print(m.group(0))
            return int(m.group(1).replace("'", ""))

    parse_methods = {
        'price_brutto': parse_price.__func__,
        'num_rooms': float,
        'living_area': lambda s: int(s.split(' ')[0])
    }

    def parse_ad(self, response):
        ad = Ad()
        # ad['object_id'] = response.url.split("/")[-1]
        ad['crawler'] = 'immoscout24'
        ad['url'] = response.url
        #ad['raw_data'] = response.body.decode()

        # price, number of rooms, living area
        for div in response.xpath('//div[contains(@class, "layout--columns")]/div[@class="column" and ./div[@class="data-label"]]'):
            key, value, *_ = [x.strip() for x in div.xpath('div//text()').extract()]

            try:
                key = fields[key]
                self.parse_price(value)

                ad[key] = self.parse_methods[key](value)

            except KeyError:
                self.logger.warning("Key not found: {}".format(key))
                ad['additional_data'][key] = value

            print("****************************************")

        print(ad)
