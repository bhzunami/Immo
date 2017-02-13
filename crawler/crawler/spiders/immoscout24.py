import logging
import re

import scrapy
from ..items import Ad
from ..utils import FIELDS
from ..models import utils


class Immoscout24(scrapy.Spider):
    name = "immoscout24"

    def __init__(self, *args, **kwargs):
        logger = logging.getLogger('scrapy.core.scraper')
        logger.setLevel(logging.INFO)
        super().__init__(*args, **kwargs)


    def start_requests(self):
        # the l parameter describes the canton id
        for i in range(1, 27):
            yield scrapy.Request(url='http://www.immoscout24.ch/de/suche/wohnung-haus-kaufen?s=1&t=2&l={}&se=16&pn=1&ps=120'.format(i), callback=self.parse_list)

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

    parse_methods = {
        'price_brutto': utils.parse_price,
        'num_rooms': float,
        'living_area': utils.parse_area,
        'floor': lambda s: int(s.split(" ")[0]),
        'object_id': lambda x: x,
        'num_floors': int,
        'reference_no': lambda x: x,
        'build_year': int,
        'cubature': utils.parse_area,
        'effective_area': utils.parse_area,
        'plot_area': utils.parse_area,
        'last_renovation_year': int,
    }

    def parse_ad(self, response):
        ad = Ad()
        ad['crawler'] = 'immoscout24'
        ad['url'] = response.url
        ad['raw_data'] = response.body.decode()
        ad['object_type'] = response.url.split("/")[5].split("-")[0]

        # price, number of rooms, living area
        for div in response.xpath('//div[contains(@class, "layout--columns")]/div[@class="column" and ./div[@class="data-label"]]'):
            key, value, *_ = [x.strip() for x in div.xpath('div//text()').extract()]

            try:
                key = FIELDS[key]
                ad[key] = self.parse_methods[key](value)
            except KeyError:
                self.logger.warning("Key not found: {}".format(key))
                ad['additional_data'][key] = value

        # location
        loc = response.xpath('//table//div[contains(@class, "adr")]')
        ad['street'] = loc.xpath('div[contains(@class, "street-address")]/text()').extract_first()
        ad['place'] = "{} {}".format(loc.xpath('span[contains(@class, "postal-code")]/text()').extract_first().strip(), loc.xpath('span[contains(@class, "locality")]/text()').extract_first())

        # description
        ad['description'] = ''.join(response.xpath('//div[contains(@class, "description")]//text()').extract()).strip()

        # more attributes
        ad['characteristics'] = {}

        for elm in response.xpath('//div[contains(@class, "description")]/following-sibling::h2[@class="title-secondary"]'):
            title = elm.xpath('.//text()').extract_first()
            entries = {}
            for entry in elm.xpath('./following-sibling::table[1]//tr'):
                key, value = entry.xpath('td')
                key = key.xpath('text()').extract_first()
                if len(value.xpath('span[contains(@class, "tick")]')) == 1:
                    # checkmark
                    value = True
                else:
                    # text
                    value = value.xpath('text()').extract_first()

                # write to additional data, or to structured field
                try:
                    key = FIELDS[key]
                    ad[key] = self.parse_methods[key](value)
                except KeyError:
                    entries[key] = value

            if len(entries) > 0:
                ad['characteristics'][title] = entries

        print("****************************************")
        print(ad)
