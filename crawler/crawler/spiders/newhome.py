# -*- coding: utf-8 -*-
"""
The ad is in the div result-items-list class
and every reulst has the id resultItemPanel0 with the number

Links:
http://www.newhome.ch/de/mieten/suchen/wohnung/kanton_basellandschaft/liste.aspx?pc=new https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_basellandschaft/liste.aspx?pc=new

Author: N. Mauchle <nmauchle@gmail.com>

"""
import logging
import scrapy
import re
from ..items import Ad
from ..utils import FIELDS as fields

class Newhome(scrapy.Spider):
    """Newhome crawler
    """
    name = "newhome"


    def __init__(self, *args, **kwargs):
        logger = logging.getLogger('scrapy.core.scraper')
        logger.setLevel(logging.INFO)
        super().__init__(*args, **kwargs)

    def start_requests(self):
        """Start method
        """
        # urls = ['https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_basellandschaft/liste.aspx?pc=new']
        urls = ['http://localhost:8000/newhome-kanton_basellandschaft.html']

        # Go through all urls
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)


    def parse(self, response):
        """Parse the page
        """
        # Go through all ads

        link_path = '//div[starts-with(@class, "item")]/div[@class="row"]/div[contains(@class, "detail")]/a/@href'
        for link in response.xpath(link_path).extract():
            next_add = response.urljoin(link)
            yield scrapy.Request(next_add, callback=self.parse_ad)

        next_page_path = '//ul[@class="pagination"]/li[contains(@class, "next")]/a/@href'

        next_page_url = response.xpath(next_page_path).extract_first()
        if next_page_url:
            self.logger.error("Found next page {}".format(next_page_url))
            next_page = response.urljoin(next_page_url)
            yield scrapy.Request(next_page, callback=self.parse)
            

        
    def parse_ad(self, response):
        """Parse single add
        """
        ad = Ad()
        try:
            id = re.search('id=([A-Z]*[0-9])*', response.url).group(0).split('=')[-1]
        except AttributeError:
            id = 0
        ad['object_id'] = response.url.split("/")[-1]
        ad['crawler'] = 'newhome'
        ad['url'] = id
        ad['raw_data'] = response.body.decode()

        # Owner
        owner = '//div[contains(@class, "provider-short")]/p/span/text()'
        ad['owner'] = ' '.join(response.xpath(owner).extract())

        ad['reference_no'] = None

        # Address
        address = response.xpath('//span[@class="sub"]/text()').extract_first().split(',')
        ad['street'] = address[0]
        ad['place'] = address[1].strip()


        price_path = '//div[contains(@class, "detail-price")]/ul/li/span/span/text()'
        prices = response.xpath(price_path).extract()
        if len(prices) > 1:
            ad['price_brutto'] = prices[1].replace("'", "").replace(".–", "")
        else:
            ad['price_brutto'] = prices[0]

       
        self.logger.info("Parse Add")
        yield ad


