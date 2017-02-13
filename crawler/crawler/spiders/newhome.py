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
from ..items import Ad
from ..utils import FIELDS

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
        urls = ['https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_basellandschaft/liste.aspx?pc=new']
        # urls = ['http://localhost:8000/newhome-kanton_basellandschaft.html']

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
            self.logger.info("Found next page {}".format(next_page_url))
            next_page = response.urljoin(next_page_url)
            yield scrapy.Request(next_page, callback=self.parse)
            

        
    def parse_ad(self, response):
        """Parse single add
        """
        ad = Ad()
        ad['crawler'] = 'newhome'
        ad['url'] = response.url
        ad['raw_data'] = response.body.decode()

        # Owner
        owner = '//div[contains(@class, "provider-short")]/p/span/text()'
        ad['owner'] = ' '.join(response.xpath(owner).extract())

        # Address
        address = response.xpath('//span[@class="sub"]/text()').extract_first().split(',')
        ad['street'] = address[0]
        ad['place'] = address[1].strip()

        description_path = '//div[@id="dDescription"]/span//text()'
        ad['description'] = ' '.join(response.xpath(description_path).extract()).replace('"', '' )

        ad['additional_data'] = {}
        fields_path = '//div[@class="content-section details clearfix"]//div[@class="form-group"]'
        for field in response.xpath(fields_path):
            key = field.xpath('span/text()').extract_first()
            value = field.xpath('div/div/text()').extract_first()
            try:
                key = FIELDS[key]
                ad[key] = value.strip()
            except KeyError:
                self.logger.warning("This key not in database: {}".format(key))
                ad['additional_data'][key] = value


        # Characteristics / Ausstattung
        characteristics_path = '//div[contains(@class, "environment")]/div[contains(@class, "form")]/div'
        data = {}

        # response.xpath(characteristics_path+'//h4').extract() gives the two elements but
        # for el in response.xpath(characteristics_path:
        #   print(el.xpath('//h4/text()')) only returns the first?
        for title in response.xpath(characteristics_path):
            title_name = title.xpath('.//h4/text()').extract_first().strip()
            data[title_name] = {}

            for category in title.xpath('div[@class="row"]/div'):
                category_name = ''.join(category.xpath('h5//text()').extract()).strip()
                data[title_name][category_name] = {}

                for element in category.xpath('div[@class="form-group"]'):
                    element_name = element.xpath('span/text()').extract_first().strip()
                    element_value = element.xpath('div/div//text()').extract_first()
                    if not element_value:
                      data[title_name][category_name][element_name] = 1
                    else:
                        data[title_name][category_name][element_name] = element_value.strip()
        
        ad['characteristics'] = data
       
        yield ad


