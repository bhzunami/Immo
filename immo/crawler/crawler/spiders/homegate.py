# -*- coding: utf-8 -*-
"""
The ad is in the div result-items-list class
and every reulst has the id resultItemPanel0 with the number

Links:
https://www.homegate.ch/kaufen/immobilien/kanton-{kanton}/trefferliste?tab=list
https://www.homegate.ch/mieten/immobilien/kanton-{kanton}/trefferliste?tab=list

Author: N. Mauchle <nmauchle@gmail.com>

"""
import scrapy
from ..items import Ad
from ..utils import FIELDS as fields

class Homegate(scrapy.Spider):
    """Homegate crawler
    """
    name = "homegate"

    def get_clean_url(self, url):
        """Returns clean ad url for storing in database
        """
        return url.split('?')[0]

    def start_requests(self):
        """Start method
        """
        urls = ['https://www.homegate.ch/kaufen/immobilien/kanton-aargau/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-appenzellinnerrhoden/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-appenzellausserrhoden/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-baselland/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-baselstadt/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-bern/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-fribourg/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-genf/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-glarus/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-graubuenden/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-jura/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-luzern/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-neuchatel/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-nidwalden/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-obwalden/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-st-gallen/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-schaffhausen/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-schwyz/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-solothurn/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-thurgau/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-tessin/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-uri/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-vaud/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-wallis/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-zug/trefferliste?tab=list',
                'https://www.homegate.ch/kaufen/immobilien/kanton-zurich/trefferliste?tab=list']

        # Go through all urls
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)


    def parse(self, response):
        """Parse the page
        """
        # Go through all ads
        link_path = '//div[starts-with(@id, "resultItemPanel")]' \
                    '/article/a[contains(@class, "detail-page-link")]/@href'

        for link in response.xpath(link_path).extract():
            next_add = response.urljoin(link)
            yield scrapy.Request(next_add, callback=self.parse_ad)

        next_page_path = '//div[@class="paginator-container"]' \
                         '/ul/li[@class="next"]/a/@href'
        next_page_url = response.xpath(next_page_path).extract_first()
        if next_page_url:
            self.logger.debug("Found next page")
            next_page = response.urljoin(next_page_url)
            yield scrapy.Request(next_page, callback=self.parse)

    def parse_ad(self, response):
        """Parse single add
        """
        ad = Ad()
        # object id
        ad['object_id'] = response.url.split("/")[-1]
        ad['crawler'] = 'homegate'
        ad['url'] = response.url
        ad['raw_data'] = response.body.decode()

        # Owner
        owner = '//div[contains(@class, "detail-owner")]/div[@class="infos"]/div[@class="description"]/p'
        ad['owner'] = response.xpath(owner).extract_first()

        # reference number
        ref_path = '//div[contains(@class, "ref")]/span[contains(@class, "text--ellipsis")]/text()'
        ad['reference_no'] = response.xpath(ref_path).extract_first()

        # Address
        address = response.xpath('//div[contains(@class, "detail-address")]/a')
        ad['street'] = address.xpath('h2/text()').extract_first()
        ad['place'] = address.xpath('span/text()').extract_first()

        # Prices
        price_path = '//div[contains(@class, "detail-price")]/ul/li/span/span/text()'
        prices = response.xpath(price_path).extract()
        if len(prices) > 1:
            ad['price_brutto'] = prices[1].replace("'", "").replace(".–", "")
        else:
            ad['price_brutto'] = prices[0]

        # Characteristics / Merkmale und Ausstattung
        characteristics_path = '//div[contains(@class, "detail-configuration")]/ul/li/text()'
        ad['characteristics'] = response.xpath(characteristics_path).extract()

        # Description
        description_path = '//div[contains(@class, "detail-description")]//text()'
        ad['description'] = ' '.join(response.xpath(description_path).extract()).replace("'", " ")

        # list
        key_path = '//div[contains(@class, "detail-key-data")]/ul/li[not(contains(@style, "display:none"))]'
        key_data = response.xpath(key_path)
        ad['additional_data'] = {}
        for data in key_data:
            key, *values = data.xpath('span//text()').extract()
            value = values[0]
            try:
                key = fields[key]
                ad[key] = value
            except KeyError:
                self.logger.warning("This key not in database: {}".format(key))
                ad['additional_data'][key] = value

        yield ad


