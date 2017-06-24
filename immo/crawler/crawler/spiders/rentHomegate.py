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

class RentHomegate(scrapy.Spider):
    """Homegate crawler
    """
    name = "rentHomegate"

    @staticmethod
    def get_clean_url(url):
        """Returns clean ad url for storing in database
        """
        return url.split('?')[0]

    def start_requests(self):
        """Start method
        """
        urls = ['https://www.homegate.ch/mieten/immobilien/kanton-aargau/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-appenzellinnerrhoden/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-appenzellausserrhoden/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-baselland/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-baselstadt/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-bern/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-fribourg/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-genf/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-glarus/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-graubuenden/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-jura/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-luzern/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-neuchatel/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-nidwalden/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-obwalden/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-st-gallen/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-schaffhausen/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-schwyz/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-solothurn/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-thurgau/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-tessin/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-uri/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-vaud/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-wallis/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-zug/trefferliste?tab=list',
                'https://www.homegate.ch/mieten/immobilien/kanton-zurich/trefferliste?tab=list']

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
        prices = [p for p in response.xpath(price_path).extract() if p != 'CHF']
        for i, category in enumerate(['price_brutto', 'price_netto', 'additional_costs']):
            try:
                ad[category] = prices[i].replace("'", "").replace(".–", "")
            except IndexError:
                ad[category] = None

        # Characteristics / Merkmale und Ausstattung
        characteristics_path = '//div[contains(@class, "detail-configuration")]/ul/li/text()'
        data = {}
        characteristics = response.xpath(characteristics_path).extract()
        for characteristic in characteristics:
            data[characteristic] = True

        ad['characteristics'] = data

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
