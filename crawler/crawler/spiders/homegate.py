# -*- coding: utf-8 -*-
# The ad is in the div result-items-list class
# and every reulst has the id resultItemPanel0 with the number
#
import scrapy
from ..items import HomegateAd

class Homegate(scrapy.Spider):
    name = "homegate"

    def start_requests(self):
        urls = ['https://www.homegate.ch/mieten/immobilien/kanton-baselland/trefferliste?tab=list']
        # https://www.homegate.ch/kaufen/immobilien/kanton-appenzellinnerrhoden/trefferliste?tab=list
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)


    def parse(self, response):
        """Parse the page
        """
        page = response.url.split("/")[-2]
        filename = 'homegate-{}.html'.format(page)
        with open(filename, 'wb') as file:
            file.write(response.body)
        self.log('Saved file %s' % filename)

        # Go throw all ads
        for link in response.xpath('//div[starts-with(@id, "resultItemPanel")]/article/a[contains(@class, "detail-page-link")]/@href').extract():
            next_add = response.urljoin(link)
            yield scrapy.Request(next_add, callback=self.parse_ad)

        next_page_url = response.xpath('//div[@class="paginator-container"]/ul/li[@class="next"]/a/@href').extract_first()
        if next_page_url:
            print("Found next page")
            next_page = response.urljoin(next_page_url)
            yield scrapy.Request(next_page, callback=self.parse)

    def parse_ad(self, response):
        """Parse single add
        """
        ad = HomegateAd()
        ad['ad_id'] = response.url.split("/")[-1]
        filename = 'ad-{}.html'.format(ad.get('ad_id'))
        with open(filename, 'wb') as file:
            file.write(response.body)
        self.log('Saved file %s' % filename)

        address = response.xpath('//div[contains(@class, "detail-address")]/a')
        ad['street'] = address.xpath('h2/text()').extract_first()
        ad['place']  = address.xpath('span/text()').extract_first()

        prices = response.xpath('//div[contains(@class, "detail-price")]/ul/li/span/span/text()').extract()
        ad['price_total'] = prices[1]
        ad['price_netto'] = prices[3]
        ad['additional_costs'] = prices[5]

        key_data = response.xpath('//div[contains(@class, "detail-key-data")]/ul/li[not(contains(@style, "display:none"))]')
        infos = {}
        for data in key_data:
            # Unfortently Wohnflaeche is different
            if data.xpath('span/text()').extract_first() == u"Wohnfläche":
                key = u"Wohnfläche"
                value = data.xpath('span/span/text()').extract_first()
            else:
                key, value = data.xpath('span/text()').extract()
            infos[key] = value

        ad['info'] = infos

        ad['configuration'] = response.xpath('//div[contains(@class, "detail-configuration")]/ul/li/text()').extract()

        ad['description'] = ' '.join(response.xpath('//div[contains(@class, "detail-description")]//text()').extract())

        ad['ref_no'] = response.xpath('//div[contains(@class, "ref")]/span[contains(@class, "text--ellipsis")]/text()').extract_first()


        print("Crawelt new Add {}".format(ad))


