import json

import scrapy
from scrapy.selector import Selector

from models import utils

from ..items import Ad
from ..utils import FIELDS

class Urbanhome(scrapy.Spider):
    name = "urbanhome"

    def get_clean_url(self, url):
        """Returns clean ad url for storing in database
        """
        return url.split('?')[0]

    def start_requests(self):

        url = "http://www.urbanhome.ch/Search/DoSearch"
        headers = {
            # todo use dynamic user agent?
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
            'Content-Type': 'application/json; charset=UTF-8',
        }

        # canton ids from 7 (AI) to 32 (SH)
        for i in range(7, 33):  # range(7, 33)
            for t in ["1", "4", "512"]: # wohnung, haus, feriendomizil
                data = json.dumps({"settings": {"MainTypeGroup": "1", "Category": "2", "AdvancedSearchOpen": "false", "MailID": "", "PayType": "2", "Type": t, "RoomsMin": "0", "RoomsMax": "0", "PriceMin": "0", "PriceMax": "0", "Regions": [str(i)], "SubTypes": ["0"], "SizeMin": "0", "SizeMax": "0", "Available": "", "NoAgreement": "false", "FloorRange": "0", "equipmentgroups": [], "Email": "", "Interval": "0", "SubscriptionType1": "true", "SubscriptionType4": "true", "SubscriptionType8": "true", "SubscriptionType128": "true", "SubscriptionType512": "true", "Sort": "0"}, "manual": False, "skip": 0, "reset": True, "position": 0, "iframe": 0, "defaultTitle": True, "saveSettings": True})

                yield scrapy.Request(url=url, method="POST", headers=headers, body=data, callback=self.parse)

    def parse(self, response):
        """ Parse the ad list """

        j = json.loads(response.body)

        if not j["Success"]:
            self.logger.error("API error.\n    Exception: " + j["Exception"] + "\n    Message: " + j["Message"])

        # there is no next page option, so we need to improve the search to accomodate this (max. 200 results per search, ~6000 objects to search for)
        if j["MaximumReached"]:
            self.logger.error("Did not receive all search results for: " + j["Url"])

        if j["Count"] == 0:
            self.logger.debug("No results for: " + j["Url"])
            return

        results = Selector(text=j["Rows"])

        for ad in results.xpath("//li"):
            url = ad.xpath('./a/@href').extract_first().split('\\"')[1]

            request = scrapy.Request(url, callback=self.parse_ad)

            # price on detail page is an image, so extract the price from the search page
            request.meta['price_brutto'] = ad.xpath('.//h2/span/following-sibling::text()').extract_first()

            yield request

    def parse_ad(self, response):
        if "Wartungsarbeiten" in response.body.decode():
            return

        ad = Ad()
        ad['crawler'] = 'urbanhome'
        ad['url'] = response.url
        ad['raw_data'] = response.body.decode()
        ad['price_brutto'] = response.meta['price_brutto']
        ad['additional_data'] = {}
        ad['characteristics'] = {}

        base = '//div[@id="xInfos"]//li[@class="cb pt15"]'
        ad['objecttype'] = response.xpath(base + '[1]/h2/text()').extract_first()
        ad['place'] = utils.extract_municipality(' '.join(response.xpath(base + '//span[@itemprop="address"]/span[not(@itemprop="streetAddress")]/text()').extract()))
        ad['street'] = response.xpath(base + '//span[@itemprop="address"]/span[@itemprop="streetAddress"]/text()').extract_first()

        description = response.xpath('//*[@id="xGd"]/div/div[@class="cb pb15"]|//*[@id="xGd"]/div/div[contains(@class, "fl")]//text()').extract()
        description.pop(0) # remove title
        ad['description'] = ' '.join(description)

        for characteristics in response.xpath('//*[@id="xGd"]/div/div[@class="a d"]/ul/li'):
            title = characteristics.xpath('./h6/text()').extract_first()
            ad['characteristics'][title] = characteristics.xpath('./ul/li//span/following-sibling::text()').extract()


        # more attributes
        for entry in response.xpath(base + '/div/text()').extract():
            tokens = entry.split(':')
            if len(tokens) < 2: # ignore lines wich are not in format "key: value"
                continue

            key, value = [x.strip() for x in tokens]

            try:
                key = FIELDS[key]
                ad[key] = value
            except KeyError:
                ad['characteristics'][key] = value

        self.logger.debug("Crawled: " + ad["object_id"] + "    " + ad["url"])
        yield ad
