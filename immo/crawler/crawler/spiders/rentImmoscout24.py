import scrapy
from ..items import Ad
from ..utils import FIELDS


class Rentimmoscout24(scrapy.Spider):
    name = "rentImmoscout24"

    def get_clean_url(self, url):
        """Returns clean ad url for storing in database
        """
        return url.split('?')[0]

    def start_requests(self):
        # the l parameter describes the canton id
        for i in range(1, 27):
            yield scrapy.Request(url='http://www.immoscout24.ch/de/suche/wohnung-haus-mieten?s=1&t=1&l={}&se=16&pn=1&ps=120'.format(i), callback=self.parse)


    def parse(self, response):
        """ Parse the ad list """

        # find ads
        ad_link_path = '//a[@class="item-title"]/@href'

        for link in response.xpath(ad_link_path).extract():
            next_ad = response.urljoin(link)
            yield scrapy.Request(next_ad, callback=self.parse_ad)

        # find next page
        next_page_link = '//a[contains(@class, "next") and not(contains(@class, "disabled"))]/@href'
        next_page_url = response.xpath(next_page_link).extract_first()

        if next_page_url:
            self.logger.debug("Found next page: {}".format(next_page_url))
            next_page = response.urljoin(next_page_url)
            yield scrapy.Request(next_page, callback=self.parse)

    def parse_ad(self, response):
        ad = Ad()
        ad['crawler'] = 'immoscout24'
        ad['url'] = response.url
        ad['raw_data'] = response.body.decode()
        ad['objecttype'] = response.url.split("/")[5].split("-")[0]
        ad['additional_data'] = {}

        # price, number of rooms, living area
        for div in response.xpath('//div[contains(@class, "layout--columns")]/div[@class="column" and ./div[@class="data-label"]]'):
            key, value, *_ = [x.strip()
                              for x in div.xpath('div//text()').extract()]

            try:
                key = FIELDS[key]
                ad[key] = value
            except KeyError:
                self.logger.warning("Key not found: {}".format(key))
                ad['additional_data'][key] = value

        # location
        loc = response.xpath('//table//div[contains(@class, "adr")]')
        ad['street'] = loc.xpath('div[contains(@class, "street-address")]/text()').extract_first()
        ad['place'] = "{} {}".format(loc.xpath('span[contains(@class, "postal-code")]/text()').extract_first().strip(),
                                     loc.xpath('span[contains(@class, "locality")]/text()').extract_first())

        # description
        ad['description'] = ' '.join(response.xpath(
            '//div[contains(@class, "description")]//text()').extract()).strip()

        # more attributes
        ad['characteristics'] = {}

        for elm in response.xpath('//div[contains(@class, "description")]/following-sibling::h2[@class="title-secondary"]'):
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
                    ad[key] = value
                except KeyError:
                    ad['characteristics'][key] = value

        yield ad
