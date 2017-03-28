# -*- coding: utf-8 -*-
"""
The ad is in the div result-items-list class
and every reulst has the id resultItemPanel0 with the number

Links:
http://www.newhome.ch/de/mieten/suchen/wohnung/kanton_basellandschaft/liste.aspx?pc=new https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_basellandschaft/liste.aspx?pc=new

Author: N. Mauchle <nmauchle@gmail.com>

"""
import scrapy
from ..items import Ad
from ..utils import FIELDS

class Newhome(scrapy.Spider):
    """Newhome crawler
    """
    name = "newhome"

    @staticmethod
    def get_clean_url(url):
        """Returns clean ad url for storing in database
        """
        return url

    def start_requests(self):
        """Start method
        """
        urls = ['https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_aargau/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_appenzellinnerrhoden/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_appenzellausserrhoden/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_baselland/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_baselstadt/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_bern/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_fribourg/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_genf/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_glarus/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_graubuenden/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_jura/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_luzern/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_neuchatel/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_nidwalden/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_obwalden/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_st-gallen/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_schaffhausen/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_schwyz/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_solothurn/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_thurgau/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_tessin/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_uri/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_vaud/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_wallis/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_zug/liste.aspx?pc=new',
                'https://www.newhome.ch/de/kaufen/suchen/haus_wohnung/kanton_zurich/liste.aspx?pc=new']

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
            self.logger.debug("Found next page {}".format(next_page_url))
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
        ad['description'] = ' '.join(response.xpath(description_path).extract()).replace('"', '')

        ad['additional_data'] = {}
        fields_path = '//div[@class="content-section details clearfix"]//div[@class="form-group"]'
        for field in response.xpath(fields_path):
            key = field.xpath('span/text()').extract_first()
            value = field.xpath('div/div/text()').extract_first()
            try:
                key = FIELDS[key]
                ad[key] = value.strip()
            except KeyError:
                self.logger.warning("This key not in database: {} for url {}".format(key, response.url))
                ad['additional_data'][key] = value

        # Characteristics / Ausstattung
        characteristics_path = '//div[contains(@class, "environment")]/div[contains(@class, "form")]/div'
        data = {}

        # response.xpath(characteristics_path+'//h4').extract() gives the two elements but
        # for el in response.xpath(characteristics_path:
        #   print(el.xpath('//h4/text()')) only returns the first?
        for title in response.xpath(characteristics_path):
            for category in title.xpath('div[@class="row"]/div'):
                for element in category.xpath('div[@class="form-group"]'):
                    if not element.xpath('span/text()'):
                        continue
                    element_name = element.xpath('span/text()').extract_first().strip()
                    element_value = element.xpath('div/div//text()').extract_first()
                    if not element_value:
                        data[element_name] = True
                    else:
                        data[element_name] = element_value.strip()

        ad['characteristics'] = data

        yield ad
