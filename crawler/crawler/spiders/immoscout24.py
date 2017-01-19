
import scrapy

class Immoscout24(scrapy.Spider):
    name = "immoscout"

    def start_requests(self):
        urls = ['http://www.immoscout24.ch/de/suche/wohnung-haus-mieten-bs?s=1&t=1&l=1']
        # for i in range(1, 27):
        #     urls.append('http://www.immoscout24.ch/de/suche/wohnung-haus-mieten-bs?s=1&t=1&l={}'.format(i))
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)


    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'AI.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)
