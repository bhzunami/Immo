
import scrapy

class Immostreet(scrapy.Spider):
    name = "immostreet"

    def start_requests(self):
        urls = ['http://www.immostreet.ch/de/SearchEngine/Kaufen/Schweiz/Wohnung-Haus?AreaId=&AreaIdAgregate=623cacb2-6361-43fe-a400-71bd64378a89&SearchCriteriaImmoId=a2e6a345-51c9-6065-caa5-e4a844798ba7']
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)


    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'AI.html'
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log('Saved file %s' % filename)
