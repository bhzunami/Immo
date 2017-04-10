import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Advertisement

import crawler.settings as settings

from scrapy.http import HtmlResponse
from scrapy.utils.misc import walk_modules
from scrapy.utils.spider import iter_spider_classes
from scrapy.middleware import MiddlewareManager
from scrapy.utils.conf import build_component_list
from scrapy.settings import Settings

class ItemPipelineManager(MiddlewareManager):
    @classmethod
    def _get_mwlist_from_settings(cls, settings):
        return build_component_list(settings.ITEM_PIPELINES_RESPIDER)

    def _add_middleware(self, pipe):
        super(ItemPipelineManager, self)._add_middleware(pipe)
        if hasattr(pipe, 'process_item'):
            self.methods['process_item'].append(pipe.process_item)

    def process_item(self, item, spider):
        return self._process_chain('process_item', item, spider)


def get_spider(name):
    for module in walk_modules('crawler.spiders'):
        for spcls in iter_spider_classes(module):
            if spcls.name == name:
                return spcls

# parse arguments
try:
    spider = get_spider(sys.argv[1])()
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 100
except:
    print("Usage: python respider.py <string crawlername> [<int offset> [<int batchsize>]]")
    exit(1)

print("Reprocessing crawler: {}".format(spider.name))

# initialise item pipeline
itemProcessor = ItemPipelineManager.from_settings(settings)


engine = create_engine(settings.DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)

# start transaction
session = Session()

try:
    while True:
        print("Fetch data from {} to {}".format(offset, offset + limit))
        results = session.query(Advertisement) \
                  .filter(Advertisement.crawler == spider.name and Advertisement.raw_data != '') \
                  .order_by(Advertisement.id) \
                  .offset(offset) \
                  .limit(limit) \
                  .all()

        offset += limit

        if len(results) == 0:
            break

        for ad in results:
            if ad.raw_data == "":
                continue

            print("Reprocess Ad: {}".format(ad.id))

            new_ad = next(spider.parse_ad(HtmlResponse(url=ad.url, body=ad.raw_data, encoding='utf-8')))
            itemProcessor.process_item(new_ad, spider)

            ad.merge(new_ad)

        print("Commit")
        session.commit()
except:
    session.rollback()
    raise

print("Script finished.")
