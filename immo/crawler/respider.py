import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import Advertisement

import crawler.settings as settings

from scrapy.http import HtmlResponse
from scrapy.utils.misc import walk_modules
from scrapy.utils.spider import iter_spider_classes

def get_spider(name):
    for module in walk_modules('crawler.spiders'):
        for spcls in iter_spider_classes(module):
            if spcls.name == name:
                return spcls

try:
    spider = get_spider(sys.argv[1])()
except:
    print("Usage: python respider.py <crawlername> [<offset> [<limit>]]")
    exit(1)

print("Reprocessing crawler: {}".format(spider.name))

engine = create_engine(settings.DATABASE_URL, echo=False)
Session = sessionmaker(bind=engine)

# start transaction
session = Session()

try:
    offset = 0
    limit = 100
    while True:
        print("Fetch data from {} to {}".format(offset, offset + limit))
        results = session.query(Advertisement).filter(Advertisement.crawler == spider.name).order_by(Advertisement.id).offset(offset).limit(limit).all()
        offset += limit

        if len(results) == 0:
            break

        for ad in results:
            print("Reprocess Ad: {}".format(ad.id))

            new_ad = next(spider.parse_ad(HtmlResponse(url=ad.url, body=ad.raw_data, encoding='utf-8')))
            ad.merge(new_ad)

        print("Commit")
        session.commit()
except:
    session.rollback()
    raise

print("Script finished.")
