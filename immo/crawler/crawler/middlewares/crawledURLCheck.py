# -*- coding: utf-8 -*-
"""
    Checks if the given URL was already processed
"""
import logging
import datetime
from scrapy.exceptions import IgnoreRequest
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from models import Advertisement
from ..settings import DATABASE_URL
# from ..models import Advertisement

logger = logging.getLogger(__name__)

class CrawledURLCheck(object):

    def __init__(self):
        engine = create_engine(DATABASE_URL)
        self.Session = sessionmaker(bind=engine)


    def process_request(self, request, spider):
        """check if the url was already crawled
        """
        if type(spider).__name__ == "DefaultSpider":
            return

        session = self.Session()

        clean_url = spider.get_clean_url(request.url)

        advertisement = session.query(Advertisement).filter(Advertisement.url == clean_url).first()
        session.close()
        if advertisement:
            logger.debug("This url %s was already crawled update last seen", clean_url)
            advertisement.last_seen = datetime.datetime.now()
            session.add(advertisement)
            session.commit()
            raise IgnoreRequest
        return
