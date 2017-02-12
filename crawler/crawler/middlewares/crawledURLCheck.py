# -*- coding: utf-8 -*-
"""
    Checks if the given URL was already processed
"""
import logging
from scrapy.exceptions import IgnoreRequest
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from ..settings import DATABASE_URL
from ..models import Advertisement

class CrawledURLCheck(object):

    def __init__(self):
        engine = create_engine(DATABASE_URL)
        self.Session = sessionmaker(bind=engine)


    def process_request(self, request, spider):
        """check if the url was already crawled
        """
        session = self.Session()
        advertisement = session.query(Advertisement).filter(Advertisement.url == request.url).all()
        session.close()
        if advertisement:
            logging.info("This url %s was already crawled update last seen", request.url)
            raise IgnoreRequest
        return
