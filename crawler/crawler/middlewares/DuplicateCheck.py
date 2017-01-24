"""
    Checks if the given URL was already processed
"""
from scrapy.exceptions import IgnoreRequest


class DuplicateCheck(object):
    database = ""

    def __init__(self):
        self.database = "asdf"


    def process_request(self, request, spider):
        if request.url.endswith("robots.txt"):
            return

        print(self.database)

        raise IgnoreRequest
