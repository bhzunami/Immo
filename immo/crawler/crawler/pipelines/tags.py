# -*- coding: utf-8 -*-

import re
import json

with open('./taglist.txt') as f:
    search_words = set([x.split(':')[0] for x in f.read().splitlines()])

remove_tokens = r'[-().,+\':/}{\n\r!?"•;*\[\]%“„ ˋ\t_]'

class TagsPipeline(object):
    def process_item(self, item, spider):

        words = (str(item.get('street', '')) + str(item.get('characteristics', ''))).lower()

        clean_words = set(re.split(' ', re.sub(remove_tokens, ' ', words)))

        item['tags'] = json.dumps([w for w in clean_words if w in search_words])

        return item
