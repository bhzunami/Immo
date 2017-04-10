import os
import re
import sys

from collections import Counter

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import models

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

eprint("Read files...")
with open('./replacements.txt') as f:
    replacements = dict([x.split(' ') for x in f.read().splitlines()])
with open('./removewords.txt') as f:
    remove_words = set(f.read().splitlines())
remove_tokens = r'[-().,+\':/}{\n\r!?"•;*\[\]%“„ ˋ\t_◦—=«»~><’‘&@…|−]'


eprint("Get descriptions from database...")
engine = create_engine(os.environ.get('DATABASE_URL'), echo=False)
Session = sessionmaker(bind=engine)
session = Session()
qry = session.query(models.Advertisement.id, models.Advertisement.description, models.Advertisement.characteristics, models.Advertisement.quality_label).all()

eprint("Transform descriptions...")
all_words = []
for q in qry:
    words = set(re.split(' ', re.sub(remove_tokens, ' ', (str(q.description) + str(q.characteristics) + str(q.quality_label)).lower())))
    words = set([w for w in words if w not in remove_words and len(w) > 1 and not w.isdigit()])

    all_words += set([replacements.get(word, word) for word in words])

eprint("Count words...")
counted = Counter(all_words).most_common(100000000)

for key, count in counted:
    print("{}:{}".format(key, count))

eprint("Script finished.")
