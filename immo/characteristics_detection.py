import os


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, defer
from sqlalchemy.sql.expression import bindparam
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import select, update
import models
import json
import pdb

engine = create_engine(os.environ.get('DATABASE_URL', None), echo=True)
Session = sessionmaker(bind=engine)

session = Session()

qry = session.query(models.Advertisement.characteristics).all()

statistics = {}

def check_dict(d):
    #print(k, d)

    if isinstance(d, list):
        for val in d:
            check_dict(val)
            add_stat(val, 1)
    elif isinstance(d, dict):
        for key in d:
            add_stat(key, d[key])
            check_dict(d[key])



def add_stat(key, val):
    if isinstance(val, dict):
        val = 'dict'

    if key not in statistics:
        statistics[key] = {"occurence": 1, "values": [val]}
    else:
        statistics[key]["occurence"] += 1
        statistics[key]["values"].append(val)

for entry in qry:
    check_dict(entry[0])

sorted_arr = sorted([[x, statistics[x]["occurence"]] for x in statistics], key=lambda x: x[1])
for k in sorted_arr:
    print(str(k[0]) + " " + str(k[1]))

