# -*- coding: utf-8 -*-
"""
"""
from datetime import date, datetime
from sqlalchemy.ext.declarative import declarative_base
import re

Base = declarative_base()

CANTONS = ["ZH", "BE", "LU", "UR", "SZ", "OW", "NW", "GL", "ZG", "FR", "SO", "BS", "BL", "SH", "AR", "AI", "SG", "GR", "AG", "TG", "TI", "VD", "VS", "NE", "GE", "JU"]

def prepare_string(s):
    if s is None:
        return None
    return s.replace("'", "").replace(",", ".")

ignore_values = ['Preis auf Anfrage', 'auf Anfrage' 'Erdgeschoss', 'EG', 'Parterre']

def extract_number(num):
    try:
        return re.search(r'[0-9]*\.?[0-9]+', num).group(0)
    except IndexError:
        return None
    except AttributeError:
        if num == "UG":
            return -1
        if num in ignore_values:
            return 0

        print("ERROR REGEX INPUT: {}".format(num))
        return None


def get_int(num):
    try:
        return int(extract_number(prepare_string(num)))
    except TypeError:
        return None

def get_float(num):
    try:
        return float(extract_number(prepare_string(num)))
    except TypeError:
        return None

def get_date(s):
    if s == "sofort":
        return date.today()
    try:
        return datetime.strptime(s, '%d.%m.%Y')
    except ValueError:
        pass

    return None

def convert_to_int(num):
    if isinstance(num, int):
        return num
    num = num.replace(",", ".").replace(".–", "").replace("'", "")
    try:
        return int(num)
    except ValueError:
        return None

def extract_municipality(mun):
    return re.sub(" (" + "|".join(CANTONS) + ")$", "", mun)


