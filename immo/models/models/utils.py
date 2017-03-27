# -*- coding: utf-8 -*-
"""
"""
from datetime import date, datetime
from sqlalchemy.ext.declarative import declarative_base
import re

Base = declarative_base()

CANTONS = ["ZH", "BE", "LU", "UR", "SZ", "OW", "NW", "GL", "ZG", "FR", "SO", "BS", "BL", "SH", "AR", "AI", "SG", "GR", "AG", "TG", "TI", "VD", "VS", "NE", "GE", "JU"]

def prepare_string(input):
    if input is None:
        return None
    return input.replace("'", "").replace(",", ".")

ignore_values = ['Preis auf Anfrage', 'auf Anfrage' 'Erdgeschoss', 'EG']

def extract_number(input):
    try:
        return re.search('[0-9]*\.?[0-9]+', input).group(0)
    except IndexError:
        return None
    except AttributeError:
        if input in ignore_values:
            return 0

        print("ERROR REGEX INPUT: {}".format(input))
        return None


def get_int(input):
    try:
        return int(extract_number(prepare_string(input)))
    except TypeError:
        return None

def get_float(input):
    try:
        return float(extract_number(prepare_string(input)))
    except TypeError:
        return None

def get_date(input):
    if input == "sofort":
        return date.today()
    try:
        return datetime.strptime(input, '%d.%m.%Y')
    except ValueError:
        pass

    return None

def convert_to_int(num):
    if type(num) == int:
        return num
    num = num.replace(",", ".").replace(".–", "").replace("'", "")
    try:
        return int(num)
    except ValueError:
        return None

def extract_municipality(mun):
    return re.sub(" (" + "|".join(CANTONS) + ")$", "", mun)


