# -*- coding: utf-8 -*-
"""
"""
import re
from datetime import date, datetime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

CANTONS = ["ZH", "BE", "LU", "UR", "SZ",
           "OW", "NW", "GL", "ZG", "FR",
           "SO", "BS", "BL", "SH", "AR",
           "AI", "SG", "GR", "AG", "TG",
           "TI", "VD", "VS", "NE", "GE",
           "JU"]

def prepare_string(s):
    if s is None:
        return None
    return s.replace("'", "").replace(",", ".")

ignore_values = ['Preis auf Anfrage', 'auf Anfrage', 'Erdgeschoss', 'EG', 'Parterre']

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
    except Exception:
        return None

def get_float(num):
    try:
        return float(extract_number(prepare_string(num)))
    except Exception:
        return None

def get_date(s):
    if s == "sofort":
        return date.today()
    try:
        return datetime.strptime(s, '%d.%m.%Y')
    except (TypeError, ValueError):
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

def maybe_street(street):
    if street in ["Auf Anfrage", "Sur demande", "sur demande", "-", "."]:
        return None
    return street

def get_place(place):
    # Regex find word with 4 digits 1 withespace and then any word with spaces
    address = re.search(r'[0-9]{4}\s[\w\u00c4-\u02AF\s]*', place)
    if address:
        return address.group(0).split()  # We have plz, ort
    # Did not find any place locality try old version:
    return place.split(' ')  # Maybe we have plz and ort
