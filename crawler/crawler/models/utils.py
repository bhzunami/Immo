# -*- coding: utf-8 -*-
"""
"""
from datetime import date, datetime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

def convert_to_int(num):
    if type(num) == int:
        return num
    num = num.replace(",", ".").replace(".–", "").replace("'", "")
    try:
        return int(num)
    except ValueError:
        return None

def convert_to_float(num):
    if type(num) == float:
        return num
    num = num.replace(",", ".").replace(".–", "").replace("'", "")
    try:
        return float(num)
    except ValueError:
        return None

def convert_to_date(data):
    if data == "sofort":
        return date.today()
    try:
        return datetime.strptime(data, '%d.%m.%Y')
    except ValueError:
        pass

    return None
