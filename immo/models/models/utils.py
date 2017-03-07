# -*- coding: utf-8 -*-
"""
"""
from datetime import date, datetime
from sqlalchemy.ext.declarative import declarative_base
import re

Base = declarative_base()


def prepare_string(input):
    return input.replace("'", "").replace(",", ".")

def extract_number(input):
    try:
        return re.search('[0-9]*\.?[0-9]+', input).group(0)
    except IndexError:
        return None
    except AttributeError:
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
