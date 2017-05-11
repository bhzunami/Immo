import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import or_, and_
from sqlalchemy.orm import load_only
from models import Advertisement
import json
import requests
from models import utils
import pdb
import time


OPENSTREETMAP_BASE_URL = 'http://nominatim.openstreetmap.org/search/'
GOOGLE_MAP_API_BASE_URL = 'https://maps.googleapis.com/maps/api/geocode/json?address='
ADMIN_GEO = 'http://geodesy.geo.admin.ch/reframe/wgs84tolv03'

engine = create_engine(os.environ.get('DATABASE_URL'))
Session = sessionmaker(bind=engine)

# start transaction
session = Session()

def askGoogle(self, street=None, place=None):
    if not os.environ.get('GOOGLE_MAP_API_KEY', None):
        print("Missing Google map api key")
        return (None, None)

    if not street and not place:
        print("Google is good but can not lookup addresses with no streetname and no place")
        return (None, None)

    if not street:
        address = "{}".format(place)
    else:
        address = "{},{}".format(street, place)

    url = "{}{}&key={}".format(GOOGLE_MAP_API_BASE_URL, address, os.environ.get('GOOGLE_MAP_API_KEY', None))

    response = requests.get(url)
    if response.status_code != 200:
        return (None, None)

    res = response.json()
    if len(res['results']) > 0:
        long = res['results'][0]['geometry']['location']['lng']
        lat = res['results'][0]['geometry']['location']['lat']
        return long, lat

    return (None, None)

def get_lv03():
    # Get 
    ads = session.query(Advertisement) \
                            .options(load_only("id", "longitude", "latitude", "lv03_easting", "lv03_northing")) \
                            .filter(and_(Advertisement.longitude != None, Advertisement.longitude != 0)) \
                            .all()
    for ad in ads:
        url = "{}?easting={}&northing={}&format=json".format(ADMIN_GEO, ad.longitude, ad.latitude)
        response = requests.get(url)
        if response.status_code != 200:
            print("Could not get X and Y for addres {}, {}".format(ad.longitude, ad.latitude))
            continue

        answer = response.json()
        if len(answer) > 0:
            ad.lv03_easting = answer['easting']
            ad.lv03_northing = answer['northing']
        else:
            print("Could not get X and Y for addres {}, {}".format(ad.longitude, ad.latitude))
            continue

def main():
    try:
        ads = session.query(Advertisement) \
                            .options(load_only("id", "longitude", "latitude", "municipality_unparsed", "street")) \
                            .filter(or_(Advertisement.longitude == None, Advertisement.longitude == 0)) \
                            .filter(Advertisement.street != None) \
                            .all()

        print("There are {} long lat missing".format(len(ads)))
        count = len(ads)
        url = "{}".format(OPENSTREETMAP_BASE_URL)
        payload = {'country': 'ch', 'format': 'json', 'addressdetails': 1}
        for ad in ads:
            count -= 1
            payload['postcode'], *city = utils.get_place(ad.municipality_unparsed)
            payload['city'] = ' '.join(city)

            r = requests.get(url, params=payload)
            if r.status_code != 200:
                print("Could not get long and lat for addres {}, {}".format(ad.street, ad.municipality_unparsed))
                continue

            res = r.json()
            if len(res) > 0:
                ad.longitude = res[0]['lon']
                ad.latitude = res[0]['lat']
            else:
                ad.longitude, ad.latitude = askGoogle(ad.street, ad.municipality_unparsed)

            # Check if long lat is present
            if ad.longitude is None or ad.longitude == '':
                print("No coordinates found for {} {}".format(ad.street, ad.municipality_unparsed))
                continue

            print("STORE AD {} to go".format(count))
            session.add(ad)
            session.commit()
            time.sleep(2)
    except:
        session.rollback()
        raise


if __name__ == "__main__":
    main()