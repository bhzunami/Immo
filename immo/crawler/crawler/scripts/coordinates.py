import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import or_, and_
from sqlalchemy.orm import load_only, Load
from models import Advertisement, Municipality
import json
import sys
import requests
from models import utils
import pdb
import time


OPENSTREETMAP_BASE_URL = 'http://nominatim.openstreetmap.org/search/'
GOOGLE_MAP_API_BASE_URL = 'https://maps.googleapis.com/maps/api/geocode/json?address='
ADMIN_GEO = 'http://geodesy.geo.admin.ch/reframe/wgs84tolv03'

engine = create_engine(os.environ.get('DATABASE_URL'), connect_args={"application_name":"set_coordiantes"})
Session = sessionmaker(bind=engine)

# start transaction
session = Session()

def askGoogle(self, street=None, zip=None, name=None):
    if not os.environ.get('GOOGLE_MAP_API_KEY', None):
        print("Missing Google map api key")
        return (None, None)

    if not street or not zip:
        print("Google is good but can not lookup addresses with no streetname and no place")
        return (None, None)

    address = "{},{} {}".format(street, zip, name)

    url = "{}{}&key={}".format(GOOGLE_MAP_API_BASE_URL, address, os.environ.get('GOOGLE_MAP_API_KEY', None))

    try:
        response = requests.get(url)
        if response.status_code != 200:
            return (None, None)
        res = response.json()
        if len(res['results']) > 0:
            long = res['results'][0]['geometry']['location']['lng']
            lat = res['results'][0]['geometry']['location']['lat']
            return long, lat
        return (None, None)
    except Exception:
        return (None, None)
    
def get_lv03():
    # Get 
    index = 0
    ads = session.query(Advertisement) \
                            .options(load_only("id", "longitude", "latitude", "lv03_easting", "lv03_northing")) \
                            .filter(and_(and_(Advertisement.longitude != None, Advertisement.longitude != 0), Advertisement.longitude != 9999,)) \
                            .all()
    for ad in ads:
        index += 1
        url = "{}?easting={}&northing={}&format=json".format(ADMIN_GEO, ad.longitude, ad.latitude)
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print("Could not get X and Y for addres {}, {}".format(ad.longitude, ad.latitude))
                continue

            answer = response.json()
            if len(answer) > 0:
                ad.lv03_easting = answer['easting']
                ad.lv03_northing = answer['northing']
                if index % 100 == 0:
                    print("Still get some data....")
                session.add(ad)
                session.commit()
                time.sleep(1)
            else:
                print("Could not get X and Y for addres {}, {}".format(ad.longitude, ad.latitude))
                continue
        except Exception:
            pass


def google():
    try:
        ads = session.query(Advertisement).join(Advertisement.municipalities).options(
                Load(Advertisement).load_only("id", "longitude", "latitude", "street"),
                Load(Municipality).load_only("zip", "name")) \
                .filter(Advertisement.longitude == 8888) \
                .filter(
                    or_(Advertisement.street != None, Advertisement.street != '')) \
                .all()

        print("There are {} long lat missing".format(len(ads)))
        count = len(ads)
        for i, ad in enumerate(ads):
            count -= 1
            ad.longitude, ad.latitude = askGoogle(ad.street, ad.municipalities.zip, ad.municipalities.name)

            if not ad.longitude:
                print("Could not get long and lat for addres {}, {} {}".format(ad.street, ad.municipalities.zip, ad.municipalities.name))
                ad.longitude = 9999  # Not found by openstreetmap
                ad.latitude = 9999
            session.add(ad)

            time.sleep(2)
            session.commit()
            print("STORED AD {} - {} to go".format(ad.id, count))
    except:
        session.rollback()
        raise


def openstreetmap():
    print("Start fetching data")
    try:
        ads = session.query(Advertisement).join(Advertisement.municipalities).options(
                Load(Advertisement).load_only("id", "longitude", "latitude", "street"),
                Load(Municipality).load_only("zip", "name")) \
                .filter(
                    or_(Advertisement.longitude == None, Advertisement.longitude == 0)) \
                .filter(
                    or_(Advertisement.street != None, Advertisement.street != '')) \
                .all()

        print("There are {} long lat missing".format(len(ads)))
        count = len(ads)
        url = "{}".format(os.environ.get('OPENSTREETMAP_BASE_URL', OPENSTREETMAP_BASE_URL))
        print("BASEURL: {}".format(url))
        payload = {'country': 'ch', 'format': 'json', 'addressdetails': 1}
        for i, ad in enumerate(ads):
            count -= 1
            payload['street'] = ad.street
            payload['postcode'] = ad.municipalities.zip
            payload['city'] = ad.municipalities.name

            try:
                r = requests.get(url, params=payload)
                res = r.json()
                if len(res) > 0:
                    ad.longitude = res[0]['lon']
                    ad.latitude = res[0]['lat']
            except Exception:
                pass

            if not ad.longitude:
                print("Could not get long and lat for addres {}, {} {}".format(ad.street, ad.municipalities.zip, ad.municipalities.name))
                ad.longitude = 8888  # Not found by openstreetmap
                ad.latitude = 8888
            session.add(ad)

            if i % 100 == 0:
                session.commit()
                print("STORED AD {} - {} to go".format(ad.id, count))
        session.commit()
    except:
        session.rollback()
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please specify option: [google, open, lv03]")
        sys.exit(1)
    arg = sys.argv[1]
    if arg == "google":
        print("Get data from google")
        google()

    if arg == 'open':
        print("Get data from openstreetmap")        
        openstreetmap()

    if arg == 'lv03':
        print("Get lv03 data")        
        get_lv03()

        