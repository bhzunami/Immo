import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_
from sqlalchemy.orm import load_only
from models import Advertisement, Municipality

import numpy as np
from osgeo import gdal

class GeoData():
    def __init__(self, filename):
        driver = gdal.GetDriverByName('GTiff')
        dataset = gdal.Open(filename)
        band = dataset.GetRasterBand(1)

        self.ndval = band.GetNoDataValue()

        transform = dataset.GetGeoTransform()
        self.xOrigin = transform[0]
        self.yOrigin = transform[3]
        self.pixelWidth = transform[1]
        self.pixelHeight = -transform[5]

        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        self.geodata = band.ReadAsArray(0, 0, cols, rows)

    # lv03 coordinates
    def mean_by_coord(self, easting, northing):
        num_pix = 15

        all_values = []
        for x in range(-num_pix, num_pix+1):
            for y in range(-num_pix, num_pix+1):
                col = int((easting - self.xOrigin) / self.pixelWidth) + x
                row = int((self.yOrigin - northing) / self.pixelHeight) + y
                all_values.append(self.geodata[row][col])
        all_values = np.array(all_values)
        return np.mean(all_values[np.where(all_values != self.ndval)])

print("Loading tiff file...")
# this file is not checked in, get it here (GeoTiff): https://opendata.swiss/dataset/larmbelastung-durch-strassenverkehr-tag
geodata = GeoData("Strassenlaerm_Tag.tif")

print("Open DB connection")

engine = create_engine(os.environ.get('DATABASE_URL'), connect_args={"application_name":"GDAL data"})
Session = sessionmaker(bind=engine)

# start transaction
session = Session()

try:
    ads = session.query(Advertisement) \
        .options(load_only("id", "lv03_easting", "lv03_northing", "noise_level", 'latitude', 'longitude', 'street', 'municipality_unparsed')) \
        .filter(and_(Advertisement.lv03_easting != None, Advertisement.lv03_easting != 0)) \
        .filter(Advertisement.longitude < 1000) \
        .filter(Advertisement.noise_level == None) \
        .all()

    count = len(ads)

    print("Found {} entries to do.".format(count))

    i = 0
    for ad in ads:
        try:
            ad.noise_level = geodata.mean_by_coord(ad.lv03_easting, ad.lv03_northing)
        except:
            print("Exception on row id: {}, E {} N {} lat {} long {} address {} {}".format(ad.id, ad.lv03_easting, ad.lv03_northing, ad.latitude, ad.longitude, ad.street, ad.municipality_unparsed))

        session.add(ad)
        if i % 100 == 0:
            print("Progress: {}/{} Saving...".format(i+1, count), end="")
            session.commit()
            print("Saved")

        i += 1

    print("Progress: {}/{}".format(count, count))

    session.bulk_update_mappings(Advertisement, [{'id': x.id, 'noise_level': x.noise_level} for x in ads])
    session.commit()


except:
    session.rollback()
    raise

print()
print("Finished.")
