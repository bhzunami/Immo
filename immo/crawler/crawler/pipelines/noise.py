# -*- coding: utf-8 -*-
"""
Send noise level for ad
"""
import logging
import numpy as np
from osgeo import gdal
from models.utils import get_float

logger = logging.getLogger(__name__)

class NoisePipeline(object):
    def open_spider(self, spider):
        """
        Initializes gdal stuff
        """
        driver = gdal.GetDriverByName('GTiff')
        dataset = gdal.Open("Strassenlaerm_Tag.tif")
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

    def process_item(self, item, spider):

        num_pix = 15

        easting = get_float(item.get('lv03_easting'))
        northing = get_float(item.get('lv03_northing'))

        if not easting or not northing:
            item['noise_level'] = None
            return item

        all_values = []
        for x in range(-num_pix, num_pix+1):
            for y in range(-num_pix, num_pix+1):
                col = int((easting - self.xOrigin) / self.pixelWidth) + x
                row = int((self.yOrigin - northing) / self.pixelHeight) + y
                all_values.append(self.geodata[row][col])
        all_values = np.array(all_values)

        item['noise_level'] = np.mean(all_values[np.where(all_values != self.ndval)])
        return item
