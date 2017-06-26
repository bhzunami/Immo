"""
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

https://www.analyticsvidhya.com/blog/2015/06/tuning-random-forest-model/

https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/
"""
import os
import pdb
import logging
import json


from pipeline import Pipeline


class PredictPipeline(Pipeline):
    def __init__(self, goal, settings, directory):
        super().__init__(goal, settings, directory)
        self.pipeline = [
            self.simple_stats('Before Transformation'),
            self.replace_zeros_with_nan,
            self.transform_build_year,
            self.transform_build_renovation,
            self.transform_noise_level,
            # self.transform_floor,  Floor will be droped
            self.drop_floor,
            self.transform_num_rooms,
            self.transfrom_description,
            self.transform_living_area,
            self.simple_stats('After Transformation'),
            self.transform_tags,
            self.transform_features,
            self.transform_onehot,
            self.outlier_detection,
            self.predict_living_area,
            self.transform_misc_living_area,
            self.extraTreeRegression]
