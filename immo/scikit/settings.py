import numpy as np

MODEL_FOLDER = 'models'
IMAGE_FOLDER = 'images'

OBJECT_TYPES = {'ogroup': object,
                'otype': object,
                'municipality': object,
                'floor': object,
                'canton_id': object,
                'district_id': object,
                'mountain_region_id': object,
                'language_region_id': object,
                'job_market_region_id': object,
                'agglomeration_id': object,
                'metropole_region_id': object,
                'tourism_region_id': object}

RNG = np.random.RandomState(42)