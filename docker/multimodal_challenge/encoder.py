from json import JSONEncoder
import numpy as np
from tdw.py_impact import AudioMaterial, ObjectInfo
from multimodal_challenge.multimodal_object_init_data import MultiModalObjectInitData
from multimodal_challenge.dataset.dataset_trial import DatasetTrial
from multimodal_challenge.trial import Trial


class Encoder(JSONEncoder):
    """
    Use this class to encode data to .json for this module.
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, MultiModalObjectInitData):
            return {"name": obj.name,
                    "position": obj.position,
                    "rotation": obj.rotation,
                    "scale_factor": obj.scale_factor,
                    "kinematic": obj.kinematic}
        elif isinstance(obj, ObjectInfo):
            return obj.__dict__
        elif isinstance(obj, DatasetTrial):
            return obj.__dict__
        elif isinstance(obj, Trial):
            return obj.__dict__
        elif isinstance(obj, AudioMaterial):
            return obj.name
        else:
            return super(Encoder, self).default(obj)
