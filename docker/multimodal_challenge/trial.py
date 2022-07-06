from typing import List
import numpy as np
from multimodal_challenge.multimodal_object_init_data import MultiModalObjectInitData


class Trial:
    """
    Data used to initialize a trial. In a trial, the object has already been dropped and generated audio.
    This class will place the Magnebot and every object in the scene at the position at which it stopped moving.
    """

    def __init__(self, object_init_data: List[MultiModalObjectInitData], target_object_index: int,
                 magnebot_position: np.array, magnebot_rotation: np.array):
        """
        :param magnebot_position: The position of the Magnebot as an `[x, y, z]` numpy array.
        :param magnebot_rotation: The rotation of the Magnebot as an `[x, y, z, w]` numpy array.
        :param object_init_data: [Initialization data](multimodal_object_init_data.md) for each object in the scene.
        :param target_object_index: The index of the target object in `object_init_data`.
        """

        if isinstance(object_init_data[0], dict):
            """:field
            Initialization data for each object in the scene. Includes the target object.
            """
            self.object_init_data: List[MultiModalObjectInitData] = list()
            o: dict
            for o in object_init_data:
                a = MultiModalObjectInitData(**o)
                self.object_init_data.append(a)
        else:
            self.object_init_data: List[MultiModalObjectInitData] = object_init_data
        """:field
        The index of the target object in `object_init_data`.
        """
        self.target_object_index: int = target_object_index
        """:field
        The position of the Magnebot as an `[x, y, z]` numpy array.
        """
        self.magnebot_position: np.array = np.array(magnebot_position)
        """:field
        The rotation of the Magnebot as an `[x, y, z, w]` numpy array.
        """
        self.magnebot_rotation: np.array = np.array(magnebot_rotation)
