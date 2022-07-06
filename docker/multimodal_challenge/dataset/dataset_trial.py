from typing import Dict, List, Union
from multimodal_challenge.multimodal_object_init_data import MultiModalObjectInitData


class DatasetTrial:
    """
    Parameters for defining a trial for dataset generation.
    """

    def __init__(self, target_object: MultiModalObjectInitData, force: Dict[str, float],
                 magnebot_position: Dict[str, float],
                 target_object_position: Dict[str, float],
                 distractors: List[Union[dict, MultiModalObjectInitData]]):
        """
        :param target_object: [`MultiModalObjectInitData` initialization data](multimodal_object_init_data.md) for the target object.
        :param force: The initial force of the target object as a Vector3 dictionary.
        :param magnebot_position: The initial position of the Magnebot.
        :param target_object_position: The final position of the target object.
        :param distractors: Initialization data for the distractor objects.
        """

        # Load the drop parameters from a dictionary.
        if isinstance(target_object, dict):
            target_object: dict
            """:field
            target_object: [`MultiModalObjectInitData` initialization data](multimodal_object_init_data.md) for the target object.
            """
            self.target_object = MultiModalObjectInitData(**target_object)
        else:
            self.target_object: MultiModalObjectInitData = target_object
        """:field
        The initial force of the target object as a Vector3 dictionary.
        """
        self.force: Dict[str, float] = force
        """:field
        The initial position of the Magnebot.
        """
        self.magnebot_position: Dict[str, float] = magnebot_position
        """:field
        The final position of the target object.
        """
        self.target_object_position: Dict[str, float] = target_object_position
        """:field
        Initialization data for the distractor objects.
        """
        self.distractors: List[MultiModalObjectInitData] = [d if isinstance(d, MultiModalObjectInitData) else
                                                            MultiModalObjectInitData(**d) for d in distractors]
