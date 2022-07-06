from typing import Dict
from tdw.py_impact import AudioMaterial


class EnvAudioMaterials:
    """
    PyImpact and Resonance Audio materials for the floor and walls of a scene.
    """

    """:class_var
    A dictionary. Key = A Resonance Audio material. Value = The corresponding PyImpact `AudioMaterial`.
    """
    RESONANCE_AUDIO_TO_PY_IMPACT: Dict[str, AudioMaterial] = {"roughPlaster": AudioMaterial.wood_soft,
                                                              "tile": AudioMaterial.ceramic,
                                                              "concrete": AudioMaterial.ceramic,
                                                              "wood": AudioMaterial.wood_soft,
                                                              "smoothPlaster": AudioMaterial.wood_soft,
                                                              "acousticTile": AudioMaterial.cardboard}

    def __init__(self, floor: str, wall: str):
        """
        :param floor: The Resonance Audio floor material.
        :param wall: The Resonance Audio wall material.
        """

        """:field
        The Resonance Audio floor material.
        """
        self.floor: str = floor
        """:field
        The Resonance Audio wall material.
        """
        self.wall: str = wall
