from typing import List
from abc import ABC, abstractmethod
import numpy as np
from tdw.tdw_utils import TDWUtils
from magnebot import Magnebot, ActionStatus
from multimodal_challenge.util import get_scene_librarian, check_pip_version, check_build_version
from multimodal_challenge.paths import OCCUPANCY_MAPS_DIRECTORY


class MultiModalBase(Magnebot, ABC):
    """
    Abstract class controller for the MultiModal challenge.
    The code in this controller shared between [`Dataset`](../dataset/dataset.md) and [`MultiModal`](multimodal.md).
    """

    def __init__(self, port: int = 1071, screen_width: int = 256, screen_height: int = 256, random_seed: int = None,
                 skip_frames: int = 10):
        """
        :param port: The socket port. [Read this](https://github.com/threedworld-mit/tdw/blob/master/Documentation/getting_started.md#command-line-arguments) for more information.
        :param screen_width: The width of the screen in pixels.
        :param screen_height: The height of the screen in pixels.
        :param random_seed: The seed used for random numbers. If None, this is chosen randomly. In the Magnebot API this is used only when randomly selecting a start position for the Magnebot (see the `room` parameter of `init_scene()`). The same random seed is used in higher-level APIs such as the Transport Challenge.
        :param skip_frames: The build will return output data this many physics frames per simulation frame (`communicate()` call). This will greatly speed up the simulation, but eventually there will be a noticeable loss in physics accuracy. If you want to render every frame, set this to 0.
        """

        super().__init__(port=port, launch_build=False, screen_width=screen_width, screen_height=screen_height,
                         auto_save_images=False, random_seed=random_seed, img_is_png=False, skip_frames=skip_frames,
                         check_pypi_version=False)
        check_pip_version()
        check_build_version(self._tdw_version)
        """:field
        The ID of the target object.
        """
        self.target_object_id: int = -1
        self.scene_librarian = get_scene_librarian()

    def init_scene(self, scene: str, layout: int) -> ActionStatus:
        """
        **Always call this function before starting a new trial.**

        Initialize a scene and a furniture layout. Add and position the Magnebot and dropped object.

        :param scene: The name of the scene.
        :param layout: The layout index.

        :return: An `ActionStatus` (always success).
        """

        # Add the scene.
        scene_record = self.scene_librarian.get_record(scene)
        # Load the occupancy map and scene bounds.
        self.occupancy_map = np.load(str(OCCUPANCY_MAPS_DIRECTORY.joinpath(f"{scene}_{layout}.npy").resolve()))

        return self._init_scene(scene=[{"$type": "add_scene",
                                        "name": scene_record.name,
                                        "url": scene_record.get_url()}],
                                post_processing=self._get_post_processing_commands(),
                                end=self._get_end_commands(),
                                magnebot_position=TDWUtils.array_to_vector3(self._get_magnebot_position()))

    @abstractmethod
    def _get_magnebot_position(self) -> np.array:
        """
        :return: The initial position of the Magnebot.
        """

        raise Exception()

    @abstractmethod
    def _get_end_commands(self) -> List[dict]:
        """
        :return: A list of commands to send at the end of scene initialization.
        """

        raise Exception()
