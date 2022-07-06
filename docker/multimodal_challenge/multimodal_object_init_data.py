from os.path import join
from typing import Dict
from tdw.object_init_data import AudioInitData, TransformInitData
from tdw.librarian import ModelLibrarian, ModelRecord
from multimodal_challenge.paths import OBJECT_LIBRARY_PATH, ASSET_BUNDLES_DIRECTORY


class MultiModalObjectInitData(AudioInitData):
    """
    Object initialization data for the Multi-Modal Challenge.
    This is exactly the same as `AudioInitData` except that it will always set the library to the local library.
    """

    # Remember where the local library is.
    TransformInitData.LIBRARIES[str(OBJECT_LIBRARY_PATH.resolve())] = \
        ModelLibrarian(library=str(OBJECT_LIBRARY_PATH.resolve()))

    def __init__(self, name: str, scale_factor: Dict[str, float] = None,
                 position: Dict[str, float] = None, rotation: Dict[str, float] = None, kinematic: bool = False):
        """
        :param name: The name of the model.
        :param scale_factor: The scale factor.
        :param position: The initial position. If None, defaults to: `{"x": 0, "y": 0, "z": 0`}.
        :param rotation: The initial rotation as Euler angles or a quaternion. If None, defaults to: `{"w": 1, "x": 0, "y": 0, "z": 0}`
        :param kinematic: If True, the object will be kinematic.
        """

        super().__init__(name=name, scale_factor=scale_factor, position=position, rotation=rotation,
                         kinematic=kinematic, gravity=not kinematic,
                         library=str(OBJECT_LIBRARY_PATH.resolve()))

    def _get_record(self) -> ModelRecord:
        record = TransformInitData.LIBRARIES[str(OBJECT_LIBRARY_PATH.resolve())].get_record(self.name)
        assert record is not None, f"No record for {self.name}"
        # Set the URLs to point at a remote or local asset bundle.
        for platform in record.urls:
            if "ROOT/" in record.urls[platform]:
                url = record.urls[platform].split("ROOT/")[1]
                # Fix the URLs. A few asset bundles have weird URLs.
                url = join(ASSET_BUNDLES_DIRECTORY, url).replace("\\", "/").replace("puzzle_box_composite",
                                                                                    "/puzzle_box_composite")
                if not url.startswith("http"):
                    # Replace double-slashes, e.g. in the URL for baking_sheet01.
                    url = url.replace("//", "/")
                    url = "file:///" + url
                record.urls[platform] = url
        return record
