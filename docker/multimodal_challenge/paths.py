from pathlib import Path
from pkg_resources import resource_filename
from os import environ

"""
Paths to data files in this Python module.
"""

__asset_bundles_key = "MULTIMODAL_ASSET_BUNDLES"
# Use local asset bundles.
if __asset_bundles_key in environ:
    ASSET_BUNDLES_DIRECTORY: str = environ[__asset_bundles_key]
    if ASSET_BUNDLES_DIRECTORY.startswith("~"):
        ASSET_BUNDLES_DIRECTORY = str(Path.home().joinpath(ASSET_BUNDLES_DIRECTORY[2:]).resolve())
# Use remote asset bundles.
else:
    ASSET_BUNDLES_DIRECTORY = "https://tdw-public.s3.amazonaws.com"

__dataset_directory_key = "MULTIMODAL_DATASET"
if __dataset_directory_key in environ:
    __data_dir: str = environ[__dataset_directory_key]
else:
    __data_dir: str = "D:/multimodal_challenge"
# The path to where the dataset data will be generated.
DATASET_ROOT_DIRECTORY: Path = Path(__data_dir)
# The path to the rehearsal data.
REHEARSAL_DIRECTORY: Path = DATASET_ROOT_DIRECTORY.joinpath("rehearsal")
if not REHEARSAL_DIRECTORY.exists():
    REHEARSAL_DIRECTORY.mkdir(parents=True)
# The path to the audio dataset files.
DATASET_DIRECTORY = DATASET_ROOT_DIRECTORY.joinpath("dataset")

# The path to the data files.
DATA_DIRECTORY: Path = Path(resource_filename(__name__, "data"))
# The path to object data.
OBJECT_DATA_DIRECTORY = DATA_DIRECTORY.joinpath("objects")
# The path to the object librarian metadata.
OBJECT_LIBRARY_PATH = OBJECT_DATA_DIRECTORY.joinpath("library.json")
# The path to the list of droppable target objects.
TARGET_OBJECTS_PATH = OBJECT_DATA_DIRECTORY.joinpath("target_objects.txt")
# The path to the list of kinematic objects.
KINEMATIC_OBJECTS_PATH = OBJECT_DATA_DIRECTORY.joinpath("kinematic.txt")
# The path to scene data.
SCENE_DATA_DIRECTORY = DATA_DIRECTORY.joinpath("scenes")
# The path to the occupancy maps.
OCCUPANCY_MAPS_DIRECTORY = SCENE_DATA_DIRECTORY.joinpath("occupancy_maps")
# The path to the scene librarian metadata.
SCENE_LIBRARY_PATH = SCENE_DATA_DIRECTORY.joinpath("library.json")
# The path to the .json files containing object init data.
OBJECT_INIT_DIRECTORY = SCENE_DATA_DIRECTORY.joinpath("object_init")
# The path to the scene bounds data.
SCENE_BOUNDS_DIRECTORY = SCENE_DATA_DIRECTORY.joinpath("bounds")

# The path to the audio dataset files.
AUDIO_DATASET_DIRECTORY = DATA_DIRECTORY.joinpath("dataset")
# The path to the environment audio materials.
ENV_AUDIO_MATERIALS_PATH = AUDIO_DATASET_DIRECTORY.joinpath("audio_materials.json")
# The path to the list of distractor objects.
DISTRACTOR_OBJECTS_PATH = AUDIO_DATASET_DIRECTORY.joinpath("distractor_objects.txt")
# The path to the Magnebot occupancy maps.
MAGNEBOT_OCCUPANCY_MAPS_DIRECTORY = AUDIO_DATASET_DIRECTORY.joinpath("magnebot_occupancy_maps")
