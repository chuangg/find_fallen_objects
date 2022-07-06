from packaging import version
from json import loads
from typing import List, Dict
from pkg_resources import get_distribution
from os.path import join
from tdw.librarian import SceneLibrarian
from tdw.tdw_utils import TDWUtils
from tdw.version import __version__
from tdw.release.pypi import PyPi
from multimodal_challenge.paths import TARGET_OBJECTS_PATH, OBJECT_INIT_DIRECTORY, SCENE_LIBRARY_PATH, \
    ASSET_BUNDLES_DIRECTORY
from multimodal_challenge.multimodal_object_init_data import MultiModalObjectInitData

# A list of the names of target objects models.
TARGET_OBJECTS: List[str] = TARGET_OBJECTS_PATH.read_text(encoding="utf-8").split("\n")
# The required version of TDW.
TDW_REQUIRED_VERSION = "1.8.25.0"
# The required version of Magnebot.
MAGNEBOT_REQUIRED_VERSION = "1.3.1"


def get_scene_librarian() -> SceneLibrarian:
    """
    :return: The `SceneLibrarian`, which can point to local or remote asset bundles.
    """

    lib = SceneLibrarian(library=str(SCENE_LIBRARY_PATH.resolve()))
    for i in range(len(lib.records)):
        # Set all of the URLs based on the root path.
        for platform in lib.records[i].urls:
            if "ROOT/" in lib.records[i].urls[platform]:
                url = lib.records[i].urls[platform].split("ROOT/")[1]
                url = join(ASSET_BUNDLES_DIRECTORY, url).replace("\\", "/")
                if not url.startswith("http"):
                    url = "file:///" + url
                lib.records[i].urls[platform] = url
    return lib


def get_object_init_commands(scene: str, layout: int) -> List[dict]:
    """
    :param scene: The name of the scene.
    :param layout: The layout variant.

    :return: A list of commands to instantiate objects.
    """

    data = loads(OBJECT_INIT_DIRECTORY.joinpath(f"{scene}_{layout}.json").read_text(encoding="utf-8"))
    commands = list()
    for o in data:
        commands.extend(MultiModalObjectInitData(**o).get_commands()[1])
    return commands


def get_scene_layouts() -> Dict[str, int]:
    """
    :return: A dictionary. Key = The scene name (of the asset bundle). Value = Number of layouts available.
    """

    scene_layouts: Dict[str, int] = dict()
    for f in OBJECT_INIT_DIRECTORY.iterdir():
        # Expected: mm_kitchen_1a_0.json, mm_kitchen_1a_1.json, ... , mm_kitchen_2a_2.json, ...
        if f.is_file() and f.suffix == ".json":
            # Expected: mm_kitchen_1a_0
            s = f.name.replace(".json", "")
            # Expected: mm_kitchen_1a
            scene = s[:-2]
            # Expected: 0
            layout = s[-1]
            scene_layouts[scene] = int(layout) + 1
    return scene_layouts


def get_trial_filename(trial: int) -> str:
    """
    :param trial: The trial number.

    :return: A zero-padded filename for the trial.
    """

    return TDWUtils.zero_padding(trial, 5)


def check_pip_version() -> bool:
    """
    Check the version of TDW and Magenbot.

    :return: True if both the tdw and magnebot pip modules are at the correct version.
    """

    ok = True
    # Check the version of TDW.
    # Use the __version__ variable because it's more likely to be accurate.
    # The version returned by get_distribution can be wrong if this is a test branch.
    # We don't really need to worry about this re: magnebot because that repo isn't updated as frequently.
    if version.parse(TDW_REQUIRED_VERSION) > version.parse(__version__):
        print(f"WARNING! You have tdw {__version__} but you need tdw {TDW_REQUIRED_VERSION}. "
              f"To install the correct version:"
              f"\nIf you installed tdw from the GitHub repo (pip3 install -e .): "
              f"git checkout v{PyPi.strip_post_release(TDW_REQUIRED_VERSION)}"
              f"\nIf you installed tdw from PyPi (pip3 install tdw): "
              f"pip3 install tdw=={TDW_REQUIRED_VERSION}")
        ok = False
    magnebot_installed_version = get_distribution("magnebot").version
    if version.parse(MAGNEBOT_REQUIRED_VERSION) >= version.parse(magnebot_installed_version):
        print(f"WARNING! You have magnebot {magnebot_installed_version} but you need magnebot {MAGNEBOT_REQUIRED_VERSION}. "
              f"To install the correct version:"
              f"\nIf you installed tdw from the GitHub repo (pip3 install -e .): "
              f"git checkout {MAGNEBOT_REQUIRED_VERSION}"
              f"\nIf you installed tdw from PyPi (pip3 install magnebot): "
              f"pip3 install magnebot=={MAGNEBOT_REQUIRED_VERSION}")
        ok = False
    return ok


def check_build_version(build_version: str) -> bool:
    """
    Check the version of the build.

    :param build_version: The version of the build.

    :return: True if this is the correct version of the build.
    """

    tdw_required_version_stripped = PyPi.strip_post_release(TDW_REQUIRED_VERSION)
    if version.parse(build_version) != version.parse(tdw_required_version_stripped):
        print(f"WARNING! You are using TDW build {build_version} but you need TDW build {tdw_required_version_stripped}. "
              f"\nDownload and extract from here: "
              f"https://github.com/threedworld-mit/tdw/releases/tag/v{tdw_required_version_stripped}")
        return False
    else:
        return True
