# Finding Fallen Objects

Official implementation of CVPR 2022 paper "Finding Fallen Objects Via Asynchronous Audio-Visual Integration".

## Usage

### Data

Download the dataset from [here](https://github.com/chuangg/find_fallen_objects/releases/download/fallen_objects/fallen_objects_dataset.tar.gz), and extract it in the project root.

The `dataset` sub-directory contains the necessary information of a case to be loaded into our environment.
The `.wav` files within it are the recorded audio of object falling in each case.

The `perception` sub-directory contains some information helpful for utilizing our environment. Each `.json` file contains several fields for the case.
+ `position` ($x, y, z$) stands for the position of the fallen object __relative to__ the initial state of the agent.
The $y$-axis represents the vertical direction.
$(0, 0, 1)$ is the facing direction of the agent.
+ `name` the name of the fallen object. The same name represents the exact same object model.
+ `category` the category of the fallen object. Each category may have multiple different object models.

### Prerequisite

The environment is based on [TDW](https://github.com/threedworld-mit/tdw). We tested it on version 1.8.29, which you can download TDW_Linux.tar.gz from [here](https://github.com/threedworld-mit/tdw/releases/tag/v1.8.29).

You should follow [this](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/install.md#install-nvidia-and-x-on-your-server) to install NVIDIA and X on your linux server.
If you need to run this environment in docker (suggested), you need also install `nvidia-docker` following [this](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).

After downloading TDW_Linux.tar.gz, extract it into the `docker` directory. The executable TDW should be located at `docker/TDW/TDW.x86_64`.

tdw environment setup:
```sh
conda create -n tdw
conda activate tdw
pip install gym pyastar magnebot==1.3.2 tdw==1.8.29
```


planner environment setup:
```sh
conda create -n planner
conda activate planner
pip install librosa scikit-image pystar2d docker-compose tdw
pip install 'git+https://github.com/facebookresearch/detectron2.git'
cd env/openai_baselines
pip install -e .
```
### Launch the environment

#### Launch 



You can then launch the environment via
```sh
conda activate tdw
python interface.py --display=<display> --split=<split> --port=<port>
```

#### Validate

You can use the `docker/test.py` script to validate the installation for either case. Use port `2590` when launching, or you should edit it in the test script.

The environment will output some information in `env_log/` after each case.

### Use in `gym`

We provide a `gym` wrapper of our environment in `env` directory.
It will launch the environment in a docker container,
so be sure you can launch it inside docker before using this wrapper.
You can create an enviroment like this

```python
import gym
import env
env = gym.make('find_fallen-v0', port=port, display=display, split=split, max_steps=max_steps, rank=rank, world_size=num_processes)
obs, info = env.reset()
obs, reward, done, info = env.step(5)
```

Notes: You should reset the environment after done for each case.

Notes: If you run the python script without `sudo`, you need to ensure that you can also use `docker` commands without `sudo`. If you cannot, you can follow [this instruction](https://docs.docker.com/engine/install/linux-postinstall/#manage-docker-as-a-non-root-user).

`obs` contains following entries:
+ `rgb`, `depth`: the RGB or depth image captured by the agent in the current frame
+ `camera_matrix`: the camera matrix of the captured RGB and depth image
+ `agent` ($x, y, z, fx, fy, fz$): $(x, y, z)$ denotes the current location of the agent, $(fx, fy, fz)$ denotes the current facing direction of the agent
+ `FOV`: field-of-view
+ `audio`: the audio recorded when the object falls down. It's a byte array by padding `1`s to the right of the bytes of `.wav` file.

`info` contains following entries:
+ `scene_info`: a `dict` representing the name of the case
+ `status`: (of type `magnebot.ActionStatus`) the result of the last object, e.g. success or collide
+ `finish`: whether the task has succeeded

Use the following numbers for `action`
+ `0`: move forward
+ `1`: turn left
+ `2`: turn right
+ `3`: move camera up
+ `4`: move camera down
+ `5`: claim that the target is in view within the threshold distance

If you want to run multiple environments in parralel, e.g. for training,
we borrow the code from [openai/baselines](https://github.com/openai/baselines) (slightly modified) so that you can run:

```python
from env.envs import make_vec_envs
envs = make_vec_envs('find_fallen-v0', num_processes, log_dir, device, True, spaces=(observation_space, action_space), port=<port>, displays=<displays>, split='train')
obs, info = envs.reset()
obs, reward, done, info = envs.step([5 for _ in range(num_processes)])
```

Notes: In this case, if a case is `done`, the `obs` and `info` returned by `step` will be the initial status of the next case.

It will use port numbers [`port`, `port + num_processes`), and use X displays in `displays` (it should be a list of strings such as `[":4", ":5"]`).
A single X display can be used for multiple instances simultaneously, so the length of `displays` can be smaller than `num_processes`.

### Baseline

We provide the code of our modular planner in `baseline/planner`.
Run it with (replace `:4 :5` with your available X displays).
You can download the pretrained modular models [here](https://github.com/chuangg/find_fallen_objects/releases/download/fallen_objects/pretrained.tar.gz) and place them in `<project root>/pretrained`.

```sh
conda activate planner
python baseline/planner/main_planner.py --displays :4 :5
```

### Evaluation
You can evaluate the result (SR, SPL, SNA) by putting [script](https://github.com/sjtuyinjie/toolkit/blob/main/eval.py) into the env_log folder and run

```sh
python eval.py
```
you can replace "non_distractor" with "distractor" 

