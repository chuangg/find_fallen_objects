import gym
from gym.envs.registration import register
import numpy as np

register(
    id='find_fallen-v0',
    entry_point='env.find_fallen:FindFallen',
)

observation_space = gym.spaces.Dict({
    'rgb': gym.spaces.Box(0, 256, (300, 300, 3), dtype=np.uint8),
    # 'seg_mask': gym.spaces.Box(0, 256, (300, 300, 3), dtype=np.uint8),
    'depth': gym.spaces.Box(0, 256, (300, 300), dtype=np.float32),
    # 'category': gym.spaces.Box(0, 256, (300, 300, 3), dtype=np.uint8),
    'agent': gym.spaces.Box(-30, 30, (6, ), dtype=np.float32),
    'FOV': gym.spaces.Box(0, 120, (1,), dtype=np.float32),
    'camera_matrix': gym.spaces.Box(-30, 30, (16,), dtype=np.float32),
    'audio': gym.spaces.Box(0, 256, (12582912, ), dtype=np.uint8),
})

action_space = gym.spaces.Dict({'type': gym.spaces.Discrete(6)})
