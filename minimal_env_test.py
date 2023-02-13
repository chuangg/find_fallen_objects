from env.envs import make_vec_envs
from env import observation_space, action_space

device = "cpu"
envs = make_vec_envs('find_fallen-v0', 1, "", device, True, spaces=(observation_space, action_space), port=2590, displays=":4", split='train')
obs, info = envs.reset()
obs, reward, done, info = envs.step([5 for _ in range(num_processes)])
print(reward, done)
