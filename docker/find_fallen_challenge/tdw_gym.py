import json

import gym
import numpy as np
import os
from .controller import Basic_controller as Controller
from magnebot import ActionStatus as TaskStatus

import random
import shlex
import subprocess

class TDW:
    
    def __init__(self, seed=201, port=1071, display=':4', screen_size=300, allow_try=2, split='train', shuffle=True, max_step=300, rank=0, world_size=1):
        self.seed = seed
        self.split = split
        self.port = port
        self.docker_id = None
        self.port = port
        self.random = random.Random(seed) if shuffle else None
        self.p = subprocess.Popen(shlex.split(f'./TDW/TDW.x86_64 -port={port}'), env={'DISPLAY': display})
        self.screen_size = screen_size
        self.controller = Controller(port=port, screen_size = self.screen_size)
        print("Controller connected")

        self.observation_space = gym.spaces.Dict({
            'rgb': gym.spaces.Box(0, 256, (self.screen_size, self.screen_size, 3), dtype=np.uint8),
            # 'seg_mask': gym.spaces.Box(0, 256, (self.screen_size, self.screen_size, 3), dtype=np.uint8),
            'depth': gym.spaces.Box(0, 256, (self.screen_size, self.screen_size), dtype=np.float32),
            # 'category': gym.spaces.Box(0, 256, (self.screen_size, self.screen_size, 3), dtype=np.uint8),
            'agent': gym.spaces.Box(-30, 30, (6, ), dtype=np.float32),
            'FOV': gym.spaces.Box(0, 120, (1,), dtype=np.float32),
            'camera_matrix': gym.spaces.Box(-30, 30, (16,), dtype=np.float32),
            'audio': gym.spaces.Box(0, 256, (12582912, ), dtype=np.uint8),
        })
        self.action_space = gym.spaces.Dict({'type': gym.spaces.Discrete(6)})
        self.max_step = max_step
        self.allow_try = allow_try

        with open(f'split/{split}.txt') as f:
            self.scene_list = f.readlines()
            self.cur_idx = rank
            self.world_size = world_size

    def get_next_scene(self):
        type, mid, trial = self.scene_list[self.cur_idx].strip().split('/')
        self.cur_idx = self.cur_idx + self.world_size
        if self.cur_idx >= len(self.scene_list):
            self.cur_idx = self.cur_idx % len(self.scene_list)
            if self.random is not None:
                self.random.shuffle(self.scene_list)
        scene = mid[:-2]
        layout = int(mid[-1])
        return {
            'type': type,
            'scene': scene,
            'layout': layout,
            'trial': int(trial),
            'seed': self.seed,
        }

        
    def reset(self):
        self.scene_info = self.get_next_scene()
        # print(scene_info)
        self.controller.init_scene(type=self.scene_info['type'], scene=self.scene_info['scene'], layout=self.scene_info['layout'], trial=self.scene_info['trial'])
        
        obs = self.controller._obs()
        info = self.controller._info(self)
        info['num_step'] = 0
        info['scene_info'] = self.scene_info
        
        self.done = False
        self.reward = 0
        self.num_step = 0
        self.distance = 0
        self.action_list = []
        self.first_seen = -1
        self.first_seen_dis = -1
        self.last_position = obs['agent'][[0, 2]]
        self.tried_times = 0

        return obs, info
        
    def step(self, action):
        '''
        Run one timestep of the environment's dynamics
        '''
        self.controller.step_reward = 0
        
        if not isinstance(action, dict):
            a = {"type": action}
            action = a

        pos_before = self.controller.state.magnebot_transform.position
        forward_before = self.controller.state.magnebot_transform.forward
        task_status = TaskStatus.success
        if action["type"] == 0:       #move forward
            task_status = self.controller._move_forward(distance=0.25)
        elif action["type"] == 1:     #turn left
            task_status = self.controller._turn(left=True, angle = 30)
        elif action["type"] == 2:     #turn right
            task_status = self.controller._turn(left=False, angle = 30)
        elif action["type"] == 3:     # slide torso up 
            task_status = self.controller._set_torso(slide_up=True, dis=0.5)
        elif action["type"] == 4:     # slide torso down 
            task_status = self.controller._set_torso(slide_up=False, dis=0.3)
        success = self.controller._check_success(threshold=2)
        if action["type"] == 5:
            self.tried_times += 1
            task_status = success
        success = success == TaskStatus.success
        self.action_list.append(action)
        pos_after = self.controller.state.magnebot_transform.position
        forward_after = self.controller.state.magnebot_transform.forward

        reward = -0.01  # time cost
        done = False
        if action['type'] == 5 and success:
            reward += 10
            done = True
        else:
            reward += self.controller._cal_dis_diff(pos_before,pos_after, forward_before, forward_after)

        self.controller.reward += reward

        obs = self.controller._obs()
        info = self.controller._info(self)

        self.num_step += 1
        self.distance += np.linalg.norm(self.last_position - obs['agent'][[0, 2]])
        self.last_position = obs['agent'][[0, 2]]

        if success and self.first_seen < 0:
            self.first_seen = self.num_step
            self.first_seen_dis = self.distance

        info['scene_info'] = self.scene_info
        obs['status'] = task_status == TaskStatus.success
        info['status'] = task_status == TaskStatus.success
        info['finish'] = done
        if self.num_step >= self.max_step or self.tried_times >= self.allow_try:
            done = True
            self.done = True
        info['done'] = done
        info['num_step'] = self.num_step
        if done:
            info['reward'] = self.controller.reward
            os.makedirs(f'env_log/{self.scene_info["type"]}/{self.scene_info["scene"]}_{self.scene_info["layout"]}', exist_ok=True)
            with open(f'env_log/{self.scene_info["type"]}/{self.scene_info["scene"]}_{self.scene_info["layout"]}/{self.scene_info["trial"]:05d}.log', 'w') as f:
                f.write(json.dumps({
                    'actions': [int(a['type']) for a in self.action_list],
                    'finish': info['finish'],
                    'reward': info['reward'],
                    'steps': len(self.action_list),
                    'distance': float(self.distance),
                    'first_seen': self.first_seen,
                    'first_seen_dis': float(self.first_seen_dis),
                }))

        return obs, reward, done, info
     
    def render(self, mode='human'):
        return None
        
    def save_images(self, dir='./Images'):
        if not os.path.exists(dir):
            os.mkdir(dir)
        self.controller.state.save_images(dir)
    
    def seed(self, seed=None):
        self.seed = np.random.RandomState(seed)
    
    def close(self):
        if self.p is not None:
            print('kill tdw')
            self.p.kill()
            self.p = None
