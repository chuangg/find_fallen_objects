import math
import random
from collections import OrderedDict
from multiprocessing import Pipe, Process
from typing import List

import numpy as np
import pyastar2d
from scipy.spatial import ConvexHull
from skimage.measure import label
from tdw.tdw_utils import TDWUtils

from .utils.bresenham import bresenhamline

from .module.sound_category import CATE_IDX_COLOR, CATEGORY_IDX
from .module.obs_mask import SegMaskPredictor

CELL_SIZE = 0.1
ANGLE = 30
CATE_THRESHOLD = 0.05

def convex_hull(p):
    hull = p[ConvexHull(p).vertices]
    points = []
    for i in range(hull.shape[0]):
        points.append(bresenhamline(hull[None, i], hull[None, (i + 1) % hull.shape[0]], max_iter=-1))
    return np.concatenate(points, axis=0)

class H_agent:
    def __init__(self, device, learned):
        random.seed(1024)
        self.map_size = (100, 100)
        self._scene_bounds = {
            "x_min": -5,
            "x_max": 5,
            "z_min": -5,
            "z_max": 5,
        }
        self.mask_predictor = SegMaskPredictor(device, learned)

    def pos2map(self, pos):
        x, z = pos[0], pos[-1]
        i = int(round((x - self._scene_bounds["x_min"]) / CELL_SIZE))
        j = int(round((z - self._scene_bounds["z_min"]) / CELL_SIZE))
        return np.array((i, j))

    def map2pos(self, map):
        i, j = map
        x = i * CELL_SIZE + self._scene_bounds["x_min"]
        z = j * CELL_SIZE + self._scene_bounds["z_min"]
        return np.array((x, z))

    def dep2map(self, show_rpc=None):
        depth = self.obs['depth']
        rpc = TDWUtils.get_point_cloud(depth.astype(float), self.obs['camera_matrix'].astype(float), vfov=90)

        self.rpc = rpc

        frpc = rpc.reshape(3, -1)
        X = np.rint((frpc[0, :] - self._scene_bounds["x_min"]) / CELL_SIZE).astype(np.int32)
        X = np.clip(X, 0, self.map_size[0] - 1)
        Z = np.rint((frpc[2, :] - self._scene_bounds["z_min"]) / CELL_SIZE).astype(np.int32)
        Z = np.clip(Z, 0, self.map_size[1] - 1)
        height = frpc[1, :]
        depth = depth.reshape(-1)
        # seg_mask = self.obs['seg_mask'][::-1].reshape(-1, 3)
        seg_mask = self.seg_mask = self.mask_predictor.get_cate_mask(self.obs['rgb'])
        nothing = (seg_mask == 0).all(axis=-1)

        for i in range(height.shape[0]):
            if 0.2 < height[i] < 1.5:
                self.occupancy_map[X[i], Z[i]] = 1
            pos = frpc[[0, 2], i]
            if np.linalg.norm(pos - self.position[[0, 2]]) <= 2:
                self.known_map[X[i], Z[i]] = 1

        segm = np.zeros(seg_mask.shape[:-1], dtype=np.long)
        for i in range(30):
            if self.category_belief[i] > CATE_THRESHOLD:
                segm[np.all(seg_mask == CATE_IDX_COLOR[i], axis=-1)] = i + 1
        # segm[np.all(seg_mask == CATE_IDX_COLOR[CATEGORY_IDX[self.category_gt]], axis=-1)] = 1
        segm_l, l_cnt = label(segm, return_num=True, connectivity=2)
        self.seg_l = segm_l
        for i in range(1, l_cnt + 1):
            x, y = np.argwhere(segm_l == i).mean(axis=0).astype(np.int)
            x, y, z = rpc[:, x, y]
            x, z = self.pos2map((x, z))
            self.seg_pos_map[x-2:x+2, z-2:z+2] = 1


    def get_seg_position(self):
        ret = []
        for _, _, loc in self.obj_infos.values():
            ret.append(self.pos2map(loc))
        return ret

    def get_object_position(self, object_id):
        return self.object_position[object_id]

    def get_angle_delta(self, f, d):
        dot = (f * d).sum()
        det = f[0] * d[1] - f[1] * d[0]
        return np.rad2deg(np.arctan2(det, dot))

    def get_angle(self, forward, origin, position):
        p0 = np.array([origin[0], origin[2]])
        p1 = np.array([position[0], position[2]])
        d = p1 - p0
        d = d / np.linalg.norm(d)
        f = np.array([forward[0], forward[2]])

        dot = f[0] * d[0] + f[1] * d[1]
        det = f[0] * d[1] - f[1] * d[0]
        angle = np.arctan2(det, dot)
        angle = np.rad2deg(angle)
        return angle

    def conv2d(self, map, kernel=3):
        from scipy.signal import convolve2d
        conv = np.ones((kernel, kernel))
        return convolve2d(map, conv, mode='same', boundary='fill')

    def reset(self, goal, category, category_gt=None, target=None, log_file=None):
        self.W = self.map_size[0]
        self.H = self.map_size[1]
        # 0: free, 1: occupied
        self.occupancy_map = np.zeros(self.map_size, np.int32)
        # 0: unknown, 1: known
        self.move_failed_map = np.zeros(self.map_size, np.int32)
        # 0: free, 1: target object, 2: container, 3: goal
        self.seg_pos_map = np.zeros(self.map_size, np.int32)
        self.tried_pos_map = np.zeros(self.map_size, np.int32)
        self.checked_pos = []
        self.last_agent = None
        self.last_action = 0
        self.obs_dis_map = np.zeros(self.map_size)
        self.obs_dis_map[...] = 1e20
        self.known_map = np.zeros(self.map_size, np.int32)
        self.obj_infos = OrderedDict()
        self.category_belief = category
        if category_gt is not None:
            if self.category_belief[CATEGORY_IDX[category_gt]] > CATE_THRESHOLD:
                print('cate in belief')
            else:
                print('cate not in belief')
            print(category_gt, CATE_IDX_COLOR[CATEGORY_IDX[self.category_gt]])
        self.goal = goal
        self.local_goal = goal
        self.tried_obj = set()
        self.target_obj = False
        self.visited_map = np.zeros(self.map_size, np.int32)
        self.log_file = log_file
        self.step = 0
        self.torso_cnt = 0
        self.move_fail_cnt = 0
        self.initial_move_cnt = 0
        if self.log_file is not None:
            open(self.log_file, 'w').close()
        self.target = target

    def log(self, content):
        if self.log_file is not None:
            with open(self.log_file, 'a') as f:
                f.write(content + '\n')

    def find_goal(self):
        # if self.step >= 150:
        pos_l, num = label(self.seg_pos_map & (1 - self.tried_pos_map), return_num=True)
        yy, xx = np.meshgrid(np.arange(self.seg_pos_map.shape[1]), np.arange(self.seg_pos_map.shape[0]))
        positions = []
        for i in range(1, num + 1):
            x = xx[pos_l == i].mean()
            y = yy[pos_l == i].mean()
            positions.append(self.map2pos((x, y)))
        if len(positions):
            self.target_obj = True
            self.local_goal = min(positions, key=lambda p: np.linalg.norm(p - self.position[[0, -1]]))
            return
        self.log(f'explore more')
        x, y = self.pos2map(self.goal)
        cx, cy = self.pos2map(self.position)
        map = self.get_occupancy_map()
        map = (self.conv2d(map, kernel=5) > 0).astype(np.int)
        map = label(1 - map, connectivity=1)
        idx = map[cx, cy]
        map[cx-20:cx+20, cy-20:cy+20] = idx - 1
        yy, xx = np.meshgrid(np.arange(self.known_map.shape[1]), np.arange(self.known_map.shape[0]))
        xx = xx[(self.known_map == 0) & (map == idx)].reshape(-1)
        yy = yy[(self.known_map == 0) & (map == idx)].reshape(-1)
        dis = np.square(xx - x) + np.square(yy - y)
        # dis[dis <= 9] = 100000
        if len(dis) > 0:
            # nearest = dis.argmin()
            nearest = random.randint(0, len(dis) - 1)
            pos = (xx[nearest], yy[nearest])
        else:
            pos = (0, 0)
        self.target_obj = False
        self.local_goal = self.map2pos(pos)

    def reachable(self):
        map = self.get_occupancy_map()
        map = (self.conv2d(map, kernel=5) > 0).astype(np.int)
        map = label(1 - map, connectivity=1)
        cx, cy = self.pos2map(self.position)
        idx = map[cx, cy]
        tx, ty = self.pos2map(self.local_goal)
        return map[tx, ty] == idx

    def check_goal(self):
        pos_l, num = label(self.seg_pos_map & (1 - self.tried_pos_map), return_num=True)
        pos: List[np.ndarray] = [None] * (num + 1)
        yy, xx = np.meshgrid(np.arange(self.seg_pos_map.shape[1]), np.arange(self.seg_pos_map.shape[0]))
        for i in range(1, num + 1):
            # print(xx.shape, pos_l.shape, i)
            x = xx[pos_l == i].mean()
            y = yy[pos_l == i].mean()
            pos[i] = self.map2pos((x, y))
        for i in range(1, num + 1):
            if np.linalg.norm(pos[i][[0, -1]] - self.position[[0, -1]]) <= 1.5:
                angle = self.get_angle_delta(self.forward[[0, -1]], pos[i][[0, -1]] - self.position[[0, -1]])
                if np.abs(angle) < 20:
                    # if self.torso_cnt:
                    #     x, z = self.pos2map(pos[i])
                    #     self.tried_pos_map[x-5:x+5, z-5:z+5] = 1
                    # self.torso_cnt += 1
                    x, z = self.pos2map(pos[i])
                    self.tried_pos_map[x-5:x+5, z-5:z+5] = 1
                    self.find_goal()
                    return 5
        for i in range(1, num + 1):
            if np.linalg.norm(pos[i][[0, -1]] - self.position[[0, -1]]) <= 1.5:
                angle = self.get_angle_delta(self.forward[[0, -1]], pos[i][[0, -1]] - self.position[[0, -1]])
                if angle > 0:
                    return 1
                else:
                    return 2

    def get_occupancy_map(self):
        return self.occupancy_map

    def get_dest(self, target):
        x, y = target
        map = self.get_occupancy_map()
        map = label(1 - map, connectivity=1)
        cx, cy = self.pos2map(self.position)
        idx = map[cx, cy]
        yy, xx = np.meshgrid(np.arange(map.shape[1]), np.arange(map.shape[0]))
        xx = xx[map == idx].reshape(-1)
        yy = yy[map == idx].reshape(-1)
        nearest = np.argmin(np.square(xx - x) + np.square(yy - y))
        return xx[nearest], yy[nearest]

    def find_shortest_path(self, goal=None):
        st = self.pos2map(self.position)
        if goal is None:
            goal = self.pos2map(self.local_goal)
        map = self.get_occupancy_map()
        dist_map = np.ones_like(map, dtype=np.float32)
        for i in reversed(range(3, 25, 2)):
            dist_map[self.conv2d(map, kernel=i) > 0] = 25**((25-i)//2)
        dist_map[map > 0] = 25**12
        dist_map[goal[0] - 5 : goal[0] + 5, goal[-1] - 5 : goal[-1] + 5] = np.minimum(dist_map[goal[0] - 5 : goal[0] + 5, goal[-1] - 5 : goal[-1] + 5], 100)
        path = pyastar2d.astar_path(dist_map, tuple(st), self.get_dest(goal))
        return path

    def move(self):
        if not self.target_obj and (not self.reachable() or np.linalg.norm(self.local_goal - self.position[[0, -1]]) < 2):
            self.find_goal()
        self.path = self.find_shortest_path()
        angle = self.get_angle_delta(self.forward[[0, 2]], self.path[min(4, len(self.path) - 1)] - self.path[0])
        self.log('follow shortest path')
        if np.abs(angle) < ANGLE:
            return 0
        elif angle > 0:
            return 1
        else:
            return 2

    def choose_action(self):
        if self.last_agent is not None:
            if self.last_action in [1, 2] and np.abs(self.last_agent - self.obs['agent'])[[3, 5]].sum() < 1e-3:
                self.move_fail_cnt += 1
                self.log(f'turn fail')
                return 0
            elif self.last_action == 0 and np.abs(self.last_agent - self.obs['agent'])[[0, 2]].sum() < 0.01:
                self.log(f'move fail {np.abs(self.last_agent - self.obs["agent"])[[0, 2]].sum()}')
                self.move_fail_cnt += 1
                if self.move_fail_cnt >= 3:
                    self.move_fail_cnt = 0
                    x, y = self.pos2map(self.obs['agent'][[0, 2]] + 0.3 * self.obs['agent'][[3, 5]])
                    self.move_failed_map[x-1:x+2, y-1:y+2] = 1
                return random.randint(1, 2)
            else:
                self.move_fail_cnt = 0
        check_result = self.check_goal()
        if check_result is not None:
            return check_result
        return self.move()

    def act(self, obs):
        self.log(f'===========\nstep {self.step}')
        self.step += 1
        self.obs = obs
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        self.dep2map()

        # if self.initial_move_cnt < 12:
        #     self.initial_move_cnt += 1
        #     self.last_action = 1
        # else:
        self.last_action = self.choose_action()
        if self.last_agent is not None:
            lp = self.pos2map(self.last_agent[:3])
            np = self.pos2map(self.position)
            line = bresenhamline(lp[None], np[None], max_iter=-1)
            self.visited_map[line[:, 0], line[:, 1]] = 1
        self.last_agent = self.obs["agent"]
        if self.last_action not in [3, 4]:
            self.torso_cnt = 0
        self.log(f'act {self.last_action}')
        return self.last_action

    def plot(self):
        map = (1 - self.get_occupancy_map()) * 255
        map = map.astype(np.uint8)
        map[(map == 255) & (self.known_map == 1)] = 180
        map = np.repeat(map[..., np.newaxis], 3, axis=-1)
        map[self.visited_map == 1] = [0, 255, 0]
        # x, z = self.pos2map(self.target)
        # map[x-1:x+1, z-1:z+1] = [255, 0, 0]
        # x, z = self.pos2map(self.goal)
        # map[x-1:x+1, z-1:z+1] = [0, 0, 255]
        return map


class H_agent_mp:
    @staticmethod
    def child(pipe, device, learned):
        agent = H_agent(device, learned)
        while True:
            cmd, args = pipe.recv()
            if cmd == 'reset':
                agent.reset(*args)
            elif cmd == 'act':
                pipe.send(agent.act(args))
            elif cmd == 'plot':
                pipe.send(agent.plot())
            elif cmd == 'seg_mask':
                pipe.send(agent.seg_mask)
    def __init__(self, n, learned='all'):
        self.pipes = []
        for i in range(n):
            p1, p2 = Pipe()
            proc = Process(target=self.child, args=(p2, i % 2, learned))
            proc.start()
            self.pipes.append(p1)
    def reset(self, i, *args):
        self.pipes[i].send(('reset', args))
    def step(self, obs):
        for p, o in zip(self.pipes, obs):
            p.send(('act', o))
        return [p.recv() for p in self.pipes]
    def plot(self, i):
        self.pipes[i].send(('plot', None))
        return self.pipes[i].recv()
    def get_seg_mask(self):
        for p in self.pipes:
            p.send(('seg_mask', None))
        return [p.recv() for p in self.pipes]
