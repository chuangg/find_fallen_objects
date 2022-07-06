import sys

import numpy as np
import pyastar
from magnebot import ActionStatus as TaskStatus
from magnebot import Arm
from magnebot.action_status import ActionStatus
from multimodal_challenge.multimodal import MultiModal
from scipy.signal import convolve2d
from tdw.output_data import OutputData, Categories
from tdw.tdw_utils import TDWUtils


# from ppo.sound_texture import Sound, texture_stats, subband_envs


class Basic_controller(MultiModal):
    '''
    Currently a toy(uncompleted) controller just for testing
    '''

    def __init__(self, port: int = 1071, screen_size: int = 300, fov=90):
        super().__init__(port, screen_width=screen_size, screen_height=screen_size)
        self.screen_size = screen_size
        self.FOV = fov
        self.port = port

    def init_scene(self, type: str, scene: str, layout: int, trial: int = None):
        sys.stdout.flush()
        super().init_scene(type, scene, layout, trial)
        self.reward = 0
        resp = self.communicate([{"$type": "set_field_of_view", "field_of_view": self.FOV, "avatar_id": "a"},
                                 {"$type": "set_pass_masks", "pass_masks": ["_img", "_depth", '_id'], "avatar_id": "a"},])
        self.category_map = {}
        self._end_action()
        print('init scene complete')
        self.audio_pad = np.pad(np.frombuffer(self.audio, dtype=np.uint8), (0, 12582912 - len(self.audio)),
                                constant_values=1)  # wav files only have last byte 0 or 255

    def _move_forward(self, distance=0.2):
        return self.move_by(distance, arrived_at=0.05)

    def _turn(self, left=True, angle=15):
        if left:
            angle = -angle
        else:
            angle = angle

        return self.turn_by(angle=angle, aligned_at=2)

    def _set_torso(self, slide_up=True, dis=0.5):
        if slide_up:
            dis += 1
        return self.set_torso(dis)

    def _cal_dis_diff(self, p1, p2, f1, f2):
        # p1: before step; p2: after step
        if self.l2_distance(p1, p2) < 1e-2:
            return 0
        p_target = self.state.object_transforms[self.target_object_id].position
        p1_dis = self.l2_distance(p_target, p1)  # distance between target and a position
        p2_dis = self.l2_distance(p_target, p2)
        # TODO: facing direction
        f1_target = self.state.object_transforms[self.target_object_id].position - p1
        f1_dis = 1 - self.cosine_similarity(f1, f1_target)
        f2_target = self.state.object_transforms[self.target_object_id].position - p2
        f2_dis = 1 - self.cosine_similarity(f2, f2_target)
        # diff = (p1_dis + f1_dis) - (p2_dis + f2_dis)
        diff = p1_dis - p2_dis
        return 1 if diff > 0 else 0 if diff == 0 else -1
        # return 1 if p1_dis > p2_dis else 0 if p1_dis == p2_dis else -1

    def _cal_path_diff(self, p1, p2):
        # p1: before step; p2: after step
        p_target = self.state.object_transforms[self.target_object_id].position
        info = self._info(self)
        occ_map = self.occupancy_map
        p1_path = self.find_shortest_path(p1, p_target, occ_map, info)
        p2_path = self.find_shortest_path(p2, p_target, occ_map, info)
        return len(p1_path) - len(p2_path)

    def conv2d(self, map, kernel=3):
        conv = np.ones((kernel, kernel))
        return convolve2d(map, conv, mode='same', boundary='fill')

    def get_idx(self, x, z, _scene_bounds):
        x_min, x_max = _scene_bounds["x_min"], _scene_bounds["x_max"]
        map_width = self.occupancy_map.shape[0]
        CELL_SIZE = (abs(x_min) + abs(x_max)) / map_width
        i = int((x - _scene_bounds["x_min"]) / CELL_SIZE)
        j = int((z - _scene_bounds["z_min"]) / CELL_SIZE)
        return i, j

    def find_shortest_path(self, st, goal, map, info):
        # map = self.env.controller.occupancy_map
        # map: 0-free, 1-occupied
        st_x, _, st_z = st
        g_x, _, g_z = goal
        st_i, st_j = self.get_idx(st_x, st_z, info['_scene_bounds'])
        g_i, g_j = self.get_idx(g_x, g_z, info['_scene_bounds'])
        dist_map = np.ones_like(map, dtype=np.float32)
        super_map1 = self.conv2d(map, kernel=5)
        dist_map[super_map1 > 0] = 10
        super_map2 = self.conv2d(map)
        dist_map[super_map2 > 0] = 1000
        dist_map[map > 0] = 100000
        path = pyastar.astar_path(dist_map, (st_i, st_j), (g_i, g_j), allow_diagonal=False)
        return path

    def check_arrival(self, threshold=1.):
        p_bot = self.state.magnebot_transform.position
        p_target = self.state.object_transforms[self.target_object_id].position
        dis = self.l2_distance(p_bot, p_target)
        if dis <= 1:  # arrive goal
            return True
        else:
            return False

    def _check_success(self, threshold=3):  # TODO: decide whether client's claim of finding the object is correct
        color = self.objects_static[self.target_object_id].segmentation_color
        seg_mask = np.array(self.state.get_pil_images()['id'])
        where = (color == seg_mask).all(axis=-1)
        if not where.any():
            return ActionStatus.failure
        dis = np.linalg.norm(self.state.object_transforms[self.target_object_id].position - self.state.magnebot_transform.position)
        # print(dis, threshold)
        if dis < threshold:
            return ActionStatus.success
        else:
            return ActionStatus.failure

    def _go_to(self, object):
        '''
        go to an object based on an object id
        '''
        p1 = self.state.object_transforms[object].position
        p2 = self.state.magnebot_transform.position
        d = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) ** 0.5

        if d > 2:
            return TaskStatus.failed_to_move
        status = self.turn_to(int(object))
        status = self.move_to(int(object), arrived_at=0.33)
        status = self.reset_arm(Arm.right)
        status = self.grasp(int(object), Arm.right)
        return status

    def get_object_pos(self, id):
        x, y, z = self.state.object_transforms[id].position
        _, i, j, _ = self.check_occupied(x, z)
        return i, j

    def _obs(self):
        obs = {}
        obs['rgb'] = np.array(self.state.get_pil_images()['img'])
        # obs['seg_mask'] = np.array(self.state.get_pil_images()['id'])
        # raw_category = np.array(self.state.get_pil_images()['category'])
        # category = np.zeros_like(raw_category)
        # for k, v in self.category_map.items():
        #     category[np.all(raw_category == k, axis=-1)] = v
        # obs['category'] = category
        obs['depth'] = TDWUtils.get_depth_values(np.flip(self.state.images['depth'], 0), width=self.screen_size,
                                                 height=self.screen_size)
        obs['audio'] = self.audio_pad
        x, y, z = self.state.magnebot_transform.position
        fx, fy, fz = self.state.magnebot_transform.forward
        obs['agent'] = np.array([x, y, z, fx, fy, fz])  # position and forward vector of the agent
        # visible_objects = self.get_visible_objects()
        # obs['visible_objects'] = [{
        #     'id': o,
        #     # 'type': self.get_object_type(o),
        #     'seg_color': self.objects_static[o].segmentation_color} for o in visible_objects]
        # for _ in range(len(visible_objects), 20):
        #     obs['visible_objects'].append({'object_id': None, 'type': None, 'seg_color': None})
        obs['FOV'] = self.FOV
        obs['camera_matrix'] = np.array(self.state.camera_matrix)
        return obs

    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[1] - g[1]) ** 2 + (st[2] - g[2]) ** 2) ** 0.5

    def cosine_similarity(self, a, b):
        return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_object_type(self, id):
        if id in self.target_objects:
            return 0
        if id in self.containers:
            return 1
        if self.objects_static[id].category == 'bed':
            return 2
        return 3

    def _info(self, env):
        '''
        return the env's info
        '''
        info = {}
        info['_scene_bounds'] = self._scene_bounds
        return info
