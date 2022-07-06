import sys
sys.path.append('.')

from collections import deque

import gym
from env import observation_space, action_space
import numpy as np
import torch

from baseline.planner.arguments import get_args
from env.envs import make_vec_envs

from baseline.planner.agent_explore import H_agent_mp
from tqdm import trange

from baseline.planner.module.sound_location import SoundLocationPredictor
from baseline.planner.module.sound_category import SoundCategoryPredictor

import multiprocessing

multiprocessing.set_start_method('spawn', force=True)

def main():
    args = get_args()
    num_scenes = args.num_processes
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    envs = None
    try:
        envs = make_vec_envs(args.env_name, args.num_processes, args.log_dir, device, True, spaces=(observation_space, action_space), port=args.port, displays=args.displays, split='test')
        cum_reward = np.zeros(num_scenes, dtype=float)
        episode_rewards = deque(maxlen=10)

        steps = [0] * args.num_processes
        stage = list(range(args.num_processes))

        obses, infos = envs.reset()

        agents = H_agent_mp(args.num_processes)
        location_pred = SoundLocationPredictor()
        category_pred = SoundCategoryPredictor()

        def pred_target_position(audio, agent):
            pred_delta = location_pred.predict_location(audio)
            position = agent[[0, 2]]
            forward = agent[[3, 5]]
            forward = forward / np.linalg.norm(forward)
            si, cs = forward
            rot = np.array([[cs, -si], [si, cs]])
            pred_delta = pred_delta @ rot
            return position + pred_delta

        for i in range(args.num_processes):
            target = pred_target_position(obses['audio'][i], obses['agent'][i])
            agents.reset(i, target, category_pred.predict_category(obses['audio'][i]), infos[i].get('category', None))
            print(f'{i:01d}-{stage[i]:03d}: {infos[i]["scene_info"]}')

        obses = [{k: v[t] for k, v in obses.items()} for t in range(num_scenes)]
        for j in trange(1000000):
            actions = agents.step(obses)

            obses, rewards, dones, infos = envs.step(actions)
            cum_reward += rewards.numpy()[:, 0]
            for ttt, (done, info) in enumerate(zip(dones, infos)):
                steps[ttt] += 1
                if done:
                    success = info['finish']
                    if success:
                        episode_rewards.append(info['reward'])
                        print(f'\033[32mSuccess! reward: {info["reward"]}\033[0m')
                    else:
                        episode_rewards.append(cum_reward[ttt])
                        print(f'\033[33mFailed! reward: {cum_reward[ttt]}\033[0m')
                    cum_reward[ttt] = 0
                    steps[ttt] = 0
                    stage[ttt] += args.num_processes
                    agents.reset(ttt, pred_target_position(obses['audio'][ttt], obses['agent'][ttt]), category_pred.predict_category(obses['audio'][ttt]), infos[ttt].get('category', None))
                    print(f'{ttt:01d}-{stage[ttt]:03d}: {info["scene_info"]}')
            obses = [{k: v[t] for k, v in obses.items()} for t in range(num_scenes)]
    finally:
        if envs is not None:
            envs.close()

if __name__ == "__main__":
    main()
