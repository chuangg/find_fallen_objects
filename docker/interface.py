import argparse
import socket
import pickle
import os
import atexit

os.environ['MULTIMODAL_DATASET'] = 'fallen_objects_dataset'
from find_fallen_challenge.tdw_gym import TDW

if __name__ == '__main__':
    os.environ['MULTIMODAL_DATASET'] = 'fallen_objects_dataset'
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_episode_steps', type=int, default=200, help='maximum number of steps before considering a case fail')
    parser.add_argument('--display', type=str, default=None, help='the X display to run TDW on, defaults to $DISPLAY')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid', 'test'], help='the split (train/valid/test) to load')
    parser.add_argument('--port', type=int, default=2590, help='the port for communicating with this interface')
    parser.add_argument('--rank', type=int, default=0, help='the starting index to load in the split')
    parser.add_argument('--world_size', type=int, default=1, help='the step of indices to load in the split')
    parser.add_argument('--tdw_port', type=int, default=-1, help='the port used for this interface to communicate with underlying TDW, defaults to <port> + 20000')
    parser.add_argument('--allow_try', type=int, default=2, help='maximum number of actions "claim" before considering a case fail')

    args = parser.parse_args()
    print(args)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soc:
        soc.bind(("", args.port))
        soc.listen()

        display = os.environ['DISPLAY'] if args.display is None else args.display
        tdw_port = args.port + 20000 if args.tdw_port == -1 else args.tdw_port
        env = TDW(display=display, split=args.split, max_step=args.max_episode_steps, port=tdw_port, rank=args.rank, world_size=args.world_size, allow_try=args.allow_try)
        atexit.register(lambda: env.close())

        conn, addr = soc.accept()
        while True:
            data = conn.recv(int(conn.recv(15).decode('utf-8')))
            op = pickle.loads(data)
            print(f'receive {op}')
            if op['op'] == 'reset':
                result = env.reset()
            elif op['op'] == 'step':
                result = env.step(op['action'])
            elif op['op'] == 'observation_space':
                result = env.observation_space
            elif op['op'] == 'action_space':
                result = env.action_space
            else:
                assert False
            data = pickle.dumps(result)
            print(len(data))
            conn.sendall(bytes(f"{len(data):<{15}}", 'utf-8'))
            conn.sendall(data)
