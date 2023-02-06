import docker
import docker.errors
from gym.core import Env
import os
import socket
import time
import io
import pickle
import atexit 

client = docker.from_env()
class FindFallen(Env):
    def __init__(self, display=':4', split='train', port=2590, max_steps=200, rank=0, world_size=1):
        print(display, split, port, max_steps, rank, world_size)

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print('current port:', port)
        self.socket.connect(("localhost", port))  # port

        self.observation_space = self.query({"op": "observation_space"})
        self.action_space = self.query({"op": "action_space"})

        print(f'docker connected {rank}/{world_size} {display} {port}')



    def send(self, data: bytes):
        self.socket.sendall(bytes(f"{len(data):<{15}}", 'utf-8'))
        self.socket.sendall(data)

    def receive(self):
        # print(self.container.logs())
        length = int(self.socket.recv(15).decode('utf-8'))
        buffer = io.BytesIO()
        while buffer.getbuffer().nbytes < length:
            buffer.write(self.socket.recv(4096))
        return pickle.loads(buffer.getvalue())

    def query(self, data):
        self.send(pickle.dumps(data))
        return self.receive()

    def render(self):
        pass

    def reset(self):
        # print(self.container.logs())
        return self.query({"op": "reset"})

    def step(self, action):
        # for line in self.container.logs(stream=True):
        #    print(line.strip())
        # print(self.container.logs())
        # time.sleep(3)
        return self.query({"op": "step", "action": action})


# class FindFallenDocker(Env):
#     def __init__(self, display=':4', split='train', port=2590, max_steps=200, rank=0, world_size=1):
#         print(display, split, port, max_steps, rank, world_size)
#         self.container = client.containers.run(
#             image='fallen',
#             command=f'/venv/bin/python interface.py --max_episode_steps={max_steps} --display={display} --split={split} --port={port} --rank={rank} --world_size={world_size}',
#             network='host',
#             volumes={
#                 os.path.join(os.path.abspath(os.getcwd()), 'fallen_objects_dataset'): {
#                     'mode': 'rw',
#                     'bind': '/fallen_objects_dataset',
#                 },
#                 '/tmp/.X11-unix': {
#                     'mode': 'rw',
#                     'bind': '/tmp/.X11-unix',
#                 },
#                 os.path.join(os.path.abspath(os.getcwd()), 'env_log'): {
#                     'mode': 'rw',
#                     'bind': '/env_log',
#                 },
#             },
#             auto_remove=True,
#             runtime='nvidia',
#             detach=True,
#         )
#         # input()
#         time.sleep(3) # ensure socket in docker ready
#         self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.socket.connect(("localhost", port))

#         self.observation_space = self.query({"op": "observation_space"})
#         self.action_space = self.query({"op": "action_space"})

#         print(f'docker connected {rank}/{world_size} {display} {port}')

#         self.kill_docker = lambda: self.container.kill()
#         atexit.register(self.kill_docker)

#     def send(self, data: bytes):
#         self.socket.sendall(bytes(f"{len(data):<{15}}", 'utf-8'))
#         self.socket.sendall(data)

#     def receive(self):
#         length = int(self.socket.recv(15).decode('utf-8'))
#         buffer = io.BytesIO()
#         while buffer.getbuffer().nbytes < length:
#             buffer.write(self.socket.recv(4096))
#         return pickle.loads(buffer.getvalue())
    
#     def query(self, data):
#         self.send(pickle.dumps(data))
#         return self.receive()
    
#     def render(self):
#         pass

#     def reset(self):
#         return self.query({"op": "reset"})

#     def step(self, action):
#         return self.query({"op": "step", "action": action})
    
#     def close(self):
#         try:
#             self.kill_docker()
#             atexit.unregister(self.kill_docker)
#         except docker.errors.APIError:
#             pass
    
#     def __del__(self):
#         try:
#             self.kill_docker()
#             atexit.unregister(self.kill_docker)
#         except docker.errors.APIError:
#             pass
