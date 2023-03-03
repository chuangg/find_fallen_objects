import socket
import pickle
import random
import io
import argparse

def send(soc: socket.socket, data: bytes):
    soc.sendall(bytes(f"{len(data):<{15}}", 'utf-8'))
    soc.sendall(data)

def receive(soc: socket.socket):
    length = int(soc.recv(15).decode('utf-8'))
    buffer = io.BytesIO()
    while buffer.getbuffer().nbytes < length:
        buffer.write(soc.recv(4096))
    return pickle.loads(buffer.getvalue())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hostname', type=str, default="localhost", help='Hostname to connect the interface')
    parser.add_argument('--port', type=int, default=2590, help='Port number to connect the interface')
    args = parser.parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soc:
        soc.connect((args.hostname, args.port))
        send(soc, pickle.dumps({"op": "reset"}))
        obs, info = receive(soc)
        send(soc, pickle.dumps({"op": "observation_space"}))
        observation_space = receive(soc)
        send(soc, pickle.dumps({"op": "action_space"}))
        action_space = receive(soc)
        for i in range(10):

            send(soc, pickle.dumps({"op": "step", "action": random.randint(1, 6)}))
            result = receive(soc)

            obs, reward, done, info = result
