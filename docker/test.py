import socket
import pickle
import random
import io

def send(soc: socket.socket, data: bytes):
    soc.sendall(bytes(f"{len(data):<{15}}", 'utf-8'))
    soc.sendall(data)

def receive(soc: socket.socket):
    length = int(soc.recv(15).decode('utf-8'))
    buffer = io.BytesIO()
    while buffer.getbuffer().nbytes < length:
        buffer.write(soc.recv(4096))
    return pickle.loads(buffer.getvalue())

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soc:
    soc.connect(("localhost", 2590))
    send(soc, pickle.dumps({"op": "reset"}))
    obs, info = receive(soc)
    send(soc, pickle.dumps({"op": "observation_space"}))
    observation_space = receive(soc)
    send(soc, pickle.dumps({"op": "action_space"}))
    action_space = receive(soc)
    for i in range(1000):
        print(i)
        send(soc, pickle.dumps({"op": "step", "action": random.randint(1, 6)}))
        result = receive(soc)
        print(result)
        obs, reward, done, info = result
