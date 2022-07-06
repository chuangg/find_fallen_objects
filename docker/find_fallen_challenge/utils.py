
import requests
import json
def create_tdw(port):
    url = "http://localhost:5000/get_tdw"
    data = {
        'ip_address': "localhost",
        'port': port
    }
    response = requests.post(url, json=json.dumps(data))
    print(response.status_code, response.reason)
    docker_id = response.json()['docker_id']
    return docker_id


def kill_tdw(docker_id):
    url = "http://localhost:5000/kill_tdw"
    data = {
        "container_id": docker_id
    }
    response = requests.post(url, json=json.dumps(data))
    print(response.status_code, response.reason)