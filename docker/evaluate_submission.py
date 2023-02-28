import re
from math import *
import os
import json

import matplotlib.pyplot as plt
import numpy as np
cnt=0
N=0
SR=0
SPL=0
SNA=0
base_dir = os.path.join("env_log", "distractor")
for filea in os.listdir(base_dir):
    file_dir=os.path.join(base_dir, filea)


    for file in os.listdir(file_dir):
        N+=1
        cnt+=1
        f1 = open(os.path.join(file_dir, file), 'r')
        line=f1.readline()

        while line:
            if 'false' in line:
                break
            SR+=1
            user_dict = json.loads(line)
            print(filea+'/'+file)
            distance=float(user_dict["distance"])
            steps = float(user_dict["steps"])
            first_seen = float(user_dict["first_seen"])
            first_seen_dis=float(user_dict["first_seen_dis"])
            if(distance==0):
                SPL+=1
                SNA+=1
            else:
                SPL+=first_seen_dis/distance
                SNA+=first_seen/steps


            line=f1.readline()


results = {
    'cnt': cnt,
    'success_rate': SR/cnt,
    'success_rate_path_length': SPL/cnt,
    'success_rate_number_of_action': SNA/cnt
}
if os.path.exists('/results'):
    with open('/results/eval_result.json', 'w') as f:
        json.dump(results, f)
