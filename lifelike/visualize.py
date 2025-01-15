import time
import json
from pathlib import Path

from env import Env

def load_mocap_data(file_path):
    data = json.load(open(file_path))
    leg_order = data["LegOrder"]
    frames = data["Frames"]
    jnt_names = []
    for leg in leg_order:
        for i in range(3):
            jnt_names.append(f'joint_{leg}{i+1}')
    return frames, jnt_names

def main():
    env = Env()
    file_path = "lifelike-agility-and-play/data/mocap_data/dog_fast_run_02_004_ret_mir.txt"
    frames, jnt_names = load_mocap_data(file_path)
    env.setup_joints(jnt_names)
    for frame in frames:
        env.set_dofs_position(frame[7:])
        env.step()
    

if __name__ == "__main__":
    main()