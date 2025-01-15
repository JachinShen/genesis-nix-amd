import time
import json
from pathlib import Path

import torch

from env import Env
from model import PMCNet, PMCConfig


def load_mocap_data(file_path):
    data = json.load(open(file_path))
    leg_order = data["LegOrder"]
    frames = data["Frames"]
    jnt_names = []
    for leg in leg_order:
        for i in range(3):
            jnt_names.append(f'joint_{leg}{i+1}')
    return frames, jnt_names

def init_model():
    nc = PMCConfig(ob_space=torch.randn(12), ac_space=torch.randn(12))# 假设 PMCConfig 已经定义
    model = PMCNet(nc)
    return nc, model

def main():
    env = Env()
    file_path = "lifelike-agility-and-play/data/mocap_data/dog_fast_run_02_004_ret_mir.txt"
    frames, jnt_names = load_mocap_data(file_path)
    env.setup_joints(jnt_names)
    nc, model = init_model()

    for _ in range(1000):
        with torch.no_grad():
            x = torch.randn(1, nc.ob_space.shape[0])
            head, vf, mu, logvar = model(x)
        forces = head[0].cpu()
        env.control_dofs_force(forces)
        env.step()
    

if __name__ == "__main__":
    main()