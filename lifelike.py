import time
import json
from pathlib import Path

import numpy as np
import genesis as gs

class Env:
    def __init__(self):
        self.initialize_scene()
        self.add_entities()
        self.scene.build()

    def initialize_scene(self):
        gs.init(backend=gs.gpu)
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=0.01,
            ),
            show_viewer=True,
            rigid_options=gs.options.RigidOptions(
                dt=0.02,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
        )

    def add_entities(self):
        self.cam = self.scene.add_camera(
            res=(1280, 960),
            pos=(3.5, 0.0, 2.5),
            lookat=(0, 0, 0.5),
            fov=30,
            GUI=False
        )

        self.plane = self.scene.add_entity(gs.morphs.Plane())

        self.agibot = self.scene.add_entity(
            gs.morphs.URDF(
                file='lifelike-agility-and-play/src/lifelike/sim_envs/pybullet_envs/legged_robot/data/urdf/max.urdf',
                pos=(1.0, 1.0, 0.5),
                euler=(0, 0, 0),
            ),
        )


    def load_mocap_data(self, file_path):
        data = json.load(open(file_path))
        leg_order = data["LegOrder"]
        self.frames = data["Frames"]
        jnt_names = []
        for leg in leg_order:
            for i in range(3):
                jnt_names.append(f'joint_{leg}{i+1}')
        self.dofs_idx = [self.agibot.get_joint(name).dof_idx_local for name in jnt_names]
        self.jnt_names = jnt_names
        return self.frames

    def step(self, frame):
        self.agibot.set_dofs_position(frame[7:], self.dofs_idx)
        self.scene.step()


def main():
    env = Env()
    file_path = "lifelike-agility-and-play/data/mocap_data/dog_fast_run_02_004_ret_mir.txt"
    frames = env.load_mocap_data(file_path)
    for frame in frames:
        env.step(frame)
    

if __name__ == "__main__":
    main()