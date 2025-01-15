import time
import json
from pathlib import Path

import numpy as np
import genesis as gs

def initialize_scene():
    gs.init(backend=gs.gpu)
    scene = gs.Scene(
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
    return scene

def add_entities(scene):
    cam = scene.add_camera(
        res=(1280, 960),
        pos=(3.5, 0.0, 2.5),
        lookat=(0, 0, 0.5),
        fov=30,
        GUI=False
    )

    plane = scene.add_entity(gs.morphs.Plane())

    agibot = scene.add_entity(
        gs.morphs.URDF(
            file='lifelike-agility-and-play/src/lifelike/sim_envs/pybullet_envs/legged_robot/data/urdf/max.urdf',
            pos=(1.0, 1.0, 0.5),
            euler=(0, 0, 0),
        ),
    )

    return scene, cam, plane, agibot

def load_mocap_data(file_path, agibot):
    data = json.load(open(file_path))
    leg_order = data["LegOrder"]
    frames = data["Frames"]
    jnt_names = []
    for leg in leg_order:
        for i in range(3):
            jnt_names.append(f'joint_{leg}{i+1}')
    dofs_idx = [agibot.get_joint(name).dof_idx_local for name in jnt_names]
    return jnt_names, dofs_idx, frames

def run_simulation(scene, agibot, jnt_names, dofs_idx, frames):
    scene.build()
    for frame in frames:
        agibot.set_dofs_position(frame[7:], dofs_idx)
        scene.step()

def main():
    scene = initialize_scene()
    scene, cam, plane, agibot = add_entities(scene)
    file_path = "lifelike-agility-and-play/data/mocap_data/dog_fast_run_02_004_ret_mir.txt"
    jnt_names, dofs_idx, frames = load_mocap_data(file_path, agibot)
    run_simulation(scene, agibot, jnt_names, dofs_idx, frames)

if __name__ == "__main__":
    main()