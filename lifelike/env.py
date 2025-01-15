import numpy as np
import genesis as gs


class Env:
    def __init__(self):
        self.initialize_scene()
        self.add_entities()
        # self.init_model()
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

    def setup_joints(self, jnt_names):
        self.dofs_idx = [self.agibot.get_joint(name).dof_idx_local for name in jnt_names]
        self.jnt_names = jnt_names

    def step(self):
        self.scene.step()

    def set_dofs_position(self, positions):
        self.agibot.set_dofs_position(positions, self.dofs_idx)

    def control_dofs_force(self, forces):
        self.agibot.control_dofs_force(forces, self.dofs_idx)

    # def step_vis(self, frame):
    #     self.agibot.set_dofs_position(frame[7:], self.dofs_idx)
    #     self.scene.step()

    # def step(self, frame):
    #     with torch.no_grad():
    #         x = torch.randn(1, self.nc.ob_space.shape[0])
    #         head, vf, mu, logvar = self.model(x)
    #     self.agibot.control_dofs_force(head[0], self.dofs_idx)
    #     self.scene.step()
