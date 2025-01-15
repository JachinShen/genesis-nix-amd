import time
import json
from pathlib import Path

import numpy as np
import genesis as gs

import torch
import torch.nn as nn
import torch.nn.functional as F

def reset_hs_func(hs_len=0, z_len=0):
    return np.concatenate([np.zeros(shape=(hs_len-z_len,)), np.random.randn(z_len)])


class PMCConfig(object):
    def __init__(self, ob_space, ac_space, **kwargs):
        # logical settings
        self.test = False  # negate is_training
        self.batch_size = None

        # network architecture related
        self.use_lstm = False
        self.use_value_head = False  # for RL training/testing
        self.use_loss_type = 'none'  # {'rl' | 'none'}
        self.use_self_fed_heads = False
        self.rms_momentum = 0.0001  # rms_v2 momentum
        self.main_activation_func = 'relu'
        self.append_hist_a = False

        # lstm settings
        self.nrollout = None
        self.rollout_len = 1
        self.hs_len = 64 * 3  # for separate pi, vf, hs_z_enc or z_prev
        self.nlstm = 32
        self.z_len = 8
        self.reset_hs_func = reset_hs_func
        self.norm_z = True
        self.forget_bias = 1.0  # usually it's 1.0
        self.lstm_duration = 4  # this is for k_lstm
        self.lstm_dropout_rate = 0.0
        self.lstm_cell_type = 'lstm'
        self.lstm_layer_norm = True

        # z settings
        self.z_len = 8
        self.num_embeddings = 256
        self.norm_z = False
        self.bot_neck_z_embed_size = 64
        self.bot_neck_prop_embed_size = 64
        self.z_prior_type = 'VQ'

        # hyper-parameters
        self.alpha = 0.95
        if self.z_prior_type == 'Gaussian':
            self.z_logvar_prior = 0.0
        elif self.z_prior_type == 'TemporalGaussian':
            self.z_logvar_prior = np.log(1 - self.alpha ** 2)
        else:
            pass
        self.logstd_init = -2.0

        # value head related
        self.n_v = 1
        self.reward_weights = None
        self.lam = 0.95

        # ppo related
        self.clip_range = 0.1
        self.clip_range_lower = 0.1

        # common embedding settings
        self.embed_dim = 256

        # regularization settings
        self.weight_decay = None  # None means not use. use value, e.g., 0.0005

        # loss related settings
        self.sync_statistics = None

        # finally, cache the spaces
        self.ob_space = ob_space
        self.ac_space = ac_space

        self.conditional = False
        self.total_timesteps = None
        # parameter for weight decay
        self.z_prior_param1 = 0.003
        self.z_prior_param2 = 6 * 10e8
        self.z_prior_param3 = 0.2

        # allow partially overwriting
        for k, v in kwargs.items():
            if k not in self.__dict__:
                logging.info(
                    'unrecognized config k: {}, v: {}, ignored'.format(k, v))
            self.__dict__[k] = v

        # consistency check
        if self.use_lstm:
            assert self.batch_size is not None, 'lstm requires a specific batch_size.'
            self.nrollout = self.batch_size // self.rollout_len
            assert (self.rollout_len % self.lstm_duration == 0 or self.rollout_len == 1)
            if self.lstm_cell_type == 'k_lstm':
                if self.z_prior_type == 'Gaussian':
                    assert self.hs_len == 3 * 2 * self.nlstm + 1
                elif self.z_prior_type == 'TemporalGaussian':
                    assert self.hs_len - self.z_len == 2 * 2 * self.nlstm + 1
                else:
                    raise NotImplementedError
            elif self.lstm_cell_type == 'lstm':
                if self.z_prior_type == 'Gaussian':
                    assert self.hs_len == 3 * 2 * self.nlstm
                elif self.z_prior_type == 'TemporalGaussian':
                    assert self.hs_len - self.z_len == 2 * 2 * self.nlstm
                elif self.z_prior_type == 'VQ':
                    assert self.hs_len == 3 * 2 * self.nlstm
                else:
                    raise NotImplementedError
            else:
                raise NotImplemented('Unknown lstm_cell_type {}'.format(
                    self.lstm_cell_type))

        # activate func
        # if self.main_activation_func == 'relu':
        #     self.main_activation_func_op = tf.nn.relu
        # elif self.main_activation_func == 'tanh':
        #     self.main_activation_func_op = tf.nn.tanh
        # else:
        #     raise NotImplementedError

        # update reset_hs_func args
        if self.reset_hs_func is not None:
            self.reset_hs_func.__defaults__ = (self.hs_len, self.z_len)

class Encoder(nn.Module):
    def __init__(self, nc):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(nc.ob_space.shape[0], nc.embed_dim)
        self.fc2 = nn.Linear(nc.embed_dim, nc.embed_dim)
        self.fc3 = nn.Linear(nc.embed_dim, nc.z_len)
        self.fc4 = nn.Linear(nc.embed_dim, nc.z_len)
        self.activation = nn.ReLU() if nc.main_activation_func == 'relu' else nn.Tanh()

    def forward(self, x):
        embed = self.activation(self.fc1(x))
        embed = self.activation(self.fc2(embed))
        out_mean = self.fc3(embed)
        out_logvar = self.fc4(embed)
        return out_mean, out_logvar

class Decoder(nn.Module):
    def __init__(self, nc):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(nc.z_len, nc.embed_dim)
        self.fc2 = nn.Linear(nc.embed_dim, nc.embed_dim)
        self.fc3 = nn.Linear(nc.embed_dim, 12)
        self.activation = nn.ReLU() if nc.main_activation_func == 'relu' else nn.Tanh()

    def forward(self, x):
        embed = self.activation(self.fc1(x))
        embed = self.activation(self.fc2(embed))
        out = self.fc3(embed)
        return out

class ValueFunction(nn.Module):
    def __init__(self, nc):
        super(ValueFunction, self).__init__()
        self.fc1 = nn.Linear(nc.ob_space.shape[0], nc.embed_dim)
        self.fc2 = nn.Linear(nc.embed_dim, nc.embed_dim)
        self.fc3 = nn.Linear(nc.embed_dim, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        vf = self.fc3(x)
        return vf

class PMCNet(nn.Module):
    def __init__(self, nc):
        super(PMCNet, self).__init__()
        self.encoder = Encoder(nc)
        self.decoder = Decoder(nc)
        self.vf = ValueFunction(nc)
        self.logstd = nn.Parameter(torch.zeros(1, 12) + nc.logstd_init)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        head = self.decoder(z)
        vf = self.vf(x)
        return head, vf, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Env:
    def __init__(self):
        self.initialize_scene()
        self.add_entities()
        self.init_model()
        self.scene.build()

    def init_model(self):
        self.nc = PMCConfig(ob_space=torch.randn(12), ac_space=torch.randn(12))# 假设 PMCConfig 已经定义
        self.model = PMCNet(self.nc)

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

    def step_vis(self, frame):
        self.agibot.set_dofs_position(frame[7:], self.dofs_idx)
        self.scene.step()

    def step(self, frame):
        with torch.no_grad():
            x = torch.randn(1, self.nc.ob_space.shape[0])
            head, vf, mu, logvar = self.model(x)
        self.agibot.control_dofs_force(head[0], self.dofs_idx)
        self.scene.step()


def main():
    env = Env()
    file_path = "lifelike-agility-and-play/data/mocap_data/dog_fast_run_02_004_ret_mir.txt"
    frames = env.load_mocap_data(file_path)
    for frame in frames:
        env.step(frame)
    

if __name__ == "__main__":
    main()