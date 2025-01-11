# Genesis environment in NixOS

[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) is a generative world for general-purpose robotics & embodied AI learning. This repository contains the NixOS configuration for the Genesis environment.

## Try Genesis

### Installation

Nix flake is used to manage the environment. To activate it, run:
```bash
nix develop .
```

Install python dependencies:
```bash
pip install -r requirements.txt
```

### Example

```bash
python3 hello.py
```

You should see a window with a robot.

### Note

This repository is developed on AMD Raedon RX 6750 XT so `HSA_OVERRIDE_GFX_VERSION` is set to `10.3.0` to make Torch work. If you are using a different GPU, you may need to adjust this setting.

## Run Genesis in ROS2

`flake.nix` also setup ROS2 environment. To activate it, run:

```bash
cd ros2_ws
colcon build
source install/setup.bash
ros2 run ros_genesis simulator
```

You should see a window with a robot. The joint states are published to `/genesis/{joint_name}` topic.