# Genesis environment in NixOS

[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) is a generative world for general-purpose robotics & embodied AI learning. This repository contains the NixOS configuration for the Genesis environment.

## Installation

[devenv](https://devenv.sh/) is used to manage the environment. To activate it, run:
```bash
devenv shell
```

## Example

```bash
python3 hello.py
```

You should see a window with a robot.

## Note

This repository is developed on AMD Raedon RX 6750 XT so `HSA_OVERRIDE_GFX_VERSION` is set to `10.3.0` to make Torch work. If you are using a different GPU, you may need to adjust this setting.