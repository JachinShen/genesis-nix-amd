{ pkgs, lib, config, inputs, ... }:

{
  # https://devenv.sh/basics/
  env.HSA_OVERRIDE_GFX_VERSION = "10.3.0";
  # env.LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib";
  env.PYOPENGL_PLATFORM = "glx"; # https://github.com/Genesis-Embodied-AI/Genesis/issues/10

  # https://devenv.sh/packages/
  packages = with pkgs; [ 
    git 
    glib
    xorg.libX11
    xorg.libXrender
    libdrm # for wayland monitor 
    libGL
    # libGLU
    # glew
    # glfw
    # freeglut
  ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    venv.enable = true;
    venv.requirements = ''
    https://mirror.sjtu.edu.cn/pytorch-wheels/rocm6.0/torch-2.3.1%2Brocm6.0-cp312-cp312-linux_x86_64.whl
    genesis-world
    '';
  };

}
