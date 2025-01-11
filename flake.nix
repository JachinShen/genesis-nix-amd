{
  inputs = {
    nix-ros-overlay.url = "github:lopsided98/nix-ros-overlay/master";
    nixpkgs.follows = "nix-ros-overlay/nixpkgs";  # IMPORTANT!!!
  };
  outputs = { self, nix-ros-overlay, nixpkgs }:
    nix-ros-overlay.inputs.flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nix-ros-overlay.overlays.default ];
        };
        venvDir = "./.venv";
      in {
        devShells.default = pkgs.mkShell {
          name = "ROS2 Genesis intergation";
          venvDir = venvDir;
          PYTHONPATH = "${venvDir}/lib/python3.12/site-packages";
          HSA_OVERRIDE_GFX_VERSION = "10.3.0";
          # env.LD_LIBRARY_PATH="/run/opengl-driver/lib:/run/opengl-driver-32/lib";
          PYOPENGL_PLATFORM = "glx"; # https://github.com/Genesis-Embodied-AI/Genesis/issues/10
          # LD_LIBRARY_PATH = "${pkgs.xorg.libX11}/lib:${pkgs.libGL}/lib:$LD_LIBRARY_PATH";
          LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath 
            (with pkgs; [
              stdenv.cc.cc
              glib
              xorg.libX11
              xorg.libXrender
              libdrm # for wayland monitor 
              libGL]
            )}:$LD_LIBRARY_PATH";

          packages = [
            pkgs.git
            pkgs.colcon
            (with pkgs.python312Packages; [
              python
              venvShellHook
              numpy
              (opencv4.override{ enableGtk3 = true; })
            ])
            # ... other non-ROS packages
            (with pkgs.rosPackages.humble; buildEnv {
              paths = [
                ros-core
                ament-cmake
                ament-cmake-core
                # ... other ROS packages
              ];
            })
          ];
        };
      });
  nixConfig = {
    extra-substituters = [ "https://ros.cachix.org" ];
    extra-trusted-public-keys = [ "ros.cachix.org-1:dSyZxI8geDCJrwgvCOHDoAfOm5sV1wCPjBkKL+38Rvo=" ];
  };
}
