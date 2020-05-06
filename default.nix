# https://sid-kap.github.io/posts/2018-03-08-nix-pipenv.html
with import <nixpkgs> {};
let
  manyLinuxFile =
    writeTextDir "_manylinux.py"
      ''
        print("in _manylinux.py")
        manylinux1_compatible = True
      '';
in
buildFHSUserEnv {
  name = "python-env";
  targetPkgs = pkgs: with pkgs; [
    python3
    pipenv
    which
    gcc
    binutils
    git
    gitRepo
    gnupg
    autoconf
    curl
    procps
    gnumake
    utillinux
    m4
    gperf
    unzip
    cudaPackages.cudatoolkit_10_0
  	cudnn.generic
    linuxPackages.nvidia_x11
    libGLU
    libGL
    xorg.libXi xorg.libXmu freeglut
    xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
    ncurses5
    stdenv.cc
    binutils

    # All the C libraries that a manylinux_1 wheel might depend on:
    ncurses
    xorg.libX11
    xorg.libXext
    xorg.libXrender
    xorg.libICE
    xorg.libSM
    glib
  ];

  multiPkgs = pkgs: with pkgs; [ zlib ];
  profile = ''
    export PYTHONPATH=${manyLinuxFile.out}:/usr/lib/python3.7/site-packages
    export CUDA_PATH=${pkgs.cudatoolkit}
    # export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
    export EXTRA_CCFLAGS="-I/usr/include"
  '';

  runScript = "pipenv shell";
}
