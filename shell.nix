let
  # We pin to a specific nixpkgs commit for reproducibility.
  # Last updated: 2025-05. Check for new commits at https://status.nixos.org.
  pkgs = import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/7c815e513adbf03c9098b2bd230c1e0525c8a7f9.tar.gz") {};

  python = pkgs.python3.override {
    self = python;
    packageOverrides = pyfinal: pyprev: {
      hilbertcurve = pyfinal.callPackage ./hilbertcurve.nix { };
    };
  };

  in pkgs.mkShell {
  packages = [
    (python.withPackages (python-pkgs: with python-pkgs; [
      moderngl
      numpy
      pillow
      pygame-ce
      pyglm
      hilbertcurve
      # ipaddress
    ]))
  ];
}