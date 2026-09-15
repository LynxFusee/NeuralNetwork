{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    libx11
    libxi
    libxkbcommon
    libGL
    pkg-config
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (
    with pkgs;
    [
      libx11
      libxi
      libxkbcommon
      libGL
    ]
  );
}
