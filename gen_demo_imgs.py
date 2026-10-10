"""Genera imágenes de demo a partir de brownian_motion_image.py

Uso:
    python gen_demo_imgs.py
    python gen_demo_imgs.py --dir demo --size 256
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("brownian_motion_image.py")
HURST_VALUES = [.1, .2, .3, .4, .5, .6, .7, .8, .9]


def main():
    parser = argparse.ArgumentParser(description="Genera imágenes fBm de demo.")
    parser.add_argument("--dir", type=str, default="demo_imgs", help="Directorio de salida.")
    parser.add_argument("--size", type=int, default=256, help="Tamaño (cuadrado) de imagen")
    args = parser.parse_args()

    for hurst in HURST_VALUES:
        cmd = [sys.executable, str(SCRIPT),
               "--hurst", str(hurst),
               "--dir", args.dir,
               "--count", "100",
               "--seed", "5467",
               "--width", str(args.size),
               "--height", str(args.size)]

        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
