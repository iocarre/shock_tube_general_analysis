#!/usr/bin/env python3
"""Build small validation datasets from the quiescent T1 recording.

Creates two TIFF directories under a temp root:
  quiet/    -- 160 real T1 frames, unchanged (a true-negative test).
  plume/    -- the same frames, but a faint moving Gaussian "plume" is added
               to frames 80..93 (a true-positive test with known kinematics).
Prints the temp root path on stdout.
"""
import os
import sys
import tempfile

import numpy as np
from PIL import Image

SRC = "/Users/meister/Desktop/Claude/shock_tube_image_analysis/T1"
N = 160
P_START, P_LEN = 80, 14          # plume present on frames 80..93
AMP = 60.0                        # peak brightness added (DN); faint vs 12-bit
SY, SX = 3.0, 6.0                 # Gaussian sigmas (rows, cols)
X0, VX = 200.0, 45.0              # plume centre: x = X0 + VX*(t - P_START)
Y0 = None                         # set to vertical centre at runtime


def main():
    files = sorted(f for f in os.listdir(SRC) if f.endswith(".tif"))[:N]
    root = tempfile.mkdtemp(prefix="shock_change_test_")
    qdir = os.path.join(root, "quiet")
    pdir = os.path.join(root, "plume")
    os.makedirs(qdir); os.makedirs(pdir)

    first = np.asarray(Image.open(os.path.join(SRC, files[0])))
    H, W = first.shape
    y0 = (H - 1) / 2.0
    yy, xx = np.mgrid[0:H, 0:W]

    for i, fn in enumerate(files):
        a = np.asarray(Image.open(os.path.join(SRC, fn)), dtype=np.float32)
        Image.fromarray(a.astype(np.uint16)).save(os.path.join(qdir, fn))

        b = a.copy()
        if P_START <= i < P_START + P_LEN:
            t = i - P_START
            xc = X0 + VX * t
            # brighten by a moving Gaussian, ramped up then down
            ramp = np.sin(np.pi * (t + 0.5) / P_LEN)      # 0..1..0
            g = AMP * ramp * np.exp(-(((xx - xc) ** 2) / (2 * SX ** 2)
                                      + ((yy - y0) ** 2) / (2 * SY ** 2)))
            b = b + g
        b = np.clip(b, 0, 65535).astype(np.uint16)
        Image.fromarray(b).save(os.path.join(pdir, fn))

    sys.stderr.write(
        f"frames {N}, plume frames {P_START}..{P_START+P_LEN-1}, "
        f"x: {X0:.0f}->{X0+VX*(P_LEN-1):.0f} px @ {VX} px/frame, "
        f"amp {AMP} DN over {H}x{W}\n")
    print(root)


if __name__ == "__main__":
    main()
