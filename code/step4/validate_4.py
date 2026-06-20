import glob
import os
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument(
    "-s", "--datasize", type=int, required=True, help="n_compounds"
)

args = parser.parse_args()

try:
    files = glob.glob('min/*/dg_poses.dat')
    num_files = 0
    for f in files:
        if os.path.getsize(f) > 0:
            dg = np.loadtxt(f, dtype=str, usecols = (0))
            val = (dg != "NA")
            if val.any():
                num_files += 1
    
    if num_files < (0.8 * args.datasize):
        print("Not OK")
    else:
        print("OK")

except:
    print("Not OK")

