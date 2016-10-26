import os
import sys
import h5py
import imread
from glob import glob
import numpy as np

raw_files = sorted(glob(os.path.join(sys.argv[1], '*.png')))
raw = np.stack([imread.imread(im) for im in raw_files], axis=0)

gt_files = sorted(glob(os.path.join(sys.argv[2], '*.png')))
gt = np.stack([imread.imread(im) for im in gt_files], axis=0)

f = h5py.File('training_data.h5')
f.create_dataset('raw', data=raw)
f.create_dataset('gt', data=gt)
