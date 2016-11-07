import os
import sys
import h5py
import imread
from glob import glob
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

raw_files = sorted(glob(os.path.join(sys.argv[1], '*.png')))
raw = np.stack([imread.imread(im) for im in raw_files], axis=0)

gt_files = sorted(glob(os.path.join(sys.argv[2], '*.png')))
gt = np.stack([imread.imread(im) for im in gt_files], axis=0)

f = h5py.File('training_data.h5')
f.create_dataset('raw', data=raw)
f.create_dataset('gt', data=gt)

# create distance field (from background, with synapses as background)
# sampling is 30 nm in Z, 4 in X,Y
distance = distance_transform_edt(gt == 0, (30, 4, 4))
f.create_dataset('distance', data=distance)
