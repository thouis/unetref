import os
import sys
import h5py
import imread
from glob import glob
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt

raw_files = sorted(glob('synapse_detection_images/*.png'))
raw = np.stack([imread.imread(im) for im in raw_files], axis=0)
print(raw.shape)

f = h5py.File('ecs_synapse_gt.h5')
if 'raw' in f.keys():
    del f['raw']
f.create_dataset('raw', data=raw)

num_slices = f['stack'].shape[0]
if 'gt_2channel' in f.keys():
    del f['gt_2channel']
gt = f.create_dataset('gt_2channel', shape=f['stack'].shape + (2,), dtype='float32')

for idx in range(num_slices):
    labeled = f['stack'][idx, ...]
    assert labeled.max() <= 2
    assert labeled.min() == 0

    chan1 = (labeled == 1)
    chan2 = (labeled == 2)

    # ignore pixels around synapse, but not on the other side of the pre/post
    # density
    for cidx, ch1, ch2 in [(0, chan1, chan2), (1, chan2, chan1)]:
        ignore = distance_transform_edt(~ch1) < 35
        ignore[ch1 + ch2] = 0
        gt[idx, ..., cidx] = ch1.astype(float) + 0.5 * ignore
