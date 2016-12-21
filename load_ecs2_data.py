import h5py
import imread
from glob import glob
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt

###################################################
# ECS1 membranes
###################################################
raw_files = sorted(glob('ECS2/gt_images/*.png'))
raw = np.stack([imread.imread(im) for im in raw_files], axis=0)
print(raw.shape)

gtdata = h5py.File('ECS2/combined.h5', 'r')
labels = gtdata['stack']

# slice, X, Y, C
gt = np.zeros(labels.shape + (3,), dtype=np.float32)

for idx in range(labels.shape[0]):
    mem2d = labels[idx, ...]
    borders = ndi.minimum_filter(mem2d, 3) != ndi.maximum_filter(mem2d, 3)
    borders = borders | (mem2d == 0)
    gt[idx, ..., 0] = borders
    # 65535 = ignore
    gt[idx, ..., 0][mem2d == 65535] = 0.5
    # no synapse information, 0.5 = don't care
    gt[idx, ..., 1] = 0.5
    gt[idx, ..., 2] = 0.5

f = h5py.File('training_data/ecs2_membrane_data.h5', 'w')
f.create_dataset('raw', data=raw)
f.create_dataset('gt', data=gt)
