import h5py
import imread
from glob import glob
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt

###################################################
# ECS1 membranes
###################################################
raw_files = sorted(glob('data_ecs/grayscale_maps/*.png'))

# ignore first file because it's not labeled, and move to grayscale
raw = np.stack([imread.imread(im) for im in raw_files], axis=0)[1:, ..., 0]
print(raw.shape)

# ignore first file because it's not labeled, and move to grayscale
gt = h5py.File('data_ecs/seg_groundtruth.h5', 'r')['main'][1:, ...]
print(gt.shape)

# slice, X, Y, C
membranes = np.zeros(gt.shape + (3,), dtype=np.float32)

for idx in range(membranes.shape[0]):
    mem2d = gt[idx, ...]
    borders = ndi.minimum_filter(mem2d, 3) != ndi.maximum_filter(mem2d, 3)
    borders = borders | (mem2d == 0)
    membranes[idx, ..., 0] = borders
    # no synapse information, 0.5 = don't care
    membranes[idx, ..., 1] = 0.5
    membranes[idx, ..., 2] = 0.5

f = h5py.File('training_data/ecs1_membrane_data.h5', 'w')
f.create_dataset('raw', data=raw)
f.create_dataset('gt', data=membranes)

###################################################
# ECS1 synapses - different volume
###################################################
raw_files = sorted(glob('data_ecs/synapse_detection_images/*.png'))
raw = np.stack([imread.imread(im) for im in raw_files], axis=0)
print(raw.shape)


f = h5py.File('data_ecs/ecs_synapse_gt.h5', 'r')
num_slices = f['stack'].shape[0]

gt = np.zeros(shape=f['stack'].shape + (3,), dtype=np.float32)
# fill membrane channel with 0.5 = no data / don't care
gt[..., 0] = 0.5

for idx in range(num_slices):
    labeled = f['stack'][idx, ...]
    assert labeled.max() <= 2
    assert labeled.min() == 0

    chan1 = (labeled == 1)
    chan2 = (labeled == 2)

    # ignore pixels around synapse, but not on the other side of the pre/post
    # density
    for cidx, ch1, ch2 in [(1, chan1, chan2), (2, chan2, chan1)]:
        ignore = distance_transform_edt(~ch1) < 35
        ignore[ch1 + ch2] = 0
        gt[idx, ..., cidx] = ch1.astype(float) + 0.5 * ignore

f = h5py.File('training_data/ecs1_synapse_data.h5', 'w')
f.create_dataset('raw', data=raw)
f.create_dataset('gt', data=gt)
