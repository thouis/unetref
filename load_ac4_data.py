import h5py
import imread
from glob import glob
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt

# use this imaging volume because it's *not* intensity normalized completely.
raw = np.stack(imread.imread_multi('AC4/AC4_ImageVolume.tif'), axis=0)
labels = np.stack([imread.imread(f) for f in sorted(glob('AC4/labels/*.png'))], axis=0)
synapses = np.stack(imread.imread_multi('AC4/AC4_SynTruthVolume.tif'), axis=0)

gt = np.zeros(shape=raw.shape + (3,), dtype=np.float32)

# compute membrane channel
for idx in range(raw.shape[0]):
    mem2d = labels[idx, ...]
    borders = ndi.minimum_filter(mem2d, 3) != ndi.maximum_filter(mem2d, 3)
    borders = borders | (mem2d == 0)
    gt[idx, ..., 0] = borders

# AC4 synapses only label the synaptic cleft (ECS has polarity, channels 1&2 in GT).
# Put label into both 1&2 with wide ignore region.
for idx in range(raw.shape[0]):
    syn = synapses[idx, ...] > 0
    ignore = distance_transform_edt(~syn) < 35
    # ignore includes syn
    gt[idx, ..., 1] = syn + 0.5 * (ignore - syn)
    gt[idx, ..., 2] = syn + 0.5 * (ignore - syn)


f = h5py.File('training_data/ac4_training_data.h5', 'w')
f.create_dataset('raw', data=raw)
f.create_dataset('gt', data=gt)

# --------------------------------------------------
# half res
f = h5py.File('training_data/ac4_training_data_half.h5', 'w')
f.create_dataset('raw', data=raw[:, ::2, ::2])
f.create_dataset('gt', data=gt[:, ::2, ::2, :])
