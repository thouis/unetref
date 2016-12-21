import h5py
import imread
from glob import glob
import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt

for cremifile in glob('CREMI/*.hdf'):
    f = h5py.File(cremifile, 'r')
    raw = f['volumes/raw']
    labels = f['volumes/labels/neuron_ids']
    synapses = f['volumes/labels/clefts']

    gt = np.zeros(shape=raw.shape + (3,), dtype=np.float32)

    # compute membrane channel
    for idx in range(raw.shape[0]):
        mem2d = labels[idx, ...]
        borders = ndi.minimum_filter(mem2d, 3) != ndi.maximum_filter(mem2d, 3)
        borders = borders | (mem2d == 0)
        gt[idx, ..., 0] = borders

    # CREMI synapses only label the synaptic cleft (ECS has polarity, channels 1&2 in GT).
    # Put label into both 1&2 with wide ignore region.
    for idx in range(raw.shape[0]):
        syn = synapses[idx, ...] < 2**64 - 1  # why did they do this?
        ignore = distance_transform_edt(~syn) < 35
        # ignore includes syn
        gt[idx, ..., 1] = syn + 0.5 * (ignore - syn)
        gt[idx, ..., 2] = syn + 0.5 * (ignore - syn)

    outname = cremifile.replace('CREMI', 'training_data')
    f = h5py.File(outname, 'w')
    f.create_dataset('raw', data=raw)
    f.create_dataset('gt', data=gt)
