import h5py
import numpy as np


def generate_data(path, input_shape, output_shape, batch_size, ignore_distance=35.0, channel_idx=0):
    f = h5py.File(path, 'r')
    raw_data = f['raw']
    gt_data = f['gt_2channel']

    # drop the channel idx
    input_shape = [d for i, d in enumerate(input_shape) if i != channel_idx]
    output_shape = [d for i, d in enumerate(output_shape) if i != channel_idx]
    raw_to_gt_offsets = [(i - o) // 2 for i, o in zip(input_shape, output_shape)]

    while True:
        batch = []
        for idx in range(batch_size):
            lo_corner = [np.random.randint(rd - i)
                         for rd, i in zip(raw_data.shape, input_shape)]
            slices = [slice(l, l + i) for l, i in zip(lo_corner, input_shape)]
            subraw = raw_data[tuple(slices)].astype(np.float32) / 255.0

            # GT has 2 channels (last index is channels)
            slices = [slice(l + o, l + o + i)
                      for l, i, o in zip(lo_corner, output_shape, raw_to_gt_offsets)]
            subgt = gt_data[tuple(slices)]
            assert subgt.shape[-1] == 2

            # flips
            if np.random.randint(2) == 1:
                subraw = subraw[::-1, :, :]
                subgt = subgt[::-1, :, :, :]
            if np.random.randint(2) == 1:
                subraw = subraw[:, ::-1, :]
                subgt = subgt[:, ::-1, :, :]
            if np.random.randint(2) == 1:
                subraw = subraw[:, :, ::-1]
                subgt = subgt[:, :, ::-1, :]
            if np.random.randint(2) == 1:
                subraw = np.transpose(subraw, [0, 2, 1])
                subgt = np.transpose(subgt, [0, 2, 1, 3])

            # random scale/shift of intensities
            scale = np.random.uniform(0.8, 1.2)
            offset = np.random.uniform(-0.2, .2)
            subraw = subraw * scale + offset

            # theano = BCDHW
            # tf = BDHWC
            assert channel_idx in [0, 3]
            if channel_idx == 0:
                subgt = np.transpose(subgt, [3, 0, 1, 2])

            batch.append((subraw, subgt))

        subraws, subgts = zip(*batch)


        # after stack, channel index shifts over one
        yield (np.expand_dims(np.stack(subraws, axis=0), channel_idx + 1),
               np.stack(subgts, axis=0))
