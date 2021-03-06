import h5py
import numpy as np


def generate_data(path, input_shape, output_shape, batch_size, ignore_distance=35.0, channel_idx=0):
    f = h5py.File(path, 'r')
    raw_data = f['raw']
    gt_data = f['gt']
    dists = f['distance']

    # drop the channel idx
    input_shape = [d for i, d in enumerate(input_shape) if i != channel_idx]
    output_shape = [d for i, d in enumerate(output_shape) if i != channel_idx]
    raw_to_gt_offsets = [(i - o) // 2 for i, o in zip(input_shape, output_shape)]

    while True:
        batch = []
        for idx in range(batch_size):
            while True:
                lo_corner = [np.random.randint(rd - i)
                             for rd, i in zip(raw_data.shape, input_shape)]
                slices = [slice(l, l + i) for l, i in zip(lo_corner, input_shape)]
                subraw = raw_data[tuple(slices)].astype(np.float32) / 255.0

                slices = [slice(l + o, l + o + i)
                          for l, i, o in zip(lo_corner, output_shape, raw_to_gt_offsets)]
                subgt = (gt_data[tuple(slices)] > 0).astype(np.float32)
                subdist = dists[tuple(slices)].astype(np.float32)
                subgt[(subdist <= ignore_distance) & (subgt == 0)] = 0.5

                # make sure we have enough positive pixels
                if subgt.mean() > 0.015:
                    break

            # flips
            if np.random.randint(2) == 1:
                subraw = subraw[::-1, :, :]
                subgt = subgt[::-1, :, :]
            if np.random.randint(2) == 1:
                subraw = subraw[:, ::-1, :]
                subgt = subgt[:, ::-1, :]
            if np.random.randint(2) == 1:
                subraw = subraw[:, :, ::-1]
                subgt = subgt[:, :, ::-1]
            if np.random.randint(2) == 1:
                subraw = np.transpose(subraw, [0, 2, 1])
                subgt = np.transpose(subgt, [0, 2, 1])

            # random scale/shift of intensities
            scale = np.random.uniform(0.8, 1.2)
            offset = np.random.uniform(-0.2, .2)
            subraw = subraw * scale + offset

            batch.append((subraw, subgt))

        subraws, subgts = zip(*batch)

        # after stack, channel index shifts over one
        yield (np.expand_dims(np.stack(subraws, axis=0), channel_idx + 1),
               np.expand_dims(np.stack(subgts, axis=0), channel_idx + 1))
