from keras.layers import Input, merge, Convolution3D, MaxPooling3D, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback
from depth_to_space import DepthToSpace3D
import keras.backend as K
import numpy as np

import datagen
import datagen_synapse
import h5py
from Eve import Eve


def maybe_print(tensor, msg, do_print=False):
    if do_print:
        return K.print_tensor(tensor, msg)
    else:
        return tensor


def weighted_mse(y_true, y_pred):
    # per batch positive fraction, negative fraction (0.5 = ignore)
    pos_mask = K.cast(y_true > 0.75, 'float32')
    neg_mask = K.cast(y_true < 0.25, 'float32')
    num_pixels = K.cast(K.prod(K.shape(y_true)[1:]), 'float32')
    pos_fracs = K.clip((K.sum(pos_mask, axis=[1, 2, 3, 4], keepdims=True) /
                        num_pixels),
                       0.01, 0.99)
    neg_fracs = K.clip((K.sum(neg_mask, axis=[1, 2, 3, 4], keepdims=True) /
                        num_pixels),
                       0.01, 0.99)

    pos_fracs = maybe_print(pos_fracs, "positive fraction")

    # chosen to sum to 1 when multiplied by their fractions, assuming no ignore
    pos_weight = maybe_print(1.0 / (2 * pos_fracs), "positive weight")
    neg_weight = maybe_print(1.0 / (2 * neg_fracs), "negative weight")

    per_pixel_weights = pos_weight * pos_mask + neg_weight * neg_mask
    per_pixel_weighted_sq_error = K.square(y_true - y_pred) * per_pixel_weights

    batch_weighted_mse = K.mean(per_pixel_weighted_sq_error,
                                axis=[1, 2, 3, 4])

    return K.mean(batch_weighted_mse)


def lle_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred))

def L1_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))



def residual_block(input, num_feature_maps, filter_size=3):
    conv_1 = BatchNormalization(axis=2, mode=2)(input)
    conv_1 = ELU()(conv_1)
    conv_1 = Convolution3D(num_feature_maps, filter_size, filter_size, filter_size,
                           border_mode='same', bias=False)(conv_1)

    conv_2 = BatchNormalization(axis=2, mode=2)(conv_1)
    conv_2 = ELU()(conv_2)
    conv_2 = Convolution3D(num_feature_maps, filter_size, filter_size, filter_size,
                           border_mode='same', bias=True)(conv_2)

    return merge([input, conv_2], mode='sum')


def unet(input, num_features, num_input_features,
         depth=3, feature_map_mul=3):
    # bring input up to internal number of features
    increase_features = Convolution3D(num_features, 1, 1, 1)(input)

    # preprocessing block
    chain1 = residual_block(increase_features, num_features)
    if depth == 0:
        return chain1

    # recurse to next terrace
    downsampled = MaxPooling3D(pool_size=(1, 2, 2))(chain1)
    nested = unet(downsampled, feature_map_mul * num_features, num_features,
                  depth=(depth - 1), feature_map_mul=feature_map_mul)
    # bring up to 4x features
    post_nested = Convolution3D(4 * num_features, 1, 1, 1)(nested)
    upsampled = DepthToSpace3D()(post_nested)

    # merge preprocessing block and nested block
    merged = merge([chain1, upsampled], mode='sum')

    # postprocessing block
    chain2 = residual_block(merged, num_features)

    # take back down to input size
    decrease_features = Convolution3D(num_input_features, 1, 1, 1)(chain2)

    # merge
    return merge([input, decrease_features], mode='sum')


class CB(Callback):
    def __init__(self, m, i, o):
        self.m = m
        self.i = i
        self.o = o

    def on_epoch_end(self, epoch, logs):
        print("saving ep{}.h5".format(epoch))
        f = h5py.File('ep{}.h5'.format(epoch), 'w')
        pred = self.m.predict(self.i)
        f.create_dataset('pred', data=pred)
        f.create_dataset('i', data=self.i)
        f.create_dataset('o', data=self.o)
        f.close()
        if epoch % 100 == 0:
            self.m.save_weights('weights_ep{}.h5'.format(epoch))


def alternate_and_extend(seqs, chan_idx, mask_dist=25):
    while True:
        for s in seqs:
            r, d = next(s)

            # mask border to avoid training near edges
            if chan_idx == 1:
                # B, C, D, W, H
                d[:, :, :, :mask_dist, :] = 0.5
                d[:, :, :, -mask_dist:, :] = 0.5
                d[:, :, :, :, :mask_dist] = 0.5
                d[:, :, :, :, -mask_dist:] = 0.5
            else:
                assert chan_idx == 3
                # B, D, W, H, C
                d[:, :, :mask_dist, :, :] = 0.5
                d[:, :, -mask_dist:, :, :] = 0.5
                d[:, :, :, :mask_dist, :] = 0.5
                d[:, :, :, -mask_dist:, :] = 0.5

            # make 3 channel: membrane, syn1, syn2
            if d.shape[chan_idx] == 1:
                # is membrane, extend with two planes of 0.5 (the ignore value)
                ignore = 0.5 * np.ones_like(d)
                yield r, np.concatenate([d, ignore, ignore], axis=chan_idx)
            elif d.shape[chan_idx] == 2:
                # is synapse, extend with one plane of 0.5 in first plane
                ign_shape = list(d.shape)
                ign_shape[chan_idx] = 1
                ignore = 0.5 * np.ones(ign_shape, dtype=d.dtype)
                yield r, np.concatenate([ignore, d], axis=chan_idx)
            else:
                raise ValueError("Unknown data size")

if __name__ == '__main__':
    if K._BACKEND == 'tensorflow':
        INPUT_SHAPE = (17, 256, 256, 1)
        OUTPUT_SHAPE = (17, 256, 256, 3)
    else:
        INPUT_SHAPE = (1, 11, 256, 256)
        OUTPUT_SHAPE = (3, 11, 256, 256)

    x = Input(shape=INPUT_SHAPE)
    first = Convolution3D(10, 3, 3, 3, border_mode='same', bias=True)(x)
    drop = Dropout(0.8)(first)
    middle = unet(drop, 10, 10, depth=3, feature_map_mul=3)
    out = Convolution3D(3, 1, 1, 1, activation='sigmoid')(middle)
    model = Model(input=x, output=out)

    assert all(o1 == o2 for o1, o2 in zip(OUTPUT_SHAPE, model.layers[-1].output_shape[1:]))

    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png', show_shapes=True)

    mem_files = ['ecs_training_data.h5', 'ac4_training_data.h5', 'ac4_training_data_half.h5']
    channel_idx = (3 if K._BACKEND == 'tensorflow' else 0)
    ecs_gen, ac4_gen, ac4_gen_half = [datagen.generate_data(f,
                                                            INPUT_SHAPE,
                                                            OUTPUT_SHAPE,
                                                            4,
                                                            channel_idx=channel_idx)
                                      for f in mem_files]
    syn_gen = datagen_synapse.generate_data('ecs_synapse_gt.h5',
                                            INPUT_SHAPE,
                                            OUTPUT_SHAPE,
                                            4,
                                            channel_idx=channel_idx)
    # emphasize ECS
    gens = [ecs_gen, syn_gen, ac4_gen, ecs_gen, syn_gen, ac4_gen_half]

    batchgen = alternate_and_extend(gens, channel_idx + 1)  # +1 for batch

    i, o = next(batchgen)
    print(o.shape)

    print("compiling")
    # model.compile(loss=weighted_mse, optimizer=SGD(lr=1e-3, momentum=0.95, clipvalue=0.5))
    opt = Eve(lr=1E-4, decay=1E-4, beta_1=0.9, beta_2=0.999, beta_3=0.999, small_k=0.1, big_K=10, epsilon=1e-08)
    model.compile(loss=weighted_mse, optimizer=opt)
    # model.compile(loss=weighted_mse, optimizer=Adam(lr=1e-4))
    model.load_weights('start.h5')
    print("fitting")
    model.fit_generator(batchgen, 512, 4000, verbose=1, callbacks=[CB(model, i, o)])
