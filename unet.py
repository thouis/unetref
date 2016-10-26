from keras.layers import Input, merge, Convolution3D, MaxPooling3D, UpSampling3D
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback
from crop3d import Cropping3D
import keras.backend as K
import numpy as np

import datagen
import h5py


def maybe_print(tensor, msg, do_print=False):
    if do_print:
        return K.print_tensor(tensor, msg)
    else:
        return tensor


def weighted_mse(y_true, y_pred):
    # per batch positive fraction
    pos_fracs = K.clip((K.sum(y_true, axis=[1, 2, 3, 4], keepdims=True) /
                        K.cast(K.prod(K.shape(y_true)[1:]), np.float32)),
                       0.01, 0.99)
    pos_fracs = maybe_print(pos_fracs, "positive fraction")

    # chosen to sum to 1 when multiplied by their fractions
    pos_weight = maybe_print(1.0 / (2 * pos_fracs), "positive weight")
    neg_weight = maybe_print(1.0 / (2 * (1.0 - pos_fracs)), "negative weight")

    per_pixel_weights = (pos_weight - neg_weight) * y_true + neg_weight
    per_pixel_weighted_sq_error = K.square(y_true - y_pred) * per_pixel_weights

    batch_weighted_mse = K.mean(per_pixel_weighted_sq_error,
                                axis=[1, 2, 3, 4])

    return K.mean(batch_weighted_mse)


def unet(input, num_features, depth=3, mult=2):
    # pre convolutions
    # TODO: replace with resnet module
    conv1 = Convolution3D(num_features, 3, 3, 3, activation='relu')(input)
    conv2 = Convolution3D(num_features, 3, 3, 3, activation='relu')(conv1)
    if depth == 0:
        return conv2

    downsampled = MaxPooling3D(pool_size=(1, 2, 2))(conv2)
    # recurse to next terrace
    nested = unet(downsampled, mult * num_features, depth=(depth - 1), mult=mult)
    upsampled = UpSampling3D(size=(1, 2, 2))(nested)
    upconved = Convolution3D(num_features, 1, 1, 1)(upsampled)

    # merge
    # TODO - remove cropping
    size_diff = ((s1 - s2) for s1, s2 in zip(conv2.get_shape()[1:],
                                             upconved.get_shape()[1:]))
    crops = [(int(d) // 2, int(d) // 2) for d in size_diff][:3]
    cropped = Cropping3D(crops)(conv2)
    merged = merge([cropped, upconved], mode='concat', concat_axis=4)

    # post convolutions
    # TODO: replace with resnet module
    conv3 = Convolution3D(num_features, 3, 3, 3, activation='relu')(merged)
    conv4 = Convolution3D(num_features, 3, 3, 3, activation='relu')(conv3)
    return conv4


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

if __name__ == '__main__':
    INPUT_SHAPE = (35, 412, 412, 1)
    OUTPUT_SHAPE = (7, 324, 324, 1)

    x = Input(shape=INPUT_SHAPE)    middle = unet(x, 10, depth=3, mult=3)
    out = Convolution3D(1, 1, 1, 1, activation='sigmoid')(middle)
    assert all(o1 == o2 for o1, o2 in zip(OUTPUT_SHAPE, out.get_shape()[1:]))

    model = Model(input=x, output=out)
    from keras.utils.visualize_util import plot
    plot(model, to_file='model.png', show_shapes=True)

    batchgen = datagen.generate_data('training_data.h5',
                                     INPUT_SHAPE[:-1],
                                     OUTPUT_SHAPE[:-1],
                                     2)

    i, o = next(batchgen)
    while o.mean() < 0.03:
        i, o = next(batchgen)

    #model.compile(loss=weighted_mse, optimizer=SGD(lr=1e-3, momentum=0.95, clipvalue=0.5))
    model.compile(loss=weighted_mse, optimizer=Adam(lr=1e-4))
    # model.load_weights('start.h5')
    model.fit_generator(batchgen, 50, 4000, verbose=2, callbacks=[CB(model, i, o)])
