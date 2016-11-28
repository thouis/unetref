# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras.engine import Layer, InputSpec

# from https://github.com/fchollet/keras/issues/3162 - credit to https://github.com/ironbar


# imports for backwards namespace compatibility

class DepthToSpace3D(Layer):
    '''Cropping layer for 3D input (e.g. multichannel picture).
    '''
    input_ndim = 5

    def __init__(self, block_size=2, dim_ordering=K.image_dim_ordering(), **kwargs):
        super(DepthToSpace3D, self).__init__(**kwargs)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.block_size = block_size
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=5)]

    def get_output_shape_for(self, input_shape):
        block_size_sq = self.block_size ** 2
        if self.dim_ordering == 'tf':
            assert K._BACKEND == 'tensorflow'
            assert input_shape[4] % block_size_sq == 0
            return (input_shape[0],
                    input_shape[1],
                    input_shape[2] * self.block_size,
                    input_shape[3] * self.block_size,
                    input_shape[4] // block_size_sq)
        elif self.dim_ordering == 'th':
            assert K._BACKEND == 'theano'
            assert input_shape[1] % block_size_sq == 0
            return (input_shape[0],
                    input_shape[1] // block_size_sq,
                    input_shape[2],
                    input_shape[3] * self.block_size,
                    input_shape[4] * self.block_size)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        if self.dim_ordering == 'tf':
            Xsplit = K.tf.unpack(x, axis=1)
            return K.tf.pack([K.tf.depth_to_space(subx, self.block_size) for subx in Xsplit], axis=1)
        else:
            block_size = self.block_size
            b, k, d, r, c = x.shape
            r1 = x.reshape((b, k // (block_size ** 2), block_size, block_size, d, r, c))
            r2 = r1.transpose(0, 1, 4, 5, 2, 6, 3)
            return r2.reshape((b, k // (block_size ** 2), d, r * block_size, c * block_size))

    def get_config(self):
        config = {'block_size': self.block_size}
        base_config = super(DepthToSpace3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
