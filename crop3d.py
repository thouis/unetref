# -*- coding: utf-8 -*-
from __future__ import absolute_import

from keras import backend as K
from keras.engine import Layer, InputSpec

# from https://github.com/fchollet/keras/issues/3162 - credit to https://github.com/ironbar


# imports for backwards namespace compatibility

class Cropping3D(Layer):
    '''Cropping layer for 3D input (e.g. multichannel picture).
    '''
    input_ndim = 5

    def __init__(self, cropping=((1, 1), (1, 1), (1, 1)), dim_ordering=K.image_dim_ordering(), **kwargs):
        super(Cropping3D, self).__init__(**kwargs)
        assert len(cropping) == 3, 'cropping mus be two tuples, e.g. ((1,1),(1,1))'
        assert len(cropping[0]) == 2, 'cropping[0] should be a tuple'
        assert len(cropping[1]) == 2, 'cropping[1] should be a tuple'
        assert len(cropping[2]) == 2, 'cropping[1] should be a tuple'
        self.cropping = tuple(cropping)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=5)]

    def get_output_shape_for(self, input_shape):
        assert self.dim_ordering == 'tf'
        if self.dim_ordering == 'th':

            return (input_shape[0],
                    input_shape[1],
                    input_shape[2] - self.cropping[0][0] - self.cropping[0][1],
                    input_shape[3] - self.cropping[1][0] - self.cropping[1][1],
                    input_shape[4] - self.cropping[2][0] - self.cropping[2][1])
        elif self.dim_ordering == 'tf':
            return (input_shape[0],
                    input_shape[1] - self.cropping[0][0] - self.cropping[0][1],
                    input_shape[2] - self.cropping[1][0] - self.cropping[1][1],
                    input_shape[3] - self.cropping[2][0] - self.cropping[2][1],
                    input_shape[4])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        print(self.dim_ordering, x.get_shape())
        return x[:,
                 self.cropping[0][0]:-self.cropping[0][1],
                 self.cropping[1][0]:-self.cropping[1][1],
                 self.cropping[2][0]:-self.cropping[2][1],
                 :]

    def get_config(self):
        config = {'cropping': self.cropping}
        base_config = super(Cropping3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
