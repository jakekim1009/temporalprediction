import numpy as np
import tensorflow as tf

import static_ops


def apply_static_op_to_sequence(input_, op):
    """
    Expects input_ to have shape
    [num_frames, num_batches, image_height, image_width, image_channels]
    or
    [num_frames, num_batches, dimension]
    op should have input_ as its only argument and should return a Tensor such that the first
    dimension is across the batches
    op should also operate independently across batches
    """
    shape = input_.get_shape().as_list()
    shape_length = len(shape)
    assert not np.any([dim is None for dim in shape]) # sadly you cannot specify None for this to work
    assert shape_length >= 3
    reshaped_input = tf.reshape(input_, [shape[0]*shape[1]] + shape[2:] )
    out = op(reshaped_input)
    out_shape = out.get_shape().as_list()
    return tf.reshape(out, [shape[0], shape[1]] + out_shape[1:] )

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    return apply_static_op_to_sequence(input_, lambda x : static_ops.conv2d(x, output_dim, k_h=k_h, k_w=k_w,
                                                                            d_h=d_h, d_w=d_w, stddev=stddev, name=name))
def max_pool(input_, k_h=2, k_w=2, d_h=2, d_w=2, name='max_pool'):
    return apply_static_op_to_sequence(input_, lambda x : static_ops.max_pool(x, k_h=k_h, k_w=k_w,

def resnet()                                                                              d_h=d_h, d_w=d_w, name=name))


def lrelu(x, leak=0.2, name="lrelu"):
    return static_ops.lrelu(x, leak=leak, name=name)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    return apply_static_op_to_sequence(input_, lambda x : static_ops.linear(x, output_size, scope=scope,










                                                                            stddev=stddev, bias_start=bias_start, with_w=with_w))

def deconv2d(input_, output_shape,
                 k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                 name="deconv2d", with_w=False):
    return apply_static_op_to_sequence(input_, lambda x: static_ops.deconv2d(x, output_dim, k_h=k_h, k_w=k_w,
                                                                         d_h=d_h, d_w=d_w, stddev=stddev, name=name))

"""
TODO
class batch_norm(object):
    #Code modification of http://stackoverflow.com/a/33950177
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
            self.name = name
    def __call__(self, x, train=True):
        shape = x.get_shape().as_list()
        needs_reshape = len(shape) != 4
        if needs_reshape:
            orig_shape = shape
            if len(shape) == 2:
                x = tf.reshape(x, [shape[0], 1, 1, shape[1]])
            elif len(shape) == 1:
                x = tf.reshape(x, [shape[0], 1, 1, 1])
            else:
                assert False, shape
            shape = x.get_shape().as_list()
        if train:
            with tf.variable_scope(self.name) as scope:
                self.beta = tf.get_variable("beta", [shape[-1]],
                                    initializer=tf.constant_initializer(0.))
                self.gamma = tf.get_variable("gamma", [shape[-1]],
                                    initializer=tf.random_normal_initializer(1., 0.02))
                batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
                ema_apply_op = self.ema.apply([batch_mean, batch_var])
                self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)
                with tf.control_dependencies([ema_apply_op]):
                    mean, var = tf.identity(batch_mean), tf.identity(batch_var)
        else:
            mean, var = self.ema_mean, self.ema_var
        normed = tf.nn.batch_norm_with_global_normalization(
                x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)
        if needs_reshape:
           normed = tf.reshape(normed, orig_shape)
        return normed
"""