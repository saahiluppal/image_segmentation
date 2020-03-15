from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def log_n(x, n=10):

    log_e = tf.math.log(x)
    div_log_n = tf.math.log(tf.constant(n, dtype=log_e.dtype))
    return log_e / div_log_n


def binary_dilation(x, kernel_size=3):

    with tf.name_scope("binary_dilation"):
        num_channels = tf.shape(x)[-1]
        kernel = tf.ones((kernel_size, kernel_size, num_channels, 1), dtype=x.dtype)
        conv = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        clip = tf.clip_by_value(conv, 1., 2.) - 1.
        return clip


def binary_erosion(x, kernel_size=3):

    with tf.name_scope("binary_erosion"):
        num_channels = tf.shape(x)[-1]
        kernel = tf.ones((kernel_size, kernel_size, num_channels, 1), dtype=x.dtype)
        conv = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], padding='SAME')
        max_val = tf.constant(kernel_size * kernel_size, dtype=x.dtype)
        clip = tf.clip_by_value(conv, max_val - 1, max_val)
        return clip - (max_val - 1)


def binary_opening(tensor, kernel_size=3):

    with tf.name_scope("binary_opening"):
        return binary_dilation(binary_erosion(tensor, kernel_size), kernel_size)


def binary_closing(x, kernel_size=3):

    with tf.name_scope("binary_opening"):
        return binary_erosion(binary_dilation(x, kernel_size), kernel_size)


def binary_outline(x, kernel_size=3):

    with tf.name_scope("binary_outline"):
        return binary_dilation(x, kernel_size) - binary_erosion(x, kernel_size)