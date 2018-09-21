import tensorflow as tf


def conv(input, filter_num, kernel_size, stride, scope):
    output = tf.contrib.slim.conv2d(input, filter_num, kernel_size, stride, activation_fn=None,
                                    biases_initializer=tf.zeros_initializer(dtype=tf.float32), scope=scope)
    return output


def res_block(input, filter_num, kernel_size, stride, scope):
    with tf.variable_scope(scope):
        output = conv(input, filter_num, kernel_size, stride, "conv1")
        output = conv(output, filter_num, kernel_size, stride, "conv2")
    return tf.nn.relu(tf.add(input, output))


def max_pool(inputs, kernel_size, stride):
    output = tf.contrib.slim.max_pool2d(inputs, kernel_size, stride)
    return output
