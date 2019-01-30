import tensorflow as tf

# in ->
# 3 -> 64 -> k9s1p4 -> ReLU
# 64 -> 32 -> k1s1p0 -> ReLU
# 32 -> 3 -> k5s1p2 -> out


class Net:
    def __init__(self, opt):
        self.opt = opt

    def __call__(self, X):
        with tf.name_scope('VGG16'):
            out = tf.layers.conv2d(X, filters=64, kernel_size=9, strides=(1, 1), padding='same')
            out = tf.layers.conv2d(out, 32, 1, (1, 1), 'same')
            out = tf.layers.conv2d(out, 3, 9, (1, 1), 'same')
        return out


def createModel(opt):
    model = Net(opt)
    return model
