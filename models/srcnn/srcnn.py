import tensorflow as tf


# in ->
# 3 -> 64 -> k9s1p4 -> ReLU
# 64 -> 32 -> k1s1p0 -> ReLU
# 32 -> 3 -> k5s1p2 -> out


class Net:
    def __init__(self, opt):
        self.opt = opt
        with tf.name_scope('SRCNN'):
            self.conv1 = tf.layers.Conv2D(filters=64, kernel_size=9, strides=(1, 1), padding='same')
            self.conv2 = tf.layers.Conv2D(filters=32, kernel_size=1, strides=(1, 1), padding='same')
            self.conv3 = tf.layers.Conv2D(filters=3, kernel_size=9, strides=(1, 1), padding='same')

    def __call__(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        return out


def createModel(opt):
    model = Net(opt)
    return model
