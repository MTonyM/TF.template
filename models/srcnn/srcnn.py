import tensorflow as tf

# in ->
# 3 -> 64 -> k9s1p4 -> ReLU
# 64 -> 32 -> k1s1p0 -> ReLU
# 32 -> 3 -> k5s1p2 -> out


class Net:
    def __init__(self, opt):
        self.opt = opt
    def model_fn(self, X, Y):
        # conv1 = tf.layers.Conv2D(filters=self.opt.numChannels, kernel_size=[9, 9], strides=1,
        #                          padding='same', activation=tf.nn.relu)(X)
        # conv2 = tf.layers.Conv2D(filters=self.opt.numChannels//2, kernel_size=[1, 1], strides=1,
        #                          padding='same', activation=tf.nn.relu)(conv1)
        # conv3 = tf.layers.Conv2D(filters=3, kernel_size=[9, 9], strides=1,
        #                          padding='same')(conv2)
        out = X
        out = tf.layers.conv2d(out,64,9,(1,1),'same')
        out = tf.layers.conv2d(out,32,1,(1,1),'same')
        out = tf.layers.conv2d(out,3,9,(1,1),'same')
        return out

def createModel(opt):
    model = Net(opt)
    return model
