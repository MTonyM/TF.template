import tensorflow as tf
import numpy as np


class CelebA:
    def __init__(self, imageInfo, opt, split):
        self.recordPath = imageInfo[split]
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']
        self.transform = None
        self.total = imageInfo['n_' + split]

    def get_dataset(self):
        dataset = tf.data.TFRecordDataset(self.recordPath)
        return dataset.map(self.parse_example), self.total

        # example = tf.train.Example(features=tf.train.Features(feature={
        #     'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[*shape_in, *shape_tar])),
        #     'input_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[in_raw])),
        #     'target_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tar_raw]))
        # }))

    def parse_example(self, serial_exmp):
        features = tf.parse_single_example(
            serial_exmp,
            features={
                'shape': tf.FixedLenFeature([4], tf.int64),
                'input_raw': tf.FixedLenFeature([], tf.string),
                'target_raw': tf.FixedLenFeature([], tf.string)
            })

        # contains preprocess !
        # TODO: reshape the raw image.
        # with tf.Session() as sess:
        #     print(features['shape'])
        shape = tf.cast(features['shape'], tf.int32)
        #     print(shape.eval(session=sess))
        #     tf.initialize_all_variables().run()
        #     nump = sess.run(shape)

        # print(nump.shape)
        shape_in, shape_tar = shape[:2], shape[2:]
        img_in = tf.decode_raw(features['input_raw'], tf.float32)
        img_in = tf.reshape(img_in, shape_in)
        img_in = tf.cast(img_in, tf.float32) * (1. / 255) - 0.5

        img_tar = tf.decode_raw(features['target_raw'], tf.float32)
        img_tar = tf.reshape(img_tar, shape_tar)
        img_tar = tf.cast(img_tar, tf.float32) * (1. / 255) - 0.5
        # print("=> pass test")
        # ( transforms.
        #   )
        return img_in, img_tar, shape_in, shape_tar


def getInstance(info, opt, split):
    myInstance = CelebA(info, opt, split)
    return myInstance.get_dataset()
