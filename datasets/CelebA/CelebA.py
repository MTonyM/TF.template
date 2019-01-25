import tensorflow as tf


class CelebA:
    def __init__(self, imageInfo, opt, split):
        self.recordPath = imageInfo[split]
        self.opt = opt
        self.split = split
        self.dir = imageInfo['basedir']
        self.transform = lambda x: x
        self.total = imageInfo['n_' + split]
        self.keysToFeatures = {
            'input/image': tf.FixedLenFeature([], default_value='', dtype=tf.string,),
            'input/height': tf.FixedLenFeature([], tf.int64, default_value=0),
            'input/width': tf.FixedLenFeature([], tf.int64, default_value=0),
            'target/image': tf.FixedLenFeature([], default_value='', dtype=tf.string,),
            'target/height': tf.FixedLenFeature([], tf.int64, default_value=0),
            'target/width': tf.FixedLenFeature([], tf.int64, default_value=0),
            'format': tf.FixedLenFeature([], default_value='jpeg', dtype=tf.string,),
        }

    def parse_example(self, serial_exmp):
        features = tf.parse_single_example(
            serial_exmp,
            features=self.keysToFeatures)
        # contains preprocess !
        # TODO: reshape the raw image.
        #     print(features['shape'])
        h = tf.cast(features['input/height'], tf.int32)
        w = tf.cast(features['input/width'], tf.int32)
        img_in = tf.image.decode_jpeg(features['input/image'], channels=3)
        img_in = tf.image.resize_images(img_in, [h, w])
        img_in = tf.cast(img_in, tf.float32) * (1. / 255) - 0.5

        img_tar = tf.image.decode_jpeg(features['target/image'], channels=3)
        img_tar = tf.image.resize_images(img_tar, [h, w])
        img_tar = tf.cast(img_tar, tf.float32) * (1. / 255) - 0.5


        # post process.
        return img_in, img_tar

    def get_dataset(self):
        dataset = tf.data.TFRecordDataset(self.recordPath)
        dataset = dataset.map(self.transform, num_parallel_calls=4)
        return dataset.map(self.parse_example, num_parallel_calls=4), self.total


def getInstance(info, opt, split):
    myInstance = CelebA(info, opt, split)
    return myInstance.get_dataset()
