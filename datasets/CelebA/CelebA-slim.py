import tensorflow as tf
import tensorflow.contrib.slim as slim
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
        return self.readTFRecords(), self.total

    def readTFRecords(self):
        keysToFeatures = {
            'input/image': tf.FixedLenFeature([], default_value='', dtype=tf.string,),
            'input/height': tf.FixedLenFeature([], tf.int64, default_value=0),
            'input/width': tf.FixedLenFeature([], tf.int64, default_value=0),
            'target/image': tf.FixedLenFeature([], default_value='', dtype=tf.string,),
            'target/height': tf.FixedLenFeature([], tf.int64, default_value=0),
            'target/width': tf.FixedLenFeature([], tf.int64, default_value=0),
            'format': tf.FixedLenFeature([], default_value='jpeg', dtype=tf.string,),
        }
        itemsToHandlers = {
            'input/image': slim.tfexample_decoder.Image(image_key='input/image', format_key='format', channels=3),
            'target/image': slim.tfexample_decoder.Image(image_key='input/image', format_key='format', channels=3),
            'input/height': slim.tfexample_decoder.Tensor('input/height', shape=[]),
            'input/width': slim.tfexample_decoder.Tensor('input/width', shape=[]),
            'target/height': slim.tfexample_decoder.Tensor('target/height', shape=[]),
            'target/width': slim.tfexample_decoder.Tensor('target/width', shape=[])
        }
        decoder = slim.tfexample_decoder.TFExampleDecoder(keysToFeatures, itemsToHandlers)
        itemsToDescriptions = {
            'input/image': 'input image',
            'target/image': 'target image'
        }
        dataset = slim.dataset.Dataset(
            data_sources=self.recordPath,
            reader=tf.TFRecordReader,
            decoder=decoder,
            num_samples=self.total,
            items_to_descriptions=itemsToDescriptions,
        )
        return dataset



def getInstance(info, opt, split):
    myInstance = CelebA(info, opt, split)
    return myInstance.get_dataset()
