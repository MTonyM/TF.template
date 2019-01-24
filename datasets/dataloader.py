import sys
sys.path.append("..")
import datasets.init as datasets
import tensorflow as tf
slim = tf.contrib.slim

def genDataLoader(dataset, opt, split, total):
    provider = slim.dataset_data_provider.DatasetDataProvider(dataset=dataset,
                                                              num_readers=opt.nThreads,
                                                              shuffle=(split == 'train'),
                                                              common_queue_capacity=256,
                                                              common_queue_min=128,
                                                              seed=None)
    return provider


def create(opt):
    loaders = []
    for split in ['train', 'val']:
        dataset, total_number = datasets.create(opt, split)
        loaders.append((genDataLoader(dataset, opt, split, total_number), total_number))
    return loaders[0], loaders[1]

