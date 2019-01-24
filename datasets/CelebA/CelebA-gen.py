import os
import pickle
import tensorflow as tf
from tqdm import tqdm
from PIL import Image


def exec(opt, cacheFilePath):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    print(">>>=====> Generating list of dat ....")

    path_test, n_test = loadSaveTFRecords(opt.data, 'test')
    path_val, n_val = loadSaveTFRecords(opt.data, 'val')
    path_train, n_train = loadSaveTFRecords(opt.data, 'train')
    info = {
        'basedir': opt.data,
        'test': path_test,
        'val': path_val,
        'train': path_train,
        'n_test': n_test,
        'n_val': n_val,
        'n_train': n_train
    }
    pickle.dump(info, open(cacheFilePath, 'wb'))
    return info


def loadSaveTFRecords(base, split):
    path_input = os.path.join(base, split + "_blur")
    path_target = os.path.join(base, split + "_clear")
    list_input, list_target = listdir_(path_input), listdir_(path_target)
    assert list_input[10].split("/")[-1] == list_target[10].split("/")[-1], \
        "WARNING, sanity check in data loader failed."

    def shape_raw(img_target_path):
        img = Image.open(img_target_path)
        shape_ = img.size
        img_raw = img.tobytes()
        return shape_, img_raw
    tf_path = os.path.join(base, split + ".tfrecords")
    if os.path.exists(tf_path):     # TODO: sum check
        return tf_path, len(list_input)

    print("=> Saving tf records to " + tf_path)
    writer = tf.python_io.TFRecordWriter(tf_path)
    for i in tqdm(range(len(list_input))):
        shape_tar, tar_raw = shape_raw(list_target[i])
        shape_in, in_raw = shape_raw(list_input[i])
        example = tf.train.Example(features=tf.train.Features(feature={
            'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=[*shape_in, *shape_tar])),
            'input_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[in_raw])),
            'target_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tar_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    return tf_path, len(list_input)


def listdir_(path):
    filenames = os.listdir(path)  # label
    filenames = [os.path.join(path, f) for f in filenames if f.endswith('.jpg')]  # label
    return filenames
