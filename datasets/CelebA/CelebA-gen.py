import os
import pickle

from tqdm import tqdm

from util.dataset import *


def exec(opt, cacheFilePath):
    assert os.path.exists(opt.data), 'Data directory not found: ' + opt.data
    print(">>>=====> Generating list of dat ....")

    path_test, n_test = createTFRecords(opt.data, 'test', opt.resize)
    path_val, n_val = createTFRecords(opt.data, 'val', opt.resize)
    path_train, n_train = createTFRecords(opt.data, 'train')
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


def createTFExample(patha, pathb, resize=None):
    img_in, inh, inw = process_img(patha, resize)
    img_tar, tarh, tarw = process_img(pathb, resize)
    tf_example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'input/image': bytes_feature(img_in),
                'input/height': int64_feature(inh),
                'input/width': int64_feature(inw),
                'target/image': bytes_feature(img_tar),
                'target/height': int64_feature(tarh),
                'target/width': int64_feature(tarw),
                'format': bytes_feature(b'jpg')
            }
        )
    )
    return tf_example


def createTFRecords(base, split, resize=None):
    path_input = os.path.join(base, split + "_blur")
    path_target = os.path.join(base, split + "_clear")
    list_input, list_target = listdir_(path_input), listdir_(path_target)
    assert list_input[10].split("/")[-1] == list_target[10].split("/")[-1], \
        "WARNING, sanity check in data loader failed."

    tf_path = os.path.join(base, split + ".tfrecords")
    print("=> creating tf records in " + tf_path)
    writer = tf.python_io.TFRecordWriter(tf_path)
    total = 0
    for i in tqdm(range(len(list_input))):
        tf_example = createTFExample(list_input[i], list_target[i], resize=None)
        total += 1
        writer.write(tf_example.SerializeToString())
    writer.close()
    return tf_path, total


def listdir_(path):
    filenames = os.listdir(path)  # label
    filenames = [os.path.join(path, f) for f in filenames if f.endswith('.jpg')]  # label
    return filenames
