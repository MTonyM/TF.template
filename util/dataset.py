import io

import tensorflow as tf
from PIL import Image


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def processImg(path, resize=None):
    with tf.gfile.GFile(path, 'rb') as fid:
        encode_jpg = fid.read()
    encode_jpg_io = io.BytesIO(encode_jpg)
    image = Image.open(encode_jpg_io)
    process_flag = False
    # process the 4 channels .png
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
        process_flag = True
    # process the channel image
    elif image.mode != 'RGB':
        image = image.convert("RGB")
        process_flag = True
    # resize ??
    image = processReshape(image, resize)
    if process_flag or resize is not None:
        bytes_io = io.BytesIO()
        image.save(bytes_io, format='JPEG')
        encode_jpg = bytes_io.getvalue()
    width, height = image.size
    return encode_jpg, width, height


def processReshape(image, resize):
    width, height = image.size
    if resize is not None:
        if width > height:
            width = int(width * resize / height)
            height = resize
        else:
            width = resize
            height = int(height * resize / width)
        image = image.resize((width, height), Image.ANTIALIAS)
    return image
