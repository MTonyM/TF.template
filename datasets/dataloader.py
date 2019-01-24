import sys
sys.path.append("..")
import datasets.init as datasets


def genDataLoader(dataset, opt, split, total):
    epochs = opt.nEpochs
    buffer = total
    batch_size = opt.batchSize
    dataset = dataset.repeat(epochs).batch(batch_size)
    if split == 'train':
        dataset = dataset.shuffle(buffer)
    return dataset.make_one_shot_iterator()


def create(opt):
    loaders = []
    for split in ['train', 'val']:
        dataset, total_number = datasets.create(opt, split)
        loaders.append(genDataLoader(dataset, opt, split, total_number))
    return loaders[0], loaders[1]

