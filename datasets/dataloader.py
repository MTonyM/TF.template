import sys
sys.path.append("..")
import datasets.init as datasets


def genDataLoader(dataset, opt, split, total):
    dataset = dataset.shuffle(buffer_size=min(1000, total)) if \
        split == 'train' else dataset
    dataset = dataset.batch(opt.batchSize).repeat(opt.nEpochs)
    dataset = dataset.prefetch(1)
    iterator = dataset.make_one_shot_iterator()
    return iterator


def create(opt):
    loaders = []
    for split in ['train', 'val']:
        dataset, total_number = datasets.create(opt, split)
        loaders.append((genDataLoader(dataset, opt, split, total_number), total_number))
    return loaders[0], loaders[1]

