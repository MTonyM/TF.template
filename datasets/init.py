import importlib
import os
import pickle


def isvalid(opt, cachePath):
    info = pickle.load(open(cachePath, 'rb'))
    if info['basedir'] != opt.data:
        return False
    return True


def create(opt, split):
    cachePath = os.path.join(opt.gen, opt.dataset + '.pkl')

    if not os.path.exists(cachePath) or not isvalid(opt, cachePath):
        script = opt.dataset + '-gen'
        gen = importlib.import_module('datasets.' + opt.dataset + '.' + script)
        gen.exec(opt, cachePath)

    info = pickle.load(open(cachePath, 'rb'))
    dataset = importlib.import_module('datasets.' + opt.dataset + '.' + opt.dataset)
    return dataset.getInstance(info, opt, split)
