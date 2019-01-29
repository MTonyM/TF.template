import tensorflow as tf
import math
from tqdm import tqdm
from util.misc import RunningAverage
import time

class Trainer:
    def __init__(self, model, criterion, metric, opt, optimState):
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.opt = opt
        self.optimState = optimState
        self.batchSize = opt.batchSize
        self.ep = tf.placeholder(tf.int32)
        self.lr = 0.0001 + tf.train.exponential_decay(opt.LR, self.ep, 1600, 1 / math.e)

    def process(self, dataLoader, epoch, split):
        train = split == 'train'
        num_iters = int(dataLoader[1] // self.batchSize)
        batch_X, batch_Y = tf.placeholder(tf.float32, shape=[None, 144, 144, 3]), \
                           tf.placeholder(tf.float32, shape=[None, 144, 144, 3])
        # /-------------------------^-------------------------\
        out = self.model(batch_X)
        loss = self.criterion(out, batch_Y)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss) if train else None
        AvgLoss = RunningAverage()

        with tf.Session().as_default() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            for i in tqdm(range(num_iters)):
                startTime = time.time()
                X, Y = sess.run(dataLoader[0].get_next())
                dataTime = time.time() - startTime
                if train:
                    _, cost, out_eval = sess.run([train_op, loss, out],
                                                 feed_dict={batch_X: X,
                                                            batch_Y: Y,
                                                            self.ep: num_iters * epoch + i})
                else:
                    cost, out_eval = sess.run([train_op, loss, out],
                                              feed_dict={batch_X: X,
                                                         batch_Y: Y})
                runningTime = time.time() - startTime
                # print(epoch, i, num_iters, dataTime, runningTime, cost, {})
                log = updateLog(epoch, i, num_iters, dataTime, runningTime, cost, {})
                print(log)
                AvgLoss.update(cost)
                # acc = self.criterion(out)
            coord.request_stop()
            coord.join()
        return AvgLoss()

    def train(self, dataLoader, epoch):
        return self.process(dataLoader, epoch, 'train')

    def test(self, dataLoader, epoch):
        return self.process(dataLoader, epoch, 'test')


def updateLog(epoch, i, length, time, datatime, err, Acc):
    log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f Err %1.4f   ' % (
        epoch, i, length, time, datatime, err)
    for metric in Acc:
        log += metric + " %1.4f  " % Acc[metric]()
    log += '\n'
    return log


def createTrainer(model, criterion, metric, opt, optimState):
    return Trainer(model, criterion, metric, opt, optimState)
