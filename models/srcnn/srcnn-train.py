import math
import os
import time

import tensorflow as tf

from util.misc import RunningAverage
from util.progbar import progbar


class Trainer:
    def __init__(self, model, criterion, metrics, opt, optimState):
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.opt = opt
        self.optimState = optimState
        self.batchSize = opt.batchSize
        self.ep = tf.placeholder(tf.int32)
        self.lr = 0.000001 + tf.train.exponential_decay(opt.LR, self.ep, 1600, 1 / math.e)
        self.logger = {'train': open(os.path.join(opt.resume, 'train.log'), 'a+'),
                       'val': open(os.path.join(opt.resume, 'test.log'), 'a+')}

        self.sess = tf.Session()
        self.train_op = tf.train.AdamOptimizer(self.lr)

    def process(self, dataLoader, epoch, split):
        train = split == 'train'
        num_iters = int(dataLoader[1] // self.batchSize)
        batch_X, batch_Y = tf.placeholder(tf.float32, shape=[None, 144, 144, 3]), \
                           tf.placeholder(tf.float32, shape=[None, 144, 144, 3])
        # /-------------------------^-------------------------\
        out = self.model(batch_X)
        loss = self.criterion(out, batch_Y)
        train_op = self.train_op.minimize(loss)
        # init loss and accuracy
        avgLoss = RunningAverage()
        avgAcces = {}
        for metric in self.metrics:
            avgAcces[metric] = RunningAverage()
        bar = progbar(num_iters, width=self.opt.barwidth)
        print("\n=> [{}]ing epoch : {}".format(split, epoch))
        # train
        if epoch == 1 and train:
            self.sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        # begin one epoch
        for i in range(num_iters):
            if self.opt.debug and i > 2:  # check debug.
                break

            startTime = time.time()
            X_numpy, Y_numpy = self.sess.run(dataLoader[0].get_next())
            dataTime = time.time() - startTime

            logAcc = []
            if train:
                it = num_iters * (epoch - 1) + i * self.opt.batchSize
                lr, _, cost, out_eval = self.sess.run([self.lr, train_op, loss, out],
                                                      feed_dict={batch_X: X_numpy, batch_Y: Y_numpy, self.ep: it})
                logAcc.append(('LR', lr))
            else:
                cost, out_eval = self.sess.run([loss, out],
                                               feed_dict={batch_X: X_numpy,
                                                          batch_Y: Y_numpy})

            runningTime = time.time() - startTime

            # log record.
            avgLoss.update(cost)
            for metric in self.metrics:
                avgAcces[metric].update(self.metrics[metric](Y_numpy, out_eval))
                logAcc.append((metric, float(avgAcces[metric]())))

            bar.update(i, [('Time', runningTime), ('loss', float(cost)), *logAcc])
            log = updateLog(epoch, i, num_iters, runningTime, dataTime, cost, avgAcces)
            self.logger[split].write(log)

        coord.request_stop()
        coord.join()
        return avgLoss()

    def train(self, dataLoader, epoch):
        return self.process(dataLoader, epoch, 'train')

    def test(self, dataLoader, epoch):
        return self.process(dataLoader, epoch, 'val')


def updateLog(epoch, i, length, time, datatime, err, Acc):
    log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f Err %1.4f   ' % (
        epoch, i, length, time, datatime, err)
    for metric in Acc:
        log += metric + " %1.4f  " % Acc[metric]()
    log += '\n'
    return log


def createTrainer(model, criterion, metric, opt, optimState):
    return Trainer(model, criterion, metric, opt, optimState)
