import os
import tensorflow as tf
import math
from util.misc import RunningAverage
from util.progbar import progbar
import time


class Trainer:
    def __init__(self, model, criterion, metrics, opt, optimState):
        self.model = model
        self.criterion = criterion
        self.metrics = metrics
        self.opt = opt
        self.optimState = optimState
        self.batchSize = opt.batchSize
        self.ep = tf.placeholder(tf.int32)
        self.lr = 0.0001 + tf.train.exponential_decay(opt.LR, self.ep, 1600, 1 / math.e)
        self.logger = {'train': open(os.path.join(opt.resume, 'train.log'), 'a+'),
                       'val': open(os.path.join(opt.resume, 'test.log'), 'a+')}

    def process(self, dataLoader, epoch, split):
        train = split == 'train'
        num_iters = int(dataLoader[1] // self.batchSize)
        batch_X, batch_Y = tf.placeholder(tf.float32, shape=[None, 144, 144, 3]), \
                           tf.placeholder(tf.float32, shape=[None, 144, 144, 3])
        # /-------------------------^-------------------------\
        out = self.model(batch_X)
        loss = self.criterion(out, batch_Y)
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss) if train else None

        # init loss and accuracy
        avgLoss = RunningAverage()
        avgAcces = {}
        for metric in self.metrics:
            avgAcces[metric] = RunningAverage()
        bar = progbar(num_iters, width=self.opt.barwidth)

        # train
        with tf.Session().as_default() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()

            # begin one epoch
            for i in range(num_iters):
                if self.opt.debug and i > 10:  # check debug.
                    break

                startTime = time.time()
                X_numpy, Y_numpy = sess.run(dataLoader[0].get_next())
                dataTime = time.time() - startTime

                if train:
                    _, cost, out_eval = sess.run([train_op, loss, out],
                                                 feed_dict={batch_X: X_numpy,
                                                            batch_Y: Y_numpy,
                                                            self.ep: num_iters * epoch + i})
                else:
                    cost, out_eval = sess.run([train_op, loss, out],
                                              feed_dict={batch_X: X_numpy,
                                                         batch_Y: Y_numpy})

                runningTime = time.time() - startTime

                # log record.
                avgLoss.update(cost)
                logAcc = []
                for metric in self.metrics:
                    avgAcces[metric].update(self.metrics[metric](Y_numpy, out_eval))
                    logAcc.append((metric, float(avgAcces[metric]())))

                bar.update(i, [('Time', runningTime), ('loss', float(cost)), *logAcc])
                log = updateLog(epoch, i, num_iters, runningTime, dataTime, cost, avgAcces)
                # if self.opt.debug:
                    # print(log)
                self.logger[split].write(log)

            coord.request_stop()
            coord.join()
        return avgLoss()

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
