import tensorflow as tf
from util.misc import RunningAverage
from util.progbar import progbar
import time

class Trainer:
    def __init__(self, model, criterion, metric, opt, optimState):
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.opt = opt
        self.optimState = optimState
        self.batchSize = opt.batchSize

    def process(self, dataLoader, epoch, split):
        num_iters = int(dataLoader[1] // self.batchSize)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for i in range(num_iters):
                tar = dataLoader[0].get(['target/image'])

                # =>>>get batch tensor
                out = sess.run(tar)
                print(num_iters, len(out), i)
            coord.request_stop()
            coord.join()
        # loss = tf.losses.mean_squared_error(Y, Y_)
        # train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        # sess.run(train_op)
        # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    #     sess.run(init_op)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     out = sess.run(tar)
    #     print(type(out[0]))
    #     coord.request_stop()
    #     coord.join()
        return - epoch

    def train(self, dataLoader, epoch):
        return self.process(dataLoader, epoch, 'train')

    def test(self, dataLoader, epoch):
        return self.process(dataLoader, epoch, 'test')

    def LRDecay(self, epoch):
        pass

    def LRDecayStep(self):
        pass


def updateLog(epoch, i, length ,time, datatime, err, Acc):
    log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f Err %1.4f   ' % (
            epoch, i, length, time, datatime, err)
    for metric in Acc:
        log += metric + " %1.4f  " % Acc[metric]()
    log += '\n'
    return log

def createTrainer(model, criterion, metric, opt, optimState):
    return Trainer(model, criterion, metric, opt, optimState)
