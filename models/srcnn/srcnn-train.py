import tensorflow as tf
from util.misc import RunningAverage
from util.progbar import progbar
import time
from tqdm import tqdm
class Trainer:
    def __init__(self, model, criterion, metric, opt, optimState):
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.opt = opt
        self.optimState = optimState
        self.batchSize = opt.batchSize

    def process(self, dataLoader, epoch, split):
        print(dataLoader[1])
        num_iters = int(dataLoader[1] // self.batchSize)

        X_, Y_ = tf.placeholder(tf.float32, shape=[None, 144,144,3]), tf.placeholder(tf.float32,shape=[None, 144,144,3])
        loss = self.model.model_fn(X_, Y_)
        train_op = tf.train.AdamOptimizer().minimize(loss)

        with tf.Session().as_default() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            coord = tf.train.Coordinator()
            for i in tqdm(range(num_iters)):
                X, Y = sess.run(dataLoader[0].get_next())
                # =>>>get batch tensor

                out_1 = sess.run([train_op, loss], feed_dict={X_: X,
                                                              Y_: Y})
                print(num_iters, out_1, i)
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
