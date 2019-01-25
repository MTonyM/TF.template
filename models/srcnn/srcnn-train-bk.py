import os
import torch.optim as optim
from torch.autograd import Variable
from util.misc import RunningAverage
from util.progbar import progbar
import time


class Trainer:
    def __init__(self, model, criterion, metrics, opt, optimState):
        self.model = model
        self.criterion = criterion
        self.optimState = optimState
        self.opt = opt
        self.metrics = metrics
        if opt.optimizer == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), lr=opt.LR, momentum=opt.momentum, dampening=opt.dampening,
                                       weight_decay=opt.weightDecay)
        elif opt.optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=opt.LR, betas=(opt.momentum, 0.999), eps=1e-8,
                                        weight_decay=opt.weightDecay)

        if self.optimState is not None:
            self.optimizer.load_state_dict(self.optimState)

        self.logger = {'train': open(os.path.join(opt.resume, 'train.log'), 'a+'),
                       'val': open(os.path.join(opt.resume, 'test.log'), 'a+')}

    def train(self, trainLoader, epoch):
        self.model.train()
        print("=> Training epoch")
        avgLoss = RunningAverage()
        avgAcces = {}
        for metric in self.metrics:
            avgAcces[metric] = RunningAverage()
        self.progbar = progbar(len(trainLoader), width=self.opt.barwidth)
        for i, (input, target) in enumerate(trainLoader):
            if self.opt.debug and i > 10:  # check debug.
                break
            start = time.time()
            inputV, targetV= Variable(input), Variable(target)
            if self.opt.GPU:
                inputV, targetV = inputV.cuda(), targetV.cuda()

            self.optimizer.zero_grad()
            dataTime = time.time() - start

            output = self.model(inputV)
            loss = self.criterion(output, targetV) * 255 * 255 / 144 / 144
            loss.backward()
            self.optimizer.step()

            # LOG ===
            runTime = time.time() - start
            avgLoss.update(float(loss))
            logAcc = []
            a = output.data.cpu().numpy()
            b = targetV.data.cpu().numpy()
            for metric in self.metrics:
                avgAcces[metric].update(self.metrics[metric](a, b))
                logAcc.append((metric, float(avgAcces[metric]())))
            del a, b
            log = updateLog(epoch, i, len(trainLoader), runTime, dataTime, float(loss), avgAcces)
            self.logger['train'].write(log)
            self.progbar.update(i, [('Time', runTime), ('loss', float(loss)), *logAcc])
            # END LOG ===

        log = '\n* Finished training epoch # %d  Loss: %1.4f  ' % (epoch, avgLoss())
        for metric in avgAcces:
            log += metric + " %1.4f  " % avgAcces[metric]()
        log += '\n'
        self.logger['train'].write(log)
        print(log)
        return avgLoss()

    def test(self, trainLoader, epoch):
        self.model.eval()
        print("=> Training epoch")
        avgLoss = RunningAverage()
        avgAcces = {}
        for metric in self.metrics:
            avgAcces[metric] = RunningAverage()
        self.progbar = progbar(len(trainLoader), width=self.opt.barwidth)
        for i, (input, target) in enumerate(trainLoader):
            if self.opt.debug and i > 10:  # check debug.
                break
            start = time.time()
            inputV, targetV= Variable(input), Variable(target)
            if self.opt.GPU:
                inputV, targetV = inputV.cuda(), targetV.cuda()

            self.optimizer.zero_grad()
            dataTime = time.time() - start

            output = self.model(inputV)
            loss = self.criterion(output, targetV) * 255 * 255 / 144 / 144

            # LOG ===
            runTime = time.time() - start
            avgLoss.update(float(loss))
            logAcc = []
            a = output.data.cpu().numpy()
            b = targetV.data.cpu().numpy()
            for metric in self.metrics:
                avgAcces[metric].update(self.metrics[metric](a, b))
                logAcc.append((metric, float(avgAcces[metric]())))
            del a, b
            log = updateLog(epoch, i, len(trainLoader), runTime, dataTime, float(loss), avgAcces)
            self.logger['val'].write(log)
            self.progbar.update(i, [('Time', runTime), ('loss', float(loss)), *logAcc])
            # END LOG ===

        log = '\n* Finished test epoch # %d  Loss: %1.4f ' % (epoch, avgLoss())
        for metric in avgAcces:
            log += metric + " %1.4f  " % avgAcces[metric]()
        log += '\n'
        self.logger['val'].write(log)
        print(log)
        return avgLoss()


    def LRDecay(self, epoch):
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt.LRDParam, gamma=0.1, last_epoch=epoch-2)

    def LRDecayStep(self):
        self.scheduler.step()


def updateLog(epoch, i, length ,time, datatime, err, Acc):
    log = 'Epoch: [%d][%d/%d] Time %1.3f Data %1.3f Err %1.4f   ' % (
            epoch, i, length, time, datatime, err)
    for metric in Acc:
        log += metric + " %1.4f  " % Acc[metric]()
    log += '\n'
    return log


def createTrainer(model, criterion, metric, opt, optimState):
    return Trainer(model, criterion, metric, opt, optimState)
