import argparse
import numpy
import chainer
from chainer import training
from chainer.training import extensions
from net import EncDec

import matplotlib.pyplot as plt


class DataLoader(chainer.dataset.DatasetMixin):

    def __init__(self, batch_size, is_train=True):
        # 訓練データ
        t = numpy.linspace(0, 5*numpy.pi, 500)
        self.train = 10*numpy.sin(t).reshape(-1,1)
        self.train = numpy.tile(numpy.abs(self.train), (4*batch_size, 1)).astype('f')
        self.train = self.train.reshape(batch_size, -1, 1)

        # テストデータ
        t = numpy.linspace(0, 4*numpy.pi, 400)
        self.valid = 10*numpy.sin(t).reshape(-1,1)
        self.valid = numpy.concatenate((numpy.random.randn(100).reshape(100,1), self.valid), axis=0)
        self.valid = numpy.tile(numpy.abs(self.valid), (4, 1)).astype('f')
        self.valid = self.valid.reshape(1, -1, 1)

        self.is_train = is_train

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)

    def get_example(self, index):
        if self.is_train:
            return self.train[index].astype(numpy.float32)
        else:
            return self.valid[index].astype(numpy.float32)


def main(args):
    model = EncDec(args.inputsize, args.unit, args.slotsize, args.memorysize)

    train_dataset = DataLoader(args.batchsize, True)
    test_dataset = DataLoader(args.batchsize, False)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_dataset, args.batchsize)
    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend( \
        extensions.Evaluator(test_iter, model, device=args.gpu), \
        trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(
        trigger=(args.log_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss']),
        trigger=(args.log_interval, 'iteration'))

    print('start training')
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chainer example: PRNN')
    parser.add_argument('--inputsize', '-in', type=int, default=1)
    parser.add_argument('--slotsize', '-sl', type=int, default=32)
    parser.add_argument('--memorysize', '-m', type=int, default=64)
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=128,
                        help='number of units')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=4000,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    args = parser.parse_args()

    main(args)
