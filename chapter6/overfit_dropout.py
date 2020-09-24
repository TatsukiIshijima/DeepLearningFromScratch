import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer
from dataset.mnist import load_mnist

sys.path.append(os.pardir)

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

    x_train = x_train[:300]
    t_train = t_train[:300]

    # Dropoutの有無、割合の設定
    use_dropout = True
    dropout_ratio = 0.2

    network = MultiLayerNetExtend(input_size=784,
                                  hidden_size_list=[100, 100, 100, 100, 100, 100],
                                  output_size=10,
                                  use_dropout=use_dropout,
                                  dropout_ration=dropout_ratio)
    trainer = Trainer(network,
                      x_train,
                      t_train,
                      x_test,
                      t_test,
                      epochs=301,
                      mini_batch_size=100,
                      optimizer='sgd')
    trainer.train()

    train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list

    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
    plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()