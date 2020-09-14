import numpy as np

from chapter5.two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist

if __name__ == '__main__':

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    x_batch = x_train[:3]
    t_batch = t_train[:3]

    gradient_numerical = network.numerical_gradient(x_batch, t_batch)
    gradient_backprop = network.gradient(x_batch, t_batch)

    for key in gradient_numerical.keys():
        diff = np.average(np.abs(gradient_backprop[key] - gradient_numerical[key]))
        print(key + ":" + str(diff))