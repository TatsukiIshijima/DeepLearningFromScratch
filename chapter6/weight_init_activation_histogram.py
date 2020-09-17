import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


if __name__ == '__main__':
    # 1000個のデータ
    input_data = np.random.randn(1000, 100)
    # 各隠れ層のニューロン数
    node_num = 100
    # 隠れ層の数
    hidden_layer_size = 5
    activations = {}

    x = input_data

    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i-1]

        # w = np.random.randn(node_num, node_num) * 1
        # w = np.random.randn(node_num, node_num) * 0.01
        # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
        # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)
        w = np.random.randn(node_num, node_num) / np.sqrt(node_num)

        a = np.dot(x, w)

        # z = sigmoid(a)
        # z = ReLU(a)
        z = tanh(a)

        activations[i] = z

    for i, a in activations.items():
        plt.subplot(1, len(activations), i+1)
        plt.title(str(i+1) + "-layer")
        if i != 0: plt.yticks([], [])
        plt.hist(a.flatten(), 30, range=(0,1))
    plt.show()