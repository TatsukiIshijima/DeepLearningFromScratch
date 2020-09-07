import numpy as np

from chapter3.nueralnet_mnist import get_data
from chapter3.nueralnet_mnist import init_network
from chapter3.nueralnet_mnist import predict

if __name__ == '__main__':
    x, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size):
        x_batch = x[i:i+batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print("Accuracy: " + str(float(accuracy_cnt) / len(x)))