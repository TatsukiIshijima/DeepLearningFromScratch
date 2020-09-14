import os
import sys
from collections import OrderedDict

from common.gradient import numerical_gradient
from common.layers import *

sys.path.append(os.pardir)

class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        """
        重みの初期化
        :param input_size: 入力層のニューロン数
        :param hidden_size: 中間層のニューロン数
        :param output_size: 出力層のニューロン数
        :param weight_init_std: 重み初期化時のガウス分布のスケール
        """
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 順序つき辞書
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        """
        認識（推論）
        :param x: 入力データ
        :return:
        """
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """
        損失関数の値算出
        :param x: 入力データ
        :param t: 教師データ
        :return:
        """
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        """
        認識精度
        :param x: 入力データ
        :param t: 教師データ
        :return:
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """
        数値微分による重みパラメータに対する勾配算出
        :param x: 入力データ
        :param t: 教師データ
        :return:
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        """
        誤差逆伝搬による重みパラメータに対する勾配算出
        :param x: 入力データ
        :param t: 教師データ
        :return:
        """
        # forward
        self.loss(x, t)

        # backward
        dout = self.lastLayer.backward()

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads