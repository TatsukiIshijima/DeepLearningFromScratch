import numpy as np


class SGD:
    """
    確率的勾配降下法（Stochastic Gradient Descent）
    """
    def __init__(self, lr=0.01):
        """
        :param lr: 学習係数
        """
        self.lr = lr

    def update(self, params, grads):
        """
        :param params: 重みパラメータ（複数）
        :param grads: 勾配（複数）
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    """
    モーメンタム SGD
    物理法則的に最適化する方法
    """
    def __init__(self, lr=0.01, momentum=0.9):
        """
        :param lr: 学習係数
        :param momentum: 物理定数
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        """
        :param params: 重みパラメータ（複数）
        :param grads: 勾配（複数）
        """
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    """
    AdaGrad
    各パラメータに対してオーダーメードの値を作成し、最適化する方法
    """
    def __init__(self, lr=0.01):
        """
        :param lr: 学習係数
        """
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        """
        :param params: 重みパラメータ（複数）
        :param grads: 勾配（複数）
        :return:
        """
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            # self.h[key] に 0　があった場合に 0 で除算しないよう 1e-7 を追加
            params[key] -= self.lr * grads[key] * (np.sqrt(self.h[key] + 1e-7))