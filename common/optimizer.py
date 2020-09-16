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


class Adam:
    """
    Adam
    Momentum と AdaGrad を混ぜ合わせたような最適化手法
    """
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999):
        """
        :param lr: 学習係数
        :param beta1: １次モーメント係数
        :param beta2: 2次モーメント係数
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 * self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)