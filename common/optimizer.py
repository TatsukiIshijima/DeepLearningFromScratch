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
        :return:
        """
        for key in params.keys():
            params[key] -= self.lr * grads[key]