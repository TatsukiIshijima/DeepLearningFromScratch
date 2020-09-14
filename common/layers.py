from common.functions import *


class Relu:
    """
    ReLUレイヤ
    """

    def __init__(self):
        # mask は bool の Numpy 配列
        self.mask = None

    def forward(self, x):
        # maskで0以下の要素を取り出し、0にする
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        # 保持したmaskでTrueの場所を0にする
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    """
    シグモイドレイヤ
    """

    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1.0 - self.out)
        return dx


class Affine:
    """
    バッチ版Affineレイヤ
    """

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        # 入力データの形状を戻す（テンソル対応）
        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLoss:
    """
    ソフトマックス-損失関数レイヤ
    """

    def __init__(self):
        # 損失
        self.loss = None
        # softmaxの出力
        self.y = None
        # 教師データ
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self):
        batch_size = self.t.shape[0]
        # 教師データが one-hot-vector の場合
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx
