import numpy as np

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