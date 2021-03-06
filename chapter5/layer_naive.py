class MulLayer:
    """
    乗算レイヤ
    """

    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        """ 順伝搬
        :param x: 入力1
        :param y: 入力2
        :return:
        """
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        """ 逆伝搬
        :param dout:
        :return:
        """
        # x と y をひっくり返す
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    """
    加算レイヤ
    """

    def __init__(self):
        pass

    def forward(self, x, y):
        """
        順伝搬
        :param x: 入力
        :param y: 入力
        :return:
        """
        out = x + y
        return out

    def backward(self, dout):
        """
        逆伝搬
        :param dout:
        :return:
        """
        dx = dout * 1
        dy = dout * 1
        return dx, dy