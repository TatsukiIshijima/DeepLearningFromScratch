from chapter2.and_gate import _and
from chapter2.nand_gate import _nand
from chapter2.or_gate import _or


def _xor(x1, x2):
    s1 = _nand(x1, x2)
    s2 = _or(x1, x2)
    y = _and(s1, s2)
    return y


if __name__ == '__main__':
    for xs in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        y = _xor(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
