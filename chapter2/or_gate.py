import numpy as np


def _or(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    return 0 if tmp <= 0 else 1


if __name__ == '__main__':
    for xs in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        y = _or(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
