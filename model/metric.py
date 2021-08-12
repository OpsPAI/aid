import math
import numpy as np


def cross_entropy(label, pred):
    """
    Two lists:  label, pred
    Returns the float corresponding to their cross-entropy
    """
    assert len(label) == len(
        pred), "label and pred should have the same length"
    sum = 0.0
    for x in map(lambda y, p: (1-y)*math.log(1-p)+y*math.log(p), label, pred):
        sum += x
    return -sum/len(label)


def mean_absolute_error(label, pred):
    label, pred = np.array(label), np.array(pred)
    diff = pred - label
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff


def root_mean_squared_error(label, pred):
    label, pred = np.array(label), np.array(pred)
    diff = pred - label
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
    rmse_val = np.sqrt(mean_diff)
    return rmse_val


if __name__ == "__main__":
    # Tests
    Y = [1, 1, 0, 0]
    P = [0.6, 0.2, 0.9, 0.2]
    print(cross_entropy(Y, P))
    print(mean_absolute_error(Y, P))
    print(root_mean_squared_error(Y, P))
