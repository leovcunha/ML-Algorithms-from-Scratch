import numpy as np


def sigmoid(vec):
    return 1/(1+np.exp(-vec))
