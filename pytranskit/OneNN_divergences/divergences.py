import numpy as np
import dtw as DTW_package #Name of package is actually dtw-python


def d_T(signal1, signal2, eps=0.001):
    """
    Inputs:
        signal1: dependent variable values of signal 1
        signal2: dependent variable values of signal 2
        eps: regularizing term (See documentation)
    Output:
        d_T divergence between signal1 and signal2
    """
    test_grad = np.gradient(signal1)
    test_grad = np.abs(test_grad)
    percentual_eps = eps * np.max(test_grad)
    test_grad += percentual_eps
    test_grad = test_grad/test_grad.sum()

    train_grad = np.gradient(signal2)
    train_grad = np.abs(train_grad)
    percentual_eps = eps * np.max(train_grad)
    train_grad += percentual_eps
    train_grad = train_grad/train_grad.sum()

    g = np.interp(test_grad.cumsum(), train_grad.cumsum(), np.linspace(0, 1, len(signal2)))
    reconst = np.interp(g, np.linspace(0, 1, len(signal2)), signal2)
    return weighted_2_norm(reconst - signal1, np.gradient(g) ** (1 / 2))

def weighted_2_norm(x, w):
    """
        Auciliary function for d_T
    """
    # Ensure x and w are numpy arrays
    x = np.array(x)
    w = np.array(w)

    # Calculate the weighted 2-norm
    norm = np.sqrt(np.sum(w * x ** 2))
    return norm


def dtw(signal1, signal2):
    """
    Inputs:
        signal1: dependent variable values of signal 1
        signal2: dependent variable values of signal 2
    Output:
        dtw divergence between signal1 and signal2
    """
    return DTW_package.dtw(signal1, signal2).distance