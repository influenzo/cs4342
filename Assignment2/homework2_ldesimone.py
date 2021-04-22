import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

########################################################################################################################
# PROBLEM 2
########################################################################################################################
# Given a vector x of (scalar) inputs and associated vector y of the target labels, and given
# degree d of the polynomial, train a polynomial regression model and return the optimal weight vector.
def trainPolynomialRegressor (x, y, d):
    mat = []
    for i in range(x):
        for j in range(y):
            (mat.append(x.i)^d)
    return method1(mat, y)

########################################################################################################################
# PROBLEM 1
########################################################################################################################

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    N, M, M = faces.shape
    return np.vstack((faces.reshape(N, M ** 2).T, np.ones(N)))

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    yHat = Xtilde.T.dot(w)
    return np.sum((y - yHat) ** 2) / (2 * len(y))

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    denom = len(y)
    wDupe = np.copy(w)
    wDupe[-1] = 0
    return (Xtilde.dot(Xtilde.T.dot(w) - y) + (alpha * wDupe)) / denom

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
# not linalg.inv, but linalg.solve :^)
def method1 (Xtilde, y):
    return np.linalg.solve(Xtilde.dot(Xtilde.T), Xtilde.dot(y))


# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
# "using gradient descent":
def method2 (Xtilde, y):
    return gradientDescent(Xtilde, y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
# thanks for the alpha, my gradientDescent should take nicely to that
def method3 (Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations
    #thanks for that^
    x = 0.01 * np.random.randn(len(Xtilde))
    for i in range(T):
        if i % 250 == 0: print("Testing at n =", i)
        grad = gradfMSE(x, Xtilde, y, alpha)
        x = x - EPSILON * grad
    return x
# show those 5 most egregious errors D:
def vis(w):
    im = w[:-1].reshape(48,48)
    fig,ax = plt.subplots(1)
    ax.imshow(im, cmap='gray')
    plt.show()
if __name__ == "__main__":

    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    w1 = method1(Xtilde_tr, ytr)
    w2 = method2(Xtilde_tr, ytr)
    w3 = method3(Xtilde_tr, ytr)

    vis(w1)
    vis(w2)
    vis(w3)

    # Visualizations; show top 5 most egregious errors
    if True:
        egre_err = np.argsort(abs(yte - Xtilde_te.T.dot(w1)))[::-1][:5]
        for i in egre_err:
            vis(Xtilde_te[:, i])
        egre_err = np.argsort(abs(yte - Xtilde_te.T.dot(w2)))[::-1][:5]
        for i in egre_err:
            vis(Xtilde_te[:, i])
        egre_err = np.argsort(abs(yte - Xtilde_te.T.dot(w3)))[::-1][:5]
        for i in egre_err:
            vis(Xtilde_te[:, i])
