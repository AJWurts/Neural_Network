import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
# This is useful for performing a gradient check with check_grad.


def unpack(w):
    return w[0], w[1], w[2], w[3]
    # return W1, b1, W2, b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.


def pack(W1, b1, W2, b2):
    return [W1, b1, W2, b2]

# Load the images and labels from a specified dataset (train or test).


def loadData(which):
    images = np.load("mnist_{}_images.npy".format(which))
    labels = np.load("mnist_{}_labels.npy".format(which))
    return images, labels


def plotSGDPath(trainX, trainY, ws):
    def toyFunction(x1, x2):
        return np.sin((2 * x1**2 - x2) / 10.)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    axis2 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i, j] = toyFunction(Xaxis[i, j], Yaxis[i, j])
    # Keep alpha < 1 so we can see the scatter plot too.
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Yaxis = 2*np.pi*np.random.random(8) - np.pi  # Just an example
    Zaxis = toyFunction(Xaxis, Yaxis)
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()


def reluPrime(X):
    return np.where(X > 0, 1.0, 0.0)


def softmax(x):
    exp = np.exp(x)
    exp_sum = np.sum(exp, axis=0)
    return (exp / exp_sum).T


def predict(X, w):
    W1, b1, W2, b2 = unpack(w)

    z1 = (W1.T.dot(X).T + b1).T  # Change b1 to column vector
    h1 = reluPrime(z1)
    z2 = (W2.T.dot(h1).T + b2).T  # Change b2 to column vector
    yhat = softmax(z2)
    # yHat is many results in one there X.shape[0] by 10
    return yhat

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).


def fCE(X, Y, w):
    cost = (-1/X.shape[1]) * np.sum(Y * np.log(predict(X, w)))
    return cost


def score(X, y, w):
    result = np.argmax(predict(X, w), axis=1) == np.argmax(y, axis=1)
    return np.sum(result) / result.shape[0]

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).


def gradCE(X, Y, w):
    W1, b1, W2, b2 = unpack(w)

    z1 = (W1.T.dot(X).T + b1).T
    h1 = reluPrime(z1)
    z2 = (W2.T.dot(h1).T + b2).T
    yhat = softmax(z2)

    yHatMinusY = yhat - Y

    g = ((yHatMinusY @ W2.T) * reluPrime(z1.T)).T

    grad_w2 = yHatMinusY.T @ h1.T
    grad_b2 = yHatMinusY 
    grad_w1 = g @ X.T
    grad_b1 = g

    return np.array([grad_w1, grad_b1, grad_w2, grad_b2])

def oneForwardProp(X, w):
    W1, b1, W2, b2 = unpack(w)

    z1 = W1.dot(X) + b1
    h1 = reluPrime(z1)
    z2 = W2.dot(h1) + b2
    yhat = softmax(z2)

    
    # yHatMinusY = yhat - 

    # g = ((yHatMinusY @ W2.T) * reluPrime(z1.T)).T

    # grad_w2 = yHatMinusY.T @ h1.T
    # grad_b2 = yHatMinusY 
    # grad_w1 = g @ X.T
    # grad_b1 = g


def backprop(X, Y, w):
    W1, b1, W2, b2 = unpack(w)

    z1 = (W1.T.dot(X).T + b1).T
    h1 = reluPrime(z1)
    z2 = (W2.T.dot(h1).T + b2).T
    yhat = softmax(z2)

    yHatMinusY = yhat - Y

    g = ((yHatMinusY @ W2.T) * reluPrime(z1.T)).T

    grad_w2 = yHatMinusY.T @ h1.T
    grad_b2 = np.mean(yHatMinusY, axis=0)
    grad_w1 = g @ X.T
    grad_b1 = np.mean(g, axis=1)

    return [grad_w1, grad_b1, grad_w2, grad_b2]

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN. Then return the sequence of w's obtained during SGD.
def train(X, y, testX, testY, w, E=30, alpha=0.00000, beta=0.00001, kappa=0.0025, n_hat=64):

    m = X.shape[0]  # m is number of features
    n = X.shape[1]  # n is number of training images

    # Shuffle X and y in unison https://stackoverflow.com/a/4602224/6291504
    p = np.random.permutation(n)
    X, y = X[:, p], y[p]
    w_history = [w]
    for j in range(E):
        for i in range(0, n, n_hat):
            if i + n_hat > n:
                n_hat = n - i
            X_batch = X[:, i:i+n_hat]
            y_batch = y[i:i+n_hat]

            grad_w1, grad_b1, grad_w2, grad_b2 = backprop(X_batch, y_batch, w)
            W1, b1, W2, b2 = unpack(w)

            W1 -= kappa * grad_w1.T + beta * W1 + alpha * np.sign(W1)
            b1 -= kappa * grad_b1 + beta * b1 + alpha * np.sign(b1)
            W2 -= kappa * grad_w2.T + beta * W2 + alpha * np.sign(W2)
            b2 -= kappa * grad_b2 + beta * b2 + alpha * np.sign(b2)
            w = pack(W1, b1, W2, b2)

            # print("batch:", i)
        w_history.append(w)
        print("Epoch:", j)
        print(score(testX, testY, w))
        print(fCE(X, y, w))
        

 

    np.save('W2.npy', W2)
    np.save('b2.npy', b2)
    np.save('W1.npy', W1)
    np.save('b1.npy', b1)

    return w_history


if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")
        optX, optY = loadData("validation")

    print("Loaded Data")
    # # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_INPUT, NUM_HIDDEN)) /
            NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT)) /
            NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    w = pack(W1, b1, W2, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn)## Use check grad on each individualW1, W2, b1, b2

    idxs = np.random.permutation(trainX.shape[0])[0:NUM_CHECK]
    print(fCE(np.atleast_2d(trainX[idxs, :].T), np.atleast_2d(trainY[idxs, :]), w))
    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs, :].T), np.atleast_2d(trainY[idxs, :]), w_),
                                    lambda w_: gradCE(np.atleast_2d(
                                        trainX[idxs, :]).T, np.atleast_2d(trainY[idxs, :]), w_),
                                    w))

    # # Train the network and obtain the sequence of w's obtained using SGD.
    # ws = train(trainX.T, trainY, optX.T, optY, w)

    # # Plot the SGD trajectory
    # plotSGDPath(trainX.T, trainY, ws)
