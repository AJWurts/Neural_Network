import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.decomposition import PCA

NUM_INPUT = 784  # Number of input neurons
NUM_HIDDEN = 40  # Number of hidden neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

def unpack(w):
    W1 = w[:NUM_INPUT * NUM_HIDDEN].reshape((NUM_HIDDEN,NUM_INPUT ))
    b1 = w[NUM_INPUT * NUM_HIDDEN:NUM_INPUT *
           NUM_HIDDEN + NUM_HIDDEN].reshape((NUM_HIDDEN,))
    W2 = w[-(NUM_OUTPUT * NUM_HIDDEN + 10):-
           NUM_OUTPUT].reshape((NUM_OUTPUT, NUM_HIDDEN))
    b2 = w[-NUM_OUTPUT:]
    return W1, b1, W2, b2
    # return W1, b1, W2, b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
# This is useful for performing a gradient check with check_grad.
def pack(W1, b1, W2, b2):
    w1_flattened = W1.reshape((W1.shape[0] * W1.shape[1],))
    w2_flattened = W2.reshape((W2.shape[0] * W2.shape[1],))
    b1_flattened = b1.reshape((b1.shape[0],))
    b2_flattened = b2.reshape((b2.shape[0],))
    result = np.concatenate(
        (w1_flattened, b1_flattened, w2_flattened, b2_flattened))
    return result

# Load the images and labels from a specified dataset (train or test).
def loadData(which):
    images = np.load("mnist_{}_images.npy".format(which))
    labels = np.load("mnist_{}_labels.npy".format(which))
    return images, labels


def plotSGDPath(trainX, trainY, ws):
    def zed_f(x, y):
        return fCE(trainX[:,:2500], trainY[:2500], (clf.inverse_transform([x, y])))

    fig = plt.figure()
    ax = fig.gca(projection='3d')


    clf = PCA(n_components=2)
    redux = clf.fit_transform(ws, trainX)

    ## Need to convert (x,y) => 31830 weights using inverse transform
    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.linspace(np.min(redux[:,0]) - 10, np.max(redux[:,0])+5, 20)  # Just an 
    axis2 = np.linspace(np.min(redux[:,1] )- 4, np.max(redux[:,1]) + 4, 20)  # Just an 
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)

    # Base axis on weights for 15000
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i, j] = zed_f(Xaxis[i,j], Yaxis[i, j])
    # Keep alpha < 1 so we can see the scatter plot too.
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = redux[:,0] #Just an example
    Yaxis = redux[:,1]
    Zaxis = [fCE(trainX[:,:2500], trainY[:2500],ws[i,:]) for i in range(len(ws))]
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()

def relu(X):
    return np.where(X > 0, X, 0)

def reluPrime(X):
    return np.where(X >= 0, 1.0, 0.0)


def softmax(x):
    exp = np.exp(x)
    exp_sum = np.sum(exp, axis=0)
    return (exp / exp_sum)


def predict(X, w):
    W1, b1, W2, b2 = unpack(w)

    z1 = ((W1 @ X).T + b1).T
    h1 = relu(z1)
    z2 = ((W2 @ h1).T + b2).T
    yhat = softmax(z2)

    return yhat

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).


def fCE(X, Y, w):
    pred = predict(X, w)
    logpred = np.log(pred)
    sumlogpred = np.sum(Y.T * logpred)

    cost = (-1/X.shape[1]) * sumlogpred
    return cost


def score(X, y, w):
    result = np.argmax(predict(X, w), axis=0) == np.argmax(y, axis=1)
    return np.sum(result) / result.shape[0]

# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. You might
# want to extend this function to return multiple arguments (in which case you
# will also need to modify slightly the gradient check code below).




def gradCE(X, y, w):
    W1, b1, W2, b2 = unpack(w)

    z1 = ((W1 @ X).T + b1).T
    h1 = relu(z1)
    z2 = ((W2 @ h1).T + b2).T
    yhat = softmax(z2)

    yhat_y = yhat - y.T

    gT = (yhat_y.T @ W2) * reluPrime(z1.T)
    g = gT.T

    grad_w2 = yhat_y @ h1.T
    grad_b2 = np.mean(yhat_y, axis=1)
    grad_w1 = g @ X.T
    grad_b1 = np.mean(g, axis=1)

    return pack(grad_w1, grad_b1, grad_w2, grad_b2)


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

    return grad_w1, grad_b1, grad_w2, grad_b2

# Given training and testing datasets and an initial set of weights/biases b,
# train the NN. Then return the sequence of w's obtained during SGD.

# # def train(X, y, testX, testY, w, E=30, alpha=0.00000, beta=0.00001, kappa=0.0025, n_hat=64):
def train(X, y, testX, testY, w, E=20, alpha=0, beta=0, kappa=0.0025, n_hat=16):

    m = X.shape[0]  # m is number of features
    n = X.shape[1]  # n is number of training images

    # Shuffle X and y in unison https://stackoverflow.com/a/4602224/6291504
    p = np.random.permutation(n)
    X, y = X[:,p], y[p]
    w_history = np.empty((E, 31810))
    for j in range(E):
        for i in range(0, n, n_hat):
            if i + n_hat > n:
                n_hat = n - i
            X_batch = X[:,i:i+n_hat ]
            y_batch = y[i:i+n_hat]

            grad_w1, grad_b1, grad_w2, grad_b2 = unpack(gradCE(X_batch, y_batch, w))
            W1, b1, W2, b2 = unpack(w)

            W1 -= kappa * grad_w1 + beta * W1 + alpha * np.sign(W1)
            b1 -= kappa * grad_b1 + beta * b1 + alpha * np.sign(b1)
            W2 -= kappa * grad_w2 + beta * W2 + alpha * np.sign(W2)
            b2 -= kappa * grad_b2 + beta * b2 + alpha * np.sign(b2)
            w = pack(W1, b1, W2, b2)

            # print("batch:", i)
        w_history[j,:] = w
        print("Epoch:", j)
        print(score(testX, testY, w))
        print(fCE(X, y, w))

    # np.save('W2.npy', W2)
    # np.save('b2.npy', b2)
    # np.save('W1.npy', W1)
    # np.save('b1.npy', b1)

    return w, w_history


def findBestHyperparameters(trainX, trainY, optX, optY, W, useAmt=10000):

    pv = np.array([0, 0.0001, 0.001, 0.0025, 0.01])

    n_hat = np.array([16, 32, 64])
    E = np.array([10, 20, 30])

    w = np.array([0,0,0,0,0])

    for key, i in enumerate([1,2,4]):
        values = []
        for j in range(4):
            w[key] = j
            print(w)
            final_w, _ = train(trainX.T[:, :useAmt], trainY[:useAmt], optX.T, optY, W,
             E=E[w[4]], alpha=pv[w[2] * ((i & 4) >> 2)], beta=pv[w[1] * ((i & 2) >> 1)], kappa=pv[w[0]], n_hat=n_hat[w[3]])
            fce = fCE(optX.T, optY, final_w)
            values.append(fce)
        print(values)
        w[key] = np.argmin(np.array(values))
    # w = np.array([3,0,0,0,0])
    for key, i in enumerate([1, 2]):
        values = []
        for j in range(2):
            print(w)
            w[key+3] = j
            final_w, _ = train(trainX.T[:, :useAmt], trainY[:useAmt], optX.T, optY, W, E=E[w[4] * (i & 2)], alpha=pv[w[2]], beta=pv[w[1]], kappa=pv[w[0]], n_hat=n_hat[w[3] * ((i & 1) >> 1)])
            fce = fCE(optX.T, optY, final_w)
            values.append(fce)
        
        w[key + 3] = np.argmin(np.array(values))
    
    return pv[w[0]], pv[w[1]], pv[w[2]], n_hat[w[3]], E[w[4]], w
    




if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY = loadData("train")
        testX, testY = loadData("test")
        optX, optY = loadData("validation")

    print("Loaded Data")
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT)) /
            NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN)) /
            NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)
    w = pack(W1, b1, W2, b2)

    # Check that the gradient is correct on just a few examples (randomly drawn)## Use check grad on each individualW1, W2, b1, b2
    idxs = np.random.permutation(trainX.shape[0])[0:1]

    print(scipy.optimize.check_grad(lambda w_: fCE(np.atleast_2d(trainX[idxs, :].T), np.atleast_2d(trainY[idxs, :]), w_),
                                    lambda w_: gradCE(np.atleast_2d(
                                        trainX[idxs, :].T), np.atleast_2d(trainY[idxs, :]), w_),
                                    w))
    # Looks at the weights individually. Commented out to increase run speed
    # # W1t, b1t, W2t, b2t = unpack(gradCE(np.atleast_2d(
    #     trainX[idxs, :].T), np.atleast_2d(trainY[idxs, :]), w))
    # W1a, b1a, W2a, b2a = unpack(scipy.optimize.approx_fprime(w, lambda w_: fCE(np.atleast_2d(
    #     trainX[idxs, :].T), np.atleast_2d(trainY[idxs, :]), w_), 1.49e-10))
    # # print("total: ", )


    # # Train the network and obtain the sequence of w's obtained using SGD.
    w, ws = train(trainX.T, trainY, testX.T, testY, w) 
    # print(findBestHyperparameters(trainX, trainY, optX, optY, w))
    # print(ws.shape)
    np.save("ws.npy", ws)
    ws = np.load("ws.npy")
    # Plot the SGD trajectory
    plotSGDPath(trainX.T, trainY, ws)
