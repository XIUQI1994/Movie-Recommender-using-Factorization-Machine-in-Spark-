from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from pyspark.mllib.util import MLUtils
from pyspark.storagelevel import *
import pyspark.mllib.linalg
import numpy as np
from sklearn.metrics import auc, roc_curve, average_precision_score, log_loss, mean_squared_error
import time
import pickle
from itertools import combinations

# -------------------------------------------------------------------------------
# Factorization machines


def fm_get_phi(x, w, bias):
    """
    Computes the probability of an instance given a model
    """
    # use the compress trick if x is a sparse vector
    # The compress trick allows to upload the weight matrix for the rows
    # corresponding to the indices of the non-zeros X values
    if isinstance(x, pyspark.mllib.linalg.SparseVector):
        W = w[x.indices]
        Bias = np.append(bias[x.indices], bias[-1])
        X = x.values
    elif isinstance(x, pyspark.mllib.linalg.DenseVector):
        W = w
        X = x
        Bias = bias
    else:
        return 'data type error'

    xa = np.array([X])
    VX = xa.dot(W)
    VX_square = (xa * xa).dot(W * W)
    phi = 0.5 * (VX * VX - VX_square).sum() + (Bias[:-1] * xa.reshape(-1)).sum() + Bias[-1]

    return phi


def loss_prefactor(phi, y, loss='mse'):
    if loss == 'mse':
        res = 2 * (phi - y)

    if loss == 'logloss':
        expnyt = np.exp(-y * phi)
        res = (-y * expnyt) / (1 + expnyt)
    return res


def fm_gradient_sgd_trick(X, y, W, bias, regParam, loss):
    """
    Computes the gradient for one instance using Rendle FM paper (2010) trick (linear time computation)
    """

    xa = np.array([X])
    x_matrix = xa.T.dot(xa)

    VX = xa.dot(W)
    VX_square = (xa * xa).dot(W * W)
    phi = 0.5 * (VX * VX - VX_square).sum() + (bias[:-1] * xa).sum() + bias[-1]

    np.fill_diagonal(x_matrix, 0)
    prefactor = loss_prefactor(phi, y, loss)
    result = prefactor * (np.dot(x_matrix, W))
    grads_W = regParam * W + result

    gb = np.append(xa, 1)
    grads_bias = prefactor * gb + regParam * bias

    return grads_W, grads_bias


def predictFM(data, w, bias):
    """
    Computes the probabilities given a model for the complete data set
    """
    # train_X = np.array(data.map(lambda row: row.features).collect())
    # func = lambda x: fm_get_phi(x, w, bias)
    # return np.apply_along_axis(func, 1, train_X)
    return data.map(lambda row: fm_get_phi(row.features, w, bias))


def logloss2(y_pred, y_true):
    """
    Computes the logloss given the true label and the predictions
    """
    # avoid NaN value

    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)

    # losses = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    losses = log_loss(y_true, y_pred)
    return losses


# -----------------------------------------------------------------------
# Train with parallel sgd

def trainFM_parallel_sgd(
        sc,
        train,
        val=None,
        weights=None,
        iterations=50,
        iter_sgd=5,
        alpha=0.01,
        regParam=0.01,
        factorLength=4,
        verbose=False,
        savingFilename=None,
        evalTraining=None,
        mode='reg',
        loss='mse'):
    """
    Train a Factorization Machine model using parallel stochastic gradient descent.

    Parameters:
    data : RDD of LabeledPoints
        Training data. Labels should be -1 and 1
        Features should be either SparseVector or DenseVector from mllib.linalg library
    iterations : numeric
        Nr of iterations of parallel SGD. default=50
    iter_sgd : numeric
        Nr of iteration of sgd in each partition. default = 5
    alpha : numeric
        Learning rate of SGD. default=0.01
    regParam : numeric
        Regularization parameter. default=0.01
    factorLength : numeric
        Length of the weight vectors of the FMs. default=4
    verbose: boolean
        Whether to ouptut iteration numbers, time, logloss for train and validation sets
    savingFilename: String
        Whether to save the model after each iteration
    evalTraining : instance of the class evaluation
        Plot the evaluation during the training (on a train and a validation set)
        The instance should be created before using trainFM_parallel_sgd

    returns: w
        numpy matrix holding the model weights
    """

    # split the data in train and validation sets if evalTraining or verbose
    if val: val.persist(StorageLevel.MEMORY_ONLY_SER)
    train.persist(StorageLevel.MEMORY_ONLY_SER)

    # glom() allows to treat a partition as an array rather as a single row at
    # time
    train_Y = train.map(lambda row: row.label).glom()
    train_X = train.map(lambda row: row.features).glom()
    train_XY = train_X.zip(train_Y).persist(StorageLevel.MEMORY_ONLY_SER)
    # train_XY = train_X.zip(train_Y).cache()

    # Initialize weight vectors
    nrFeat = len(train_XY.first()[0][0])
    if weights is not None:
        w = weights[0]
        bias = weights[1]
        assert(w.shape[1] == factorLength)
        print(w.shape)
        print(nrFeat)
        if w.shape[0] < nrFeat:
            w2 = np.random.ranf((nrFeat - w.shape[0], factorLength))
            w2 = w2 / np.sqrt((w2 * w2).sum())           
            bias2 = np.random.ranf(nrFeat - w.shape[0])
            bias2 = bias2 / np.sqrt((bias2 * bias2).sum())

            w = np.concatenate((w, w2), axis=0)
            tmp = bias[-1]
            bias = np.append(bias[:-1], bias2)
            bias = np.append(bias, tmp)

    else:
        np.random.seed(int(time.time()))
        w = np.random.ranf((nrFeat, factorLength))
        bias = np.random.ranf(nrFeat + 1)
        w = w / np.sqrt((w * w).sum())
        bias = bias / np.sqrt((bias * bias).sum())

    if evalTraining:
        evalTraining.evaluate(w, bias)
        if val:
            evalValidation = evaluation(val, mode, loss)
            evalValidation.modulo = evalTraining.modulo
            evalValidation.evaluate(w, bias)
        else:
            evalValidation = None

    if verbose:
        print('iter \ttime \ttrain_loss \tval_loss')
        # compute original logloss (0 iteration)
        if evalValidation:
            print('%d \t%d \t%5f \t%5f' %
                  (0, 0, evalTraining.loss[-1], evalValidation.loss[-1]))
        elif evalTraining:
            print('%d \t%d \t%5f ' %
                  (0, 0, evalTraining.loss[-1]))
        start = time.time()

    for i in range(iterations):
        wb = sc.broadcast(w)
        biasb = sc.broadcast(bias)
        weights = train_XY.map(
            lambda X_y: sgd_subset(
                X_y[0],
                X_y[1],
                wb.value,
                biasb.value,
                iter_sgd,
                alpha,
                regParam,
                loss))

        weights = weights.collect()       
        wsub = np.array([x[0] for x in weights]) 
        biassub = np.array([x[1] for x in weights]) 
        w = wsub.mean(axis = 0)
        bias = biassub.mean(axis = 0)

        # evaluate and store the evaluation figures each 'evalTraining.modulo'
        # iteration
        if evalTraining and i % evalTraining.modulo == 0:
            evalTraining.evaluate(w, bias)
            if evalValidation:
                evalValidation.evaluate(w, bias)
        if verbose:

            if i % evalTraining.modulo == 0:
                if evalValidation:
                    print('%d \t%d \t%5f \t%5f' % (i + 1, time.time() - \
                          start, evalTraining.loss[-1], evalValidation.loss[-1]))
                else:
                    print('%d \t%d \t%5f ' %(i + 1, time.time() - \
                          start, evalTraining.loss[-1]))
        if savingFilename:
            saveModel((w, bias), savingFilename + '_iteration_' + str(i + 1))

    train_XY.unpersist()

    return w, bias


def sgd_subset(train_X, train_Y, w, bias, iter_sgd, alpha, regParam, loss):
    """
    Computes stochastic gradient descent for a partition (in memory)
    Automatically detects which vector representation is used (dense or sparse)
    Parameter:
        train_X : list of pyspark.mllib.linalg dense or sparse vectors
        train_Y : list of labels
        w : numpy matrix holding the model weights
        iter_sgd : numeric
                Nr of iteration of sgd in each partition.
        alpha : numeric
                Learning rate of SGD.
        regParam : numeric
                Regularization parameter.

    return:
        numpy matrix holding the model weights for this partition
    """
    if isinstance(train_X[0], pyspark.mllib.linalg.DenseVector):
        return sgd_subset_dense(
            train_X,
            train_Y,
            w,
            bias,
            iter_sgd,
            alpha,
            regParam,
            loss)
    elif isinstance(train_X[0], pyspark.mllib.linalg.SparseVector):
        return sgd_subset_sparse(
            train_X, train_Y, w, bias, iter_sgd, alpha, regParam, loss)
    else:
        return 'data type error'


def sgd_subset_dense(
        train_X,
        train_Y,
        w,
        bias,
        iter_sgd,
        alpha,
        regParam,
        loss):
    """
    Computes stochastic gradient descent for a partition (in memory)
    Parameter:
        train_X : list of pyspark.mllib.linalg dense or sparse vectors
        train_Y : list of labels
        w : numpy matrix holding the model weights
        iter_sgd : numeric
                Nr of iteration of sgd in each partition.
        alpha : numeric
                Learning rate of SGD.
        regParam : numeric
                Regularization parameter.

    return:
        wsub: numpy matrix holding the model weights for this partition
    """
    N = len(train_X)
    wsub = w
    biassub = bias
    Gw = np.ones(w.shape)
    Gb = np.ones(bias.shape)
    for i in range(iter_sgd):
        np.random.seed(int(time.time()))
        random_idx_list = np.random.permutation(N)
        for j in range(N):
            idx = random_idx_list[j]
            X = train_X[idx]
            y = train_Y[idx]
            W_grads, bias_grads = fm_gradient_sgd_trick(
                X, y, wsub, bias, regParam, loss)
            Gw += W_grads * W_grads
            Gb += bias_grads * bias_grads
            wsub -= alpha * W_grads / np.sqrt(G)
            biassub -= alpha * bias_grads / np.sqrt(Gb)

    return wsub, biassub


def sgd_subset_sparse(
        train_X,
        train_Y,
        w,
        bias,
        iter_sgd,
        alpha,
        regParam,
        loss):
    """
    Computes stochastic gradient descent for a partition (in memory)
    The compress trick allows to upload the weight matrix for the rows corresponding to the indices of the non-zeros X values
    Parameter:
        train_X : list of pyspark.mllib.linalg dense or sparse vectors
        train_Y : list of labels
        w : numpy matrix holding the model weights
        iter_sgd : numeric
                Nr of iteration of sgd in each partition.
        alpha : numeric
                Learning rate of SGD.
        regParam : numeric
                Regularization parameter.

    return:
        wsub: numpy matrix holding the model weights for this partition
    """
    N = len(train_X)
    wsub = w
    biassub = bias
    Gw = np.ones(w.shape)
    Gb = np.ones(bias.shape)
    for i in range(iter_sgd):
        np.random.seed(int(time.time()))
        random_idx_list = np.random.permutation(N)
        for j in range(N):

            idx = random_idx_list[j]
            X = train_X[idx]
            y = train_Y[idx]
            indices = np.append(np.array(X.indices), len(bias)-1)

            W_grads, bias_grads = fm_gradient_sgd_trick(
                X.values, y, wsub[X.indices], bias[indices], regParam, loss)

            Gw[X.indices] += W_grads * W_grads
            wsub[X.indices] -= alpha * W_grads / np.sqrt(Gw[X.indices])

            Gb[indices] += bias_grads * bias_grads          
            biassub[indices] -= alpha * \
                bias_grads / np.sqrt(Gb[indices])

    return wsub, biassub


# -----------------------------------------------------------------------
# Train with non-parallel sgd
def trainFM_sgd(
        data,
        iterations=300,
        alpha=0.01,
        regParam=0.01,
        factorLength=4):
    """
    Train a Factorization Machine model using stochastic gradient descent, non-parallel.

    Parameters:
    data : RDD of LabeledPoints
            Training data. Labels should be -1 and 1
    iterations : numeric
            Nr of iterations of SGD. default=300
    alpha : numeric
            Learning rate of SGD. default=0.01
    regParam : numeric
            Regularization parameter. default=0.01
    factorLength : numeric
            Length of the weight vectors of the FMs. default=4

    returns: w
            numpy matrix holding the model weights
    """
    # data is labeledPoint RDD
    train_Y = np.array(data.map(lambda row: row.label).collect())
    train_X = np.array(data.map(lambda row: row.features).collect())
    (N, dim) = train_X.shape
    w = np.random.ranf((dim, factorLength))
    w = w / np.sqrt((w * w).sum())
    G = np.ones(w.shape)
    for i in range(iterations):
        np.random.seed(int(time.time()))
        random_idx_list = np.random.permutation(N)
        for j in range(N):
            idx = random_idx_list[j]
            X = train_X[idx]
            y = train_Y[idx]
            grads = fm_gradient_sgd_trick(X, y, w, regParam)
            G += grads * grads
            w -= alpha * grads / np.sqrt(G)

    return w

# -----------------------------------------------------------------------


def evaluate(data, w, bias, mode, loss):
    """
    Evaluate a Factorization Machine model on a data set.

    Parameters:
    data : RDD of LabeledPoints
            Evaluation data. Labels should be -1 and 1
    w : numpy matrix
            FM model, result from trainFM_sgd or trainFM_parallel_sgd

    returns : (rtv_pr_auc, rtv_auc, logl, mse, accuracy)
            rtv_pr_auc : Area under the curve of the Recall/Precision graph (average precision score)
            rtv_auc : Area under the curve of the ROC-curve
            logl : average logloss
            MSE : mean square error
            accuracy
    """
    # data.cache()
    data.persist(StorageLevel.MEMORY_ONLY_SER)
    y_true_rdd = data.map(lambda row: row.label)
    y_true = np.array(y_true_rdd.collect())
    y_pred_rdd = predictFM(data, w, bias)
    y_pred = np.array(y_pred_rdd.collect())


    if mode == 'clf':
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        if loss == 'logloss':
            logloss = logloss2(y_pred, y_true)
            return logloss

    # mse
    if mode == 'reg':
        if loss == 'mse':
            mse = mean_squared_error(y_pred, y_true)
            return mse


def saveModel(w, fileName):
    """
    Saves the model in a pickle file
    """
    # with open('model/'+fileName, 'wb') as handle :
    with open(fileName, 'wb') as handle:
        pickle.dump(w, handle)


def loadModel(fileName):
    """
    Load the model from a pickle file
    """
    # with open('model/'+fileName, 'rb') as handle :
    with open(fileName, 'rb') as handle:
        return pickle.load(handle)


def transform_data(data_01_label):
    """
    Transforms LabeledPoint RDDs that have 0/1 labels to -1/1 labels (as is needed for the FM models)
    """
    data = data_01_label.map(
        lambda row: LabeledPoint(-1 if row.label == 0 else 1, row.features))


# -----------------------------------------------------------------------
# Plot the error

class evaluation ():
    """ Store the evaluation figures (rtv_pr_auc, rtv_auc, logl, mse, accuracy) in lists
        Print the final error
        Plot the evolution of the error function of the number of iterations
    """

    def __init__(self, data, mode, loss):
        self.data = data
        self.mode = mode
        self.metric = loss
        self.loss = []
        # choose the modulo of the iterations to compute the evaluation
        self.modulo = 1

    def evaluate(self, w, bias):
        res = evaluate(self.data, w, bias, self.mode, self.metric)
        self.loss.append(res)
