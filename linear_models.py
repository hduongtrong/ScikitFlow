import ipdb, tensorflow as tf, numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.examples.tutorials.mnist import input_data
from process_data import *

class LinearRegression(BaseEstimator):
    def __init__(self):
        pass
    def fit(self, X, y):
        n, p = X.shape
        q = 1 if len(y.shape) == 1 else y.shape[1]
        W = tf.Variable(tf.random_uniform([p,q], -1., 1.))
        X = tf.constant(X)
        y = tf.constant(y, shape = (n,q))
        
        W = tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(X), X)),
                      tf.matmul(tf.transpose(X), y))
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        clf.W = sess.run(W)
        return self
    def predict(self, Xtest):
        return Xtest.dot(clf.W)
    def score(self, Xtest, ytest):
        return r2_score(ytest, self.predict(Xtest))

class RidgeRegression(BaseEstimator):
    def __init__(self, alpha = 1.):
        self.alpha = alpha
    def fit(self, X, y):
        n, p = X.shape
        q = 1 if len(y.shape) == 1 else y.shape[1]
        W = tf.Variable(tf.random_uniform([p,q], -1., 1.))
        X = tf.constant(X)
        y = tf.constant(y, shape = (n,q))
        
        W = tf.matmul(tf.matrix_inverse(tf.matmul(tf.transpose(X), X) + 
                         n * self.alpha * \
                                     tf.constant(np.eye(p, dtype = "float32"))),
                      tf.matmul(tf.transpose(X), y))
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        clf.W = sess.run(W)
        return self
    def predict(self, Xtest):
        return Xtest.dot(clf.W)
    def score(self, Xtest, ytest):
        return r2_score(ytest, self.predict(Xtest))

class LogisticRegression(BaseEstimator):
    def __init__(self, n_epoch = 10, batch_size = 128, l1_penalty = 0., 
            l2_penalty = 0., verbose = 0):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.l2_penalty = l2_penalty
        self.verbose = verbose

    def fit(self, X = None, Y = None, data_function = None):
        if X is None:
            X, Y = data_function.train.next_batch(self.batch_size)
            n = data_function.train.num_examples
        else: n = X.shape[0]
        assert len(X.shape) == 2
        assert len(X) == len(Y)
        np.random.seed(1)
        p = X.shape[1]
        
        if len(Y.shape) == 1:
            YY = np.zeros((n,2))
            YY[:,0] = 1 - Y
            YY[:,1] = Y
            Y = YY

        q = Y.shape[1]
        x = tf.placeholder(tf.float32, [None, p])
        W = tf.Variable(tf.random_normal([p, q]))
        b = tf.Variable(tf.zeros([q]))

        p = tf.nn.softmax(tf.matmul(x, W) + b)
        y = tf.placeholder(tf.float32, [None, q])

        cross_entropy = -tf.reduce_sum(y * tf.log(p)) / self.batch_size + \
                self.l2_penalty * tf.reduce_sum(W**2)
        train_step = tf.train.AdamOptimizer().\
                        minimize(cross_entropy)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        n_iter = self.n_epoch * (n / self.batch_size)
        for i in xrange(n_iter):
            if data_function is None:
                sample = np.random.randint(n, size = self.batch_size)
                sess.run(train_step, feed_dict = {x: X[sample], 
                                              y: Y[sample]})
                if self.verbose and (i % (n / self.batch_size) == 0):
                    print(sess.run(cross_entropy, feed_dict = {x: X[sample],
                                                       y: Y[sample]}))
            else:
                batch_xs, batch_ys = data_function.train.next_batch(
                        self.batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
                if self.verbose and (i % (n / self.batch_size) == 0):
                    print(sess.run(cross_entropy, feed_dict = 
                        {x: data_function.validation.images, 
                         y: data_function.validation.labels}))
        self.W = sess.run(W)
        self.b = sess.run(b)

    def predict(self, Xtest):
        sess = tf.Session()
        p = sess.run(tf.nn.softmax(Xtest.dot(self.W) + self.b))
        return np.argmax(p, axis = 1)

    def score(self, Xtest, ytest):
        yhat = self.predict(Xtest)
        return accuracy_score(ytest, yhat)


def TestOnSimulateData():
    n = 10000; p = 10
    X = np.random.randn(n,p).astype("float32")
    beta = np.random.randn(p).astype("float32")
    y = X.dot(beta)
    Y = np.zeros(n, dtype = "int")
    Y[y > mean(y)] = 1

    if 1:
        clf = LinearRegression()
        clf.fit(X, y)
        print clf.score(X, y)
    if 1:
        clf = RidgeRegression(alpha = 0.1)
        clf.fit(X, y)
        print clf.score(X, y)
    if 1:
        clf = LogisticRegression(n_epoch = 100, l2_penalty = 0.0001)
        clf.fit(X, Y)
        print clf.score(X, Y)

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    clf = LogisticRegression(n_epoch = 10, verbose = 1)
    clf.fit(data_function = mnist)
