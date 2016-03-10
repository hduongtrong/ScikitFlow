import tensorflow as tf
import numpy as np
import time

np.random.seed(1)
n = 25000
x = np.array(np.random.randn(n,n), dtype = np.float32)
X = tf.constant(x, dtype = tf.float32)
y = tf.matmul(X, X)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
a = time.time(); sess.run(y); print time.time() - a
