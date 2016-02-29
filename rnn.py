import numpy as np, tensorflow as tf
from utils import PrintMessage, loss_dict, score_dict
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell, rnn
from preprocessing import GetNietData

class RNN():
    def __init__(self, n_step = 1000, hidden_size = 50, max_grad_norm = 5., 
                       init_scale = .1, batch_size = 128, num_layers = 2):
        self.n_step = n_step
        self.hidden_size    = hidden_size
        self.max_grad_norm  = max_grad_norm
        self.init_scale     = init_scale
        self.batch_size     = batch_size
        self.num_layers     = num_layers
    def fit(self, data_function):
        with tf.Graph().as_default(), tf.Session() as sess:
            n, s, p = data_function.train.X.shape
            X_pl = tf.placeholder(tf.float32, [self.batch_size, s, p])
            Y_pl = tf.placeholder(tf.float32, [self.batch_size, p])
            lstm_cell = rnn_cell.BasicLSTMCell(self.hidden_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
            outputs, _ = rnn.rnn(cell, [X_pl[:,i,:] for i in xrange(s)],
                dtype = tf.float32)
            
            softmax_w = tf.get_variable("softmax_w", [self.hidden_size, p])
            softmax_b = tf.get_variable("softmax_b", [p])
            logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
            loss = loss_dict['ce'](logits, Y_pl)
            tvars = tf.trainable_variables()
            print([i.get_shape() for i in tvars])
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss,
                tvars), self.max_grad_norm)
            optimizer = tf.train.AdamOptimizer()
            train_op  = optimizer.apply_gradients(zip(grads, tvars))

            initializer = tf.random_uniform_initializer(-self.init_scale,
                    self.init_scale)
            tf.initialize_all_variables().run()
            for i in xrange(self.n_step):
                batch_xs, batch_ys = data_function.train.next_batch(
                                        self.batch_size)
                feed_dict = {X_pl: batch_xs, Y_pl: batch_ys}
                _, loss_value = sess.run([train_op, loss], 
                        feed_dict = feed_dict)
                if i % 100 == 0:
                    PrintMessage(data_function.train.epochs_completed, 
                            loss_value , 0, 0)

if __name__ == '__main__':
    niet = GetNietData()
    clf = RNN(n_step = 41000)
    clf.fit(niet)
