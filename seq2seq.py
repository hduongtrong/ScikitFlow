import ipdb, numpy as np, tensorflow as tf
from tensorflow.python.ops import seq2seq, rnn_cell, rnn
from preprocessing import GetAdditionData, GetPolyData
from utils import PrintMessage

class Seq2Seq():
    def __init__(self, n_step = 1000, num_layers = 2, batch_size = 128, 
            hidden_size = 50, max_grad_norm = 5, init_scale = .1,
            loss = 'ce'):
        self.n_step         = n_step
        self.num_layers     = num_layers
        self.batch_size     = batch_size
        self.hidden_size    = hidden_size
        self.max_grad_norm  = max_grad_norm
        self.init_scale     = init_scale
        self.loss           = loss
    def fit(self, data_function):
        with tf.Graph().as_default(), tf.Session() as sess:
            n, s, p = data_function.train.X.shape
            _, t, q = data_function.train.Y.shape
            X_pl    = tf.placeholder(tf.float32, [None, s, p])
            Y_pl    = tf.placeholder(tf.float32, [None, t, q])
            lstm_cell     = rnn_cell.BasicLSTMCell(self.hidden_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
            _ , enc_state = rnn.rnn(cell, [X_pl[:,i,:] for i in xrange(s)],
                    dtype = tf.float32)
            outputs, _    = seq2seq.rnn_decoder([Y_pl[:,i,:] for i in 
                                xrange(t-1)], enc_state, cell)
            concat_outputs = tf.concat(0, outputs)
            softmax_w = tf.get_variable("softmax_w", [self.hidden_size, q - 1])
            softmax_b = tf.get_variable("softmax_b", [q - 1])
            logits = tf.matmul(concat_outputs, softmax_w) + softmax_b
            labels = tf.reshape(tf.transpose(Y_pl[:,1:,0]), [-1, 1])
            ## Be really careful with reshape. They might reshape in the wrong
            ## direction, i.e. row-wise instead of column-wise
            if self.loss == 'ce':
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits, labels))
            elif self.loss == 'mse':
                loss_mat = (logits - labels)**2
                loss     = tf.reduce_mean(loss_mat)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                    self.max_grad_norm)
            optimizer = tf.train.AdamOptimizer()
            train_op  = optimizer.apply_gradients(zip(grads, tvars))

            initializer = tf.random_uniform_initializer(-self.init_scale,
                                        self.init_scale)
            tf.initialize_all_variables().run()
            PrintMessage()
            for i in xrange(self.n_step):
                batch_xs, batch_ys = data_function.train.next_batch(
                                        self.batch_size)
                feed_dict = {X_pl: batch_xs, Y_pl: batch_ys}
                _, loss_value = sess.run([train_op, loss],
                        feed_dict = feed_dict)
                if i % 100 == 0:
                    # ipdb.set_trace()
                    feed_dict = {X_pl: data_function.validation.X,
                                 Y_pl: data_function.validation.Y}
                    loss_valid = sess.run(loss, feed_dict = feed_dict)
                    score_valid = 1 - loss_valid /\
                    np.mean(data_function.validation.Y[:,1:,0]**2)
                    PrintMessage(data_function.train.epochs_completed,
                            loss_value , loss_valid, score_valid) 
    
               
if __name__ == '__main_':
    addition_data = GetAdditionData(n = 100000)
    clf = Seq2Seq(n_step = 100000)
    clf.fit(addition_data)
if __name__ == '__main__':
    poly_data = GetPolyData(100000, 40)
    clf = Seq2Seq(n_step = 600000, num_layers = 3, hidden_size = 256, loss = 'mse')
    clf.fit(poly_data)
