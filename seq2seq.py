import ipdb, numpy as np, tensorflow as tf
from tensorflow.python.ops import seq2seq, rnn_cell, rnn
from preprocessing import GetAdditionData, GetPolyData
from utils import PrintMessage, accuracy_score, r2_score


def rnn_decoder(decoder_inputs, initial_state, cell, scope=None):
    with tf.variable_scope(scope or "dnn_decoder"):
        states, sampling_states = [initial_state], [initial_state]
        outputs, sampling_outputs = [], []
        with tf.op_scope([decoder_inputs, initial_state], "training"):
            for i, inp in enumerate(decoder_inputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output, new_state = cell(inp, states[-1])
                outputs.append(output)
                states.append(new_state)
        with tf.op_scope([initial_state], "sampling"):
            for i, _ in enumerate(decoder_inputs):
                if i == 0:
                    sampling_outputs.append(outputs[i])
                    sampling_states.append(states[i])
                else:
                    sampling_output, sampling_state = cell(
                        sampling_outputs[-1], sampling_states[-1])
                    sampling_outputs.append(sampling_output)
                    sampling_states.append(sampling_state)
    return outputs, states, sampling_outputs, sampling_states


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
        """ We make some important assumption about the data:
        The Output in additional to the original dimension, has an extra
        dimension to hold the special "START" symbol at the beginning of the
        Decoder input. Also, the first element in Y, should be this START
        symbol. So for example, if our output is in 2D with seq length 3: e.g. 
        [y1, y2], [y3, y4], [y5, y6] for one sequence. Then the transformed
        output should be [0,0,1], [y1, y2, 0], [y3, y4, 0], [y5, y6, 0]
        """
        # Define function to apply to previous decoder output, to generate the
        # next decoder input. Basically, we add a 0 to the vector, since for
        # the input, we have the extra dimension that denote START symbol, but
        # not for the decoder output. 
        tf.set_random_seed(1)
        with tf.Graph().as_default(), tf.Session() as sess:
            n, s, p = data_function.train.X.shape
            _, t, q = data_function.train.Y.shape
            X_pl    = tf.placeholder(tf.float32, [None, s, p])
            Y_pl    = tf.placeholder(tf.float32, [None, t, q])
            is_test = tf.placeholder(tf.int32)
            if is_test > 512:
                print("Motherfucker")
            lstm_cell     = rnn_cell.BasicLSTMCell(self.hidden_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
            _ , enc_state = rnn.rnn(cell, [X_pl[:,i,:] for i in xrange(s)],
                    dtype = tf.float32)
            outputs, _    = seq2seq.rnn_decoder([Y_pl[:,i,:] for i in 
                                xrange(t-1)], enc_state, cell)
            
            concat_outputs = tf.concat(0, outputs)
            softmax_w = tf.get_variable("softmax_w", [self.hidden_size, q])
            softmax_b = tf.get_variable("softmax_b", [q])
            logits = tf.matmul(concat_outputs, softmax_w) + softmax_b
            # labels = tf.reshape(tf.transpose(Y_pl[:,1:,:(q-1)]), [-1, 1])
            labels = tf.reshape(tf.transpose(Y_pl[:, 1:, :], [1,0,2]),
                    [-1, q])
            if self.loss == 'ce':
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits, labels))
                score = accuracy(logits, labels)
            elif self.loss == 'mse':
                loss = tf.reduce_mean((logits - labels)**2)
                score = r2_score(logits, labels)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                    self.max_grad_norm)
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
                    # ipdb.set_trace()
                    feed_dict = {X_pl: data_function.validation.X,
                                 Y_pl: data_function.validation.Y}
                    loss_valid, score_valid = sess.run(
                            [loss, score], feed_dict = feed_dict)
                    score_valid = 1 - loss_valid /\
                            np.mean(data_function.validation.Y[:,1:,:(q-1)]**2)
                    PrintMessage(data_function.train.epochs_completed,
                            loss_value , loss_valid, score_valid) 
    
               
if __name__ == '_main__':
    print("Running Addition Data")
    addition_data = GetAdditionData(n = 100000)
    clf = Seq2Seq(n_step = 100000, loss = 'ce')
    clf.fit(addition_data)
if __name__ == '__main__':
    poly_data = GetPolyData(10000, 20)
    clf = Seq2Seq(n_step = 600000, num_layers = 2, hidden_size = 256, loss = 'mse')
    clf.fit(poly_data)
