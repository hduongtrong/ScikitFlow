import ipdb, numpy as np, tensorflow as tf
from tensorflow.python.ops import seq2seq, rnn_cell, rnn
from preprocessing import GetAdditionData, GetPolyData, GetPolyDataReal
from utils import PrintMessage, accuracy_score, r2_score
from tensorflow.python.ops import variable_scope


def rnn_decoder(decoder_inputs, initial_state, cell, softmax_w, softmax_b,
                scope=None):
  # Currently only support Mean Squared Error. Need to support Cross Entropy
  # By cchanging linear activation to argmax of the logits
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state_train = initial_state
    state_valid = initial_state
    outputs_train = []
    outputs_valid = []
    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output_train, state_train = cell(inp, state_train)
      outputs_train.append(output_train)
      if i > 0:
        # For the next decoder input, the decoder input of train and valid are
        # different. For train, we use the true decoder input, for test, we use
        # the output of the previous
        # ipdb.set_trace()
        output_valid, state_valid = cell(tf.matmul(outputs_valid[-1],
            softmax_w) + softmax_b, state_valid)
      else:
        # For the first decoder input, the decoder input of train and valid
        # are the same, since they are both fed the decoder_input[0]
        state_valid, output_valid  = state_train, output_train
      outputs_valid.append(output_valid)
  return outputs_train, state_train, outputs_valid, state_valid


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
            lstm_cell     = rnn_cell.BasicLSTMCell(self.hidden_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
            _ , enc_state = rnn.rnn(cell, [X_pl[:,i,:] for i in xrange(s)],
                   dtype = tf.float32)
            softmax_w = tf.get_variable("softmax_w", [self.hidden_size, q])
            softmax_b = tf.get_variable("softmax_b", [q])

            outputs, _, outputs_val, _    = rnn_decoder([Y_pl[:,i,:] for i in 
                                xrange(t-1)], enc_state, cell,
                                softmax_w, softmax_b)
            
            concat_outputs = tf.concat(0, outputs)
            concat_outputs_val = tf.concat(0, outputs_val)
            logits = tf.matmul(concat_outputs, softmax_w) + softmax_b
            logits_val = tf.matmul(concat_outputs_val, softmax_w) + softmax_b
            # labels = tf.reshape(tf.transpose(Y_pl[:,1:,:(q-1)]), [-1, 1])
            labels = tf.reshape(tf.transpose(Y_pl[:, 1:, :], [1,0,2]),
                    [-1, q])
            if self.loss == 'ce':
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits, labels))
                loss_val = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits_val, labels))
                score = accuracy(logits_val, labels)
            elif self.loss == 'mse':
                loss = tf.reduce_mean((logits - labels)**2)
                loss_val = tf.reduce_mean((logits_val[:,:(q-1)] - 
                                           labels[:,:(q-1)])**2)
                # score = r2_score(logits_val, labels)
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
                if i % 100 == 99:
                    # ipdb.set_trace()
                    feed_dict = {X_pl: data_function.validation.X,
                                 Y_pl: data_function.validation.Y}
                    loss1, loss2 = sess.run(
                            [loss, loss_val], feed_dict = feed_dict)
                    PrintMessage(data_function.train.epochs_completed,
                            loss1, loss2, 0) 
    
               
if __name__ == '_main__':
    print("Running Addition Data")
    addition_data = GetAdditionData(n = 100000)
    clf = Seq2Seq(n_step = 100000, loss = 'ce')
    clf.fit(addition_data)
if __name__ == '__main__':
    poly_data = GetPolyDataReal(100000, 20)
    clf = Seq2Seq(n_step = 600000, num_layers = 3, hidden_size = 256, loss = 'mse')
    clf.fit(poly_data)
