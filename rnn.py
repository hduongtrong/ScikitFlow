import numpy as np, tensorflow as tf
from utils import PrintMessage, loss_dict, score_dict
from tensorflow.python.ops import array_ops, math_ops, init_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid, tanh
from tensorflow.python.ops import rnn_cell
from preprocessing import GetNietData

class RNNCell(object):
  def __call__(self, inputs, state, scope=None):
    raise NotImplementedError("Abstract method")
  @property
  def input_size(self):
    raise NotImplementedError("Abstract method")
  @property
  def output_size(self):
    raise NotImplementedError("Abstract method")
  @property
  def state_size(self):
    raise NotImplementedError("Abstract method")
  def zero_state(self, batch_size, dtype):
    zeros = array_ops.zeros(
        array_ops.pack([batch_size, self.state_size]), dtype=dtype)
    zeros.set_shape([None, self.state_size])
    return zeros

class BasicRNNCell(RNNCell):
  def __init__(self, num_units):
    self._num_units = num_units
  @property
  def input_size(self):
    return self._num_units
  @property
  def output_size(self):
    return self._num_units
  @property
  def state_size(self):
    return self._num_units
  def __call__(self, inputs, state, scope=None):
    with vs.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
      output = tanh(linear([inputs, state], self._num_units, True))
    return output, output

def linear(args, output_size, bias, bias_start=0.0, scope=None):
  assert args
  if not isinstance(args, (list, tuple)):
    args = [args]

  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
    matrix = vs.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = math_ops.matmul(args[0], matrix)
    else:
      res = math_ops.matmul(array_ops.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = vs.get_variable(
        "Bias", [output_size],
        initializer=init_ops.constant_initializer(bias_start))
  return res + bias_term

class RNN():
    def __init__(self, n_step = 1000, hidden_size = 50, max_grad_norm = 5., 
                       init_scale = .1, batch_size = 128):
        self.n_step = n_step
        self.hidden_size    = hidden_size
        self.max_grad_norm  = max_grad_norm
        self.init_scale     = init_scale
        self.batch_size     = batch_size
    def fit(self, data_function):
        with tf.Graph().as_default(), tf.Session() as sess:
            n, s, p = data_function.train.X.shape
            X_pl = tf.placeholder(tf.float32, [self.batch_size, s, p])
            Y_pl = tf.placeholder(tf.float32, [self.batch_size, p])
            cell = rnn_cell.BasicLSTMCell(self.hidden_size)
            state = cell.zero_state(self.batch_size, tf.float32)
            outputs = []
            with tf.variable_scope("RNN"):
                for time_step in xrange(s):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(X_pl[:, time_step, :], state)
                    outputs.append(cell_output)
            softmax_w = tf.get_variable("softmax_w", [self.hidden_size, p])
            softmax_b = tf.get_variable("softmax_b", [p])
            logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
            loss = loss_dict['ce'](logits, Y_pl)
            tvars = tf.trainable_variables()
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

