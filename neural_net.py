import time, numpy as np, tensorflow as tf, os.path
from utils import *
from tensorflow.examples.tutorials.mnist import input_data

def _construct_nn(images, input_dim, hidden1_dim, hidden2_dim, output_dim):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([input_dim, hidden1_dim],
            stddev = 1. / math.sqrt(float(input_dim))), name = 'weights')
        biases = tf.Variable(tf.zeros([hidden1_dim]), name = 'biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_dim, hidden2_dim],
            stddev = 1. / math.sqrt(float(hidden1_dim))), name = 'weights')
        biases = tf.Variable(tf.zeros([hidden2_dim]), name = 'biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_dim, output_dim],
            stddev = 1. / math.sqrt(float(hidden2_dim))), name = 'weights') 
        biases = tf.Variable(tf.zeros([output_dim]), name = 'biases')
        logits = tf.matmul(hidden2, weights) + biases
    return logits

def _multinomial_loss(logits, labels):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
          logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss

def training(loss):
  tf.scalar_summary(loss.op.name, loss)
  optimizer = tf.train.AdamOptimizer()
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

class NeuralNet():
    def __init__(self, batch_size = 128, hidden1_dim = 50, hidden2_dim = 50,
                       n_epoch = 10):
        self.batch_size = batch_size
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.n_epoch = 10
    def fit(self, data_function):
        with tf.Graph().as_default():
            n, input_dim = data_function.train.images.shape
            output_dim = data_function.validation.labels.shape[1]
            images_placeholder = tf.placeholder(tf.float32, 
                    shape=(None, input_dim))
            labels_placeholder = tf.placeholder(tf.float32, 
                    shape=(None, output_dim))
            logits = _construct_nn(images_placeholder, input_dim, 
                    self.hidden1_dim, self.hidden2_dim, output_dim)
            loss = _multinomial_loss(logits, labels_placeholder)
            train_op = training(loss)
            correct_prediction = tf.equal(tf.argmax(logits,1), 
                    tf.argmax(labels_placeholder,1))
            eval_correct = tf.reduce_mean(tf.cast(correct_prediction,
                    tf.float32))
            summary_op = tf.merge_all_summaries()
            saver = tf.train.Saver()
            sess = tf.Session()
            init = tf.initialize_all_variables()
            sess.run(init)
            summary_writer = tf.train.SummaryWriter("./MNIST_data/",
                    graph_def = sess.graph_def)
            for step in xrange( self.n_epoch * n / self.batch_size):
                batch_xs, batch_ys = data_function.train.next_batch(
                        self.batch_size)
                feed_dict = {images_placeholder: batch_xs,
                             labels_placeholder: batch_ys}
                _, loss_value = sess.run([train_op, loss], 
                                         feed_dict = feed_dict)
                if step % (n / self.batch_size) == 0:
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    feed_dict = {images_placeholder: 
                                        data_function.validation.images,
                                 labels_placeholder: 
                                        data_function.validation.labels}
                    valid_loss, valid_score = sess.run([loss, eval_correct],
                            feed_dict = feed_dict) 
                    PrintMessage(data_function.train.epochs_completed,
                            loss_value, valid_loss, valid_score)

if __name__ == '__main__':
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    clf = NeuralNet(n_epoch = 10)
    clf.fit(data_function = mnist)
