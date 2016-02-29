import math, time, numpy as np, tensorflow as tf, os.path
from utils import PrintMessage, loss_dict, score_dict, training
from tensorflow.examples.tutorials.mnist import input_data

def AddLayer(input_data, input_dim, output_dim, layer_id = 0):
    with tf.name_scope('layer' + str(layer_id)):
        weights = tf.Variable(tf.truncated_normal([input_dim, output_dim],
            stddev = 1. / math.sqrt(float(input_dim))), name = 'weights')
        biases = tf.Variable(tf.zeros([output_dim]), name = 'biases')
        output_data = tf.matmul(input_data, weights) + biases
    return output_data

def _construct_nn2(images, input_dim, hidden1_dim, hidden2_dim, output_dim):
    hidden1 = tf.nn.relu(AddLayer(images, input_dim, hidden1_dim, 0)) 
    hidden2 = tf.nn.relu(AddLayer(hidden1, hidden1_dim, hidden2_dim, 1))
    logits  = AddLayer(hidden2, hidden2_dim, output_dim, 2)
    return logits

def _construct_nn3(images, layer_dims):
    hiddens = [images]
    for i in xrange(len(layer_dims) - 2):
        hiddens.append(tf.nn.relu(
            AddLayer(hiddens[-1], layer_dims[i], layer_dims[i+1], i)))
    logits = AddLayer(hiddens[-1], layer_dims[i + 1], layer_dims[i + 2],i + 1)
    return logits

class NeuralNet():
    def __init__(self, batch_size = 128, hidden1_dim = 50, hidden2_dim = 50,
                       n_epoch = 10, loss = "crossentropy"):
        self.batch_size     = batch_size
        self.hidden1_dim    = hidden1_dim
        self.hidden2_dim    = hidden2_dim
        self.n_epoch        = 10
        self.loss = loss
    def fit(self, data_function):
        # Need to add support for fit(X, Y)
        # Also support for out of memory data
        PrintMessage()
        with tf.Graph().as_default():
            n, input_dim = data_function.train.images.shape
            output_dim = data_function.validation.labels.shape[1]
            images_placeholder = tf.placeholder(tf.float32, 
                    shape=(None, input_dim))
            labels_placeholder = tf.placeholder(tf.float32, 
                    shape=(None, output_dim))
            logits = _construct_nn3(images_placeholder, [input_dim, 
                    self.hidden1_dim, self.hidden2_dim, output_dim])
            loss = loss_dict[self.loss](logits, labels_placeholder)
            score = score_dict[self.loss](logits, labels_placeholder)
            train_op = training(loss)
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
                    valid_loss, valid_score = sess.run([loss, score],
                            feed_dict = feed_dict) 
                    PrintMessage(data_function.train.epochs_completed,
                            loss_value, valid_loss, valid_score)

if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    clf = NeuralNet(n_epoch = 10)
    clf.fit(data_function = mnist)
