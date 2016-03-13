import logging, tensorflow as tf

try:
    logger
except NameError:
    logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
            filename="/tmp/history.log",
            filemode='a', level=logging.DEBUG,
            datefmt='%m/%d/%y %H:%M:%S')
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
            datefmt='%m/%d/%y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    logger = logging.getLogger(__name__)

def PrintMessage(epoch = -1, train_loss = 0, valid_loss = 0, valid_score = 0):
    if epoch == -1:
        logger.info("%6s|%10s|%10s|%10s", "Epoch", "TrainLoss",
                "ValidLoss", "ValidScore")
    else:
        logger.info("%6d|%10.4g|%10.4g|%10.4g", epoch, train_loss,
                valid_loss, valid_score)

def _multinomial_loss(logits, labels):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
          logits, labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def _mean_squared_error(yhat, labels):
    return tf.reduce_mean((labels - yhat)**2)

def accuracy_score(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits,1),
            tf.argmax(labels,1))
    eval_correct = tf.reduce_mean(tf.cast(correct_prediction,
            tf.float32))
    return eval_correct

def r2_score(yhat, labels):
    labels_mean = tf.reduce_mean(labels)
    return 1 - tf.reduce_mean((yhat - labels)**2) / tf.reduce_mean((labels -
        labels_mean)**2)


def r2_scores(labels, yhat):
    """ When there are multiple y. This will gives Rsquared for each column of
    y. Or the average R-Squared. Note that this is different from the R-squared    calculated by squashing the matrix into a column.
    """
    labels_mean = tf.reduce_mean(labels, 0, keep_dims = True)
    mse = tf.reduce_mean((labels - yhat)**2, 0)
    tse = tf.reduce_mean((labels - labels_mean)**2, 0)
    r2 = 1 - mse / tse
    return tf.reduce_mean(r2)

def training(loss):
    tf.scalar_summary(loss.op.name, loss)
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

loss_dict = {'mse'                  : _mean_squared_error,
             'ce'                   : _multinomial_loss,
             'crossentropy'         : _multinomial_loss,
             'cross_entropy'        : _multinomial_loss,
             'mean_squared_error'   : _mean_squared_error}
score_dict = {'mse'                 : r2_score,
             'ce'                   : accuracy_score,
             'crossentropy'         : accuracy_score,
             'cross_entropy'        : accuracy_score,
             'mean_squared_error'   : r2_score}
