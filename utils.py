import logging
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
