import urllib2, numpy as np

class ProcessBatch():
    def __init__(self, X, Y):
        self.index = 0
        self.images = X
        self.labels = Y
        self.X = X
        self.Y = Y
        self.num_examples = len(X)
        self.epochs_completed = 0
        assert len(X) == len(Y)
    def next_batch(self, batch_size):
        if self.index + batch_size < self.num_examples:
            batch_index = np.arange(self.index, self.index + batch_size,
                    dtype = int)
        else:
            batch_index = np.concatenate([
                    np.arange(self.index, self.num_examples), 
                    np.arange(0, batch_size - self.num_examples + self.index)])
            self.epochs_completed += 1
        self.index = (self.index + batch_size) % self.num_examples
        return self.images[batch_index], self.labels[batch_index]

class SplitDataBatch():
    def __init__(self, X, Y, valid_ratio = .1, test_ratio = .1):
        assert len(X) == len(Y)
        n = len(X)
        n1 = int(n * (1 - valid_ratio - test_ratio))
        n2 = int(n * (1 - test_ratio))
        self.train = ProcessBatch(X[:n1], Y[:n1])
        self.validation = ProcessBatch(X[n1:n2], Y[n1:n2])
        self.test  = ProcessBatch(X[n2:], Y[n2:])

def GetNietData():
    text = urllib2.urlopen( 
         "https://s3.amazonaws.com/text-datasets/nietzsche.txt").read()
    chars = set(text)
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    maxlen = 20
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])

    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return SplitDataBatch(X, y) 

def GetAdditionData(n = 10000):
    s = 4
    p = 10
    X = np.empty((n,2*s,p), dtype = np.uint8)
    Y = np.empty((n,s+1,p), dtype = np.uint8)
    a = np.random.randint(low = 1000, high = 10000, size = n)
    b = np.random.randint(low = 1000, high = 10000, size = n)
    c = a + b
    X = [list('%04d%04d' %(a[i], b[i])) for i in xrange(n)]
    X = np.array(X, dtype = np.uint8)
    Y = [list('%05d' %i) for i in c]
    Y = np.array(Y, dtype = np.uint8)
    XX = np.zeros((n, 2*s, p), dtype = bool)
    YY = np.zeros((n, s+1, p), dtype = bool)
    for i in xrange(n):
        for j in xrange(2*s):
            XX[i][j][X[i,j]] = True
        for j in xrange(s + 1):
            YY[i][j][Y[i,j]] = True
    return SplitDataBatch(XX, YY)
if __name__ == '_main__':
    X = np.c_[np.arange(10), np.arange(10)]
    Y = np.arange(10)
    ata = SplitDataBatch(X, Y)
