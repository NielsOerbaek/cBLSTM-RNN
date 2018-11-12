import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.utils import to_categorical

# Our Preprocessing Library
import prepros as pp

# -- Preprocessing
# Vocab files
print("Making vocab dicts of size " + str(pp.vocab_size))
w2i, i2w = pp.make_vocab(pp.vocab_file)


# Once you have generated the data files, you can outcomment the following line.
pp.generate_data_files(100)
train_pos, train_neg, test_pos, test_neg = pp.load_all_data_files(100)

# For some reason the for loop is only giving us the index here. I dunno why.
# Convert to sentence level training seÂt:
train_X = []
for i in train_pos:
    for j in train_pos[i]:
        train_X.append(np.array(pp.words_to_ids(train_pos[i][j], w2i)))

train_X = np.array(train_X)


def to_one_hot(word_id, vocab_size):
    v = np.full(vocab_size, 0)
    v[word_id] = 1
    return v


def from_one_hot(one_hot_vector):
    for i, v in enumerate(one_hot_vector):
        if v == 1:
            return i


def dataset_to_onehot(dataset):
    print("Making the one-hot data")
    x = len(dataset)
    y = len(dataset[0])
    z = pp.vocab_size
    o_h_x = np.ndarray((x, y, z), np.int8)
    o_h_y = np.ndarray((x, y, z), np.int8)
    for s in range(x):
        print(s/x*100)
        for w in range(y):
            o_h_x[s][w] = to_one_hot(dataset[s][w], z)
            o_h_x[s][(w+1) % y] = to_one_hot(dataset[s][w], z)
    return o_h_x, o_h_y


# Storing data to save time
one_hot_filename = "one-hot-sentenses-100.pickle"
generate_data = True
if(generate_data):
    train_X, train_y = dataset_to_onehot(train_X)
    print("Saving data")
    pp.save_data_to_pickle((train_X, train_y), one_hot_filename)
else:
    (train_X, train_y) = pp.load_data_from_pickle(one_hot_filename)

print(train_X.shape)
print(train_y.shape)


# Helper functions
def perplexity(review):
    product = 1
    size = 0
    for sentence in review:
        size += len(sentence)
        for wordProb in sentence:
            product *= wordProb
    return product**(-1/size)


# define LSTM
print("Creating model")
model = Sequential()
model.add(LSTM(pp.max_sent_length, input_shape=(pp.max_sent_length, pp.vocab_size), return_sequences=True))
model.add(TimeDistributed(Dense(pp.vocab_size)))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(train_X, train_y, epochs=5, batch_size=100, verbose=1)

#model.save("./model/lstm-pos-lm-2003vocab-100reviews-max-length-100.model")

print(model.predict(train_X[0:2]))


# Jeppe attempt to improve the above code:

# define LSTM
print("Creating model")
model = Sequential()
model.add(LSTM(pp.max_sent_length, input_shape=(pp.max_sent_length, pp.vocab_size), return_sequences=True))
model.add(TimeDistributed(Dense(pp.vocab_size)))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(train_X, train_y, epochs=5, batch_size=100, verbose=1)

model.save("./model/lstm-pos-lm-2003vocab-100reviews-max-length-100.model")

print(model.predict(train_X[0:2]))