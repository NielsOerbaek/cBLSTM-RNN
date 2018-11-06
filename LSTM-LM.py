import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import TimeDistributed

# Our Preprocessing Library
import prepros as pp

num_samples = 2000

# -- Preprocessing
# Vocab files
print("Making vocab dicts of size " + str(pp.vocab_size))
w2i, i2w = pp.make_vocab(pp.vocab_file)


# Once you have generated the data files, you can outcomment the following line.
pp.generate_data_files(num_samples)
train_pos, train_neg, test_pos, test_neg = pp.load_all_data_files(num_samples)

# For some reason the for loop is only giving us the index here. I dunno why.
# Convert to sentence level training set:
train_X = []
for i in train_pos:
    for j in train_pos[i]:
        train_X.append(np.array(pp.words_to_ids(train_pos[i][j], w2i)))

train_X = np.array(train_X)


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
            v = pp.to_one_hot(dataset[s][w], z)
            o_h_x[s][w] = v
            o_h_y[s][(w-1) % y] = v
    return o_h_x, o_h_y


# Storing data to save time
one_hot_filename = "one-hot-sentenses-1000.pickle"
generate_data = True
if generate_data:
    train_X, train_y = dataset_to_onehot(train_X)
    print("Saving data")
    #pp.save_data_to_pickle(train_X, "X-" + one_hot_filename)
    #pp.save_data_to_pickle(train_y, "Y-" + one_hot_filename)
else:
    train_X = pp.load_data_from_pickle("X-" + one_hot_filename)
    train_y = pp.load_data_from_pickle("Y-" + one_hot_filename)

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


model_name = "./model/lstm-pos-lm-" + str(pp.vocab_size) + "vocab-" + str(num_samples) + "reviews-max-length-" + str(pp.max_sent_length) + ".model"
generate_model = True
if generate_model:
    # define LSTM
    print("Creating model")
    model = Sequential()
    model.add(LSTM(pp.max_sent_length, input_shape=(pp.max_sent_length, pp.vocab_size), return_sequences=True, dropout=0.2))
    model.add(TimeDistributed(Dense(pp.vocab_size, activation="sigmoid")))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    # train LSTM
    model.fit(train_X, train_y, epochs=10, batch_size=100, verbose=1)

    model.save(model_name)
else:
    model = load_model(model_name)


predictions = model.predict(train_X[0:20])

for k in range(len(predictions)):
    print("Input:", pp.one_hots_to_sentence(train_X[k], i2w))
    print("Correct:", pp.one_hots_to_sentence(train_y[k], i2w))
    print("Prediction", pp.one_hots_to_sentence(predictions[k], i2w))
    print("------------------")
