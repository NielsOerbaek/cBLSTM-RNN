import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint

# Our Preprocessing Library
import prepros as pp

num_samples = 12500
batch_size = 20
hidden_size = 300

# -- Preprocessing
# Vocab files
print("Making vocab dicts of size " + str(pp.vocab_size))
w2i, i2w = pp.make_vocab(pp.vocab_file)


generate_embeddings = False
if generate_embeddings:
    embeddings_index = dict()
    f = open('glove.6B/glove.6B.300d.txt')
    i = 0
    for line in f:
        values = line.split()
        word = "".join(values[0:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs
        i += 1
        print(i)
    f.close()

    embedding_matrix = np.zeros((pp.vocab_size, 300))
    for word, index in zip(w2i.keys(), w2i.values()):
        if index > pp.vocab_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    pp.save_data_to_pickle(embedding_matrix, "glove-6B-300.pickle")
else:
    embedding_matrix = pp.load_data_from_pickle("glove-6B-300.pickle")


# Once you have generated the data files, you can outcomment the following line.
# pp.generate_data_files(num_samples)
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


def make_y(x_dataset):
    x = len(x_dataset)
    y = len(x_dataset[0])
    z = pp.vocab_size
    y_dataset = np.ndarray((x, y, z), np.uint8)
    for s in range(x):
        # print(s / x * 100)
        for w in range(y):
            y_dataset[s][(w - 1) % y] = pp.to_one_hot(x_dataset[s][w], z)
    return y_dataset


def data_generator():
    generator_counter = 0
    while generator_counter < num_samples:
        next_target = generator_counter + batch_size
        if next_target > num_samples:
            next_target = num_samples
        x_set = train_X[generator_counter:next_target]
        y_set = make_y(x_set)
        print(" - samples:", generator_counter, "-", next_target)
        generator_counter = next_target % num_samples
        yield x_set, y_set


# Helper functions
def perplexity(review):
    product = 1
    size = 0
    for sentence in review:
        size += len(sentence)
        for wordProb in sentence:
            product *= wordProb
    return product**(-1/size)


model_name = "./model/embedding-lstm-pos-lm-" + str(pp.vocab_size) + "vocab-" + str(num_samples) + "reviews-max-length-" + str(pp.max_sent_length) + ".model"
model_name = "./model/glove-emb-model-41.hdf5"
generate_model = False
if generate_model:
    # define LSTM
    print("Creating model")
    model = Sequential()
    model.add(Embedding(pp.vocab_size, hidden_size, input_length=pp.max_sent_length, mask_zero=True,
                        weights=[embedding_matrix], trainable=False))
    model.add(LSTM(pp.max_sent_length, input_shape=(pp.max_sent_length, hidden_size), return_sequences=True))
    model.add(LSTM(pp.max_sent_length, input_shape=(pp.max_sent_length, hidden_size), return_sequences=True))
    model.add(TimeDistributed(Dense(pp.vocab_size, activation="tanh")))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy'])
    print(model.summary())

    # Callback to save model between epochs
    checkpointer = ModelCheckpoint(filepath='./model/glove-emb-model-{epoch:02d}.hdf5', verbose=1)

    # train LSTM
    model.fit_generator(data_generator(), steps_per_epoch=(num_samples/batch_size), epochs=100, verbose=1, callbacks=[checkpointer], max_queue_size=3, use_multiprocessing=True)

    model.save(model_name)
else:
    model = load_model(model_name)


predictions = model.predict(train_X[0:20])

for k in range(len(predictions)):
    print("------------------")
    print("Input:", pp.ids_to_sentence(train_X[k], i2w))
    # print("Correct:", pp.one_hots_to_sentence(train_y[k], i2w))
    print("Prediction", pp.one_hots_to_sentence(predictions[k], i2w))
