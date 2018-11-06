import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint

# Our Preprocessing Library
import prepros as pp

num_samples = 300
hidden_size = 300

# -- Preprocessing
# Vocab files
print("Making vocab dicts of size " + str(pp.vocab_size))
w2i, i2w = pp.make_vocab(pp.vocab_file)


generate_embeddings = False
if generate_embeddings:
    print("Generating embeddings")
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
    pp.save_data_to_pickle(embeddings_index, "glove-6B-300.pickle")
else:
    print("Loading embeddings")
    embeddings_index = pp.load_data_from_pickle("glove-6B-300.pickle")

embedding_matrix = np.zeros((pp.vocab_size, 300))
for word, index in zip(w2i.keys(), w2i.values()):
    if index > pp.vocab_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

# Once you have generated the data files, you can outcomment the following line.
# pp.generate_data_files(num_samples)
train_pos, train_neg, test_pos, test_neg = pp.load_all_data_files(num_samples)

# For some reason the for loop is only giving us the index here. I dunno why.
# Convert to sentence level training set:
# For some reason the for loop is only giving us the index here. I dunno why.
# Convert to sentence level training seÂt:
train_X = []
train_y = []

for i in train_pos:
    for j in train_pos[i]:
        train_X.append(np.array(pp.words_to_ids(train_pos[i][j], w2i)))
        train_y.append(1)

for i in train_neg:
    for j in train_neg[i]:
        train_X.append(np.array(pp.words_to_ids(train_neg[i][j], w2i)))
        train_y.append(0)

train_X = np.array(train_X)
train_y = np.array(train_y)

train_X, train_y = shuffle(train_X, train_y, random_state=42)
print(train_X.shape, train_X[0])
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


model_name = "./model/embedding-blstm-bc-pos-lm-" + str(pp.vocab_size) + "vocab-" + str(num_samples) + "reviews-max-length-" + str(pp.max_sent_length) + ".model"
# model_name = "./model/emb-model-40.hdf5"
generate_model = True
if generate_model:
    # define LSTM
    print("Creating model")
    model = Sequential()
    model.add(Embedding(pp.vocab_size, hidden_size, input_length=pp.max_sent_length, mask_zero=True,
                        weights=[embedding_matrix], trainable=False))
    model.add(Bidirectional(LSTM(hidden_size)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Callback to save model between epochs
    checkpointer = ModelCheckpoint(filepath='./model/glove-emb-BC-model-{epoch:02d}.hdf5', verbose=1)

    # train LSTM
    model.fit(train_X, train_y, epochs=10, batch_size=100, verbose=1, callbacks=[checkpointer])

    model.save(model_name)
else:
    model = load_model(model_name)


predictions = model.predict(train_X[0:20])

for k in range(len(predictions)):
    print("------------------")
    print("Input:", pp.ids_to_sentence(train_X[k], i2w))
    print("Correct:", train_y[k])
    print("Prediction:", predictions[k])
