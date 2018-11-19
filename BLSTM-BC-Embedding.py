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
# Our util functions
import utils

num_samples = 100
hidden_size = 300

# -- Preprocessing
# Vocab files
print("Making vocab dicts of size " + str(pp.vocab_size))
w2i, i2w = pp.make_vocab(pp.vocab_file)


generate_embeddings = False
if generate_embeddings:
    embedding_matrix = utils.generate_embedding_matrix('glove.6B/glove.6B.300d.txt', w2i)
    pp.save_data_to_pickle(embedding_matrix, "glove-6B-300.pickle")
else:
    embedding_matrix = pp.load_data_from_pickle("glove-6B-300.pickle")

# Once you have generated the data files, you can outcomment the following line.
pp.generate_data_files(num_samples)
train_pos, train_neg, test_pos, test_neg = pp.load_all_data_files(num_samples)

train_X, train_y = utils.make_binary_classifier_dataset(train_pos, train_neg, w2i)
train_X, train_y = shuffle(train_X, train_y, random_state=420)
print("Shape of X-data: ", train_X.shape)
print("Shape of y-data: ", train_y.shape)


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
    checkpointer = ModelCheckpoint(filepath='./model/Test-glove-emb-BC-model-{epoch:02d}.hdf5', verbose=1)

    # train LSTM
    model.fit(train_X, train_y, epochs=2, batch_size=100, verbose=1, callbacks=[checkpointer])

    model.save(model_name)
else:
    model = load_model(model_name)


predictions = model.predict(train_X[0:20])

for k in range(len(predictions)):
    print("------------------")
    print("Input:", pp.ids_to_sentence(train_X[k], i2w))
    print("Correct:", train_y[k])
    print("Prediction:", predictions[k])
