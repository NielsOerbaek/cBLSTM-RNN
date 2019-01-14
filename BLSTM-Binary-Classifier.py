from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

# Our Preprocessing Library
import prepros as pp
# Our util functions
import utils

# Setting num_samples to zero means we want all the samples.
num_samples = 0
lstm_memory_cells = 300
glove_size = 300
model_name = "BLSTM-Glove-300"

# -- Preprocessing
# Vocab files
print("Making vocab dicts of size " + str(pp.vocab_size))
w2i, i2w = pp.make_vocab(pp.vocab_file)


generate_embeddings = True
if generate_embeddings:
    embedding_matrix = utils.generate_embedding_matrix('glove.6B/glove.6B.'+str(glove_size)+'d.txt', glove_size, w2i)
    pp.save_data_to_pickle(embedding_matrix, "glove-6B-"+str(glove_size)+".pickle")
else:
    embedding_matrix = pp.load_data_from_pickle("glove-6B-"+str(glove_size)+".pickle")

# Once you have generated the data files, you can outcomment the following line.
pp.generate_data_files(num_samples)
train_pos, train_neg, test_pos, test_neg = pp.load_all_data_files(num_samples)

# Split the data by 90/10 instead of 50/50
train_pos, test_pos = utils.split_data(train_pos, test_pos)
train_neg, test_neg = utils.split_data(train_neg, test_neg)

# Make the sentence-level datasets
train_pos = utils.make_language_model_sentence_dataset(train_pos, w2i)
train_neg = utils.make_language_model_sentence_dataset(train_neg, w2i)
test_pos = utils.make_language_model_sentence_dataset(test_pos, w2i)
test_neg = utils.make_language_model_sentence_dataset(test_neg, w2i)

# Attach labels to datasets
train_X, train_y = utils.from_lm_to_bc_dataset(train_pos, train_neg)
test_X, test_y = utils.from_lm_to_bc_dataset(test_pos, test_neg)


print("--------------------------------------")
print("Shape of train X-data: ", train_X.shape)
print("Shape of train y-data: ", train_y.shape)
print("--------------------------------------")
print("Shape of test X-data: ", test_X.shape)
print("Shape of test y-data: ", test_y.shape)
print("--------------------------------------")


model_filename = "./model/"+model_name+".model"
generate_model = True
if generate_model:
    # define LSTM
    print("Creating model")
    model = Sequential()
    model.add(Embedding(pp.vocab_size, glove_size, input_length=pp.max_sent_length, mask_zero=True,
                        weights=[embedding_matrix], trainable=True))
    model.add(Bidirectional(LSTM(lstm_memory_cells, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Callback to save model between epochs
    checkpointer = ModelCheckpoint(filepath='./model/'+model_name+'-{epoch:02d}.hdf5', verbose=1)

    # train LSTM
    model.fit(train_X, train_y, epochs=20, batch_size=100, verbose=1,
              callbacks=[checkpointer], validation_data=(test_X, test_y))

    model.save(model_name)
else:
    model = load_model(model_filename)
