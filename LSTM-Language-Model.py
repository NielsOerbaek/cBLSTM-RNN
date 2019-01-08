import numpy as np
import math
from sklearn.utils import shuffle
from keras.optimizers import SGD
from keras.models import load_model
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.layers import Lambda
from keras.callbacks import ModelCheckpoint

# Our Preprocessing Library
import prepros as pp
import utils

num_reviews = 0
batch_size = 100
glove_size = 100
hidden_size = 300
train_positive = True
model_name = "LSTM-LM-positive"

# -- Preprocessing
# Vocab files
print("Making vocab dicts of size " + str(pp.vocab_size))
w2i, i2w = pp.make_vocab(pp.vocab_file)


generate_embeddings = False
if generate_embeddings:
    embedding_matrix = utils.generate_embedding_matrix('glove.6B/glove.6B.'+str(glove_size)+'d.txt', glove_size, w2i)
    pp.save_data_to_pickle(embedding_matrix, "glove-6B-"+str(glove_size)+".pickle")
else:
    embedding_matrix = pp.load_data_from_pickle("glove-6B-"+str(glove_size)+".pickle")


# Once you have generated the data files, you can outcomment the following line.
# pp.generate_data_files(num_reviews)
train_pos, train_neg, test_pos, test_neg = pp.load_all_data_files(num_reviews)

# Make the 90/10 data split instead of default 50/50
if train_positive:
    train, test = utils.split_data(train_pos, test_pos)
else:
    train, test = utils.split_data(train_neg, test_neg)

# Convert to sentence level dataset
train_X = utils.make_language_model_sentence_dataset(train, w2i)
test_X = utils.make_language_model_sentence_dataset(test, w2i)


num_samples = len(train_X)


# Making the y data, which is a once shifted one-hot version of the input.
def make_y(x_dataset):
    x = len(x_dataset)
    y = len(x_dataset[0])
    z = pp.vocab_size
    y_dataset = np.ndarray((x, y, z), np.uint8)
    for s in range(x):
        for w in range(y-1):
            y_dataset[s][w] = pp.to_one_hot(x_dataset[s][w+1], z)
        y_dataset[s][y-1] = pp.to_one_hot(0,z)
    return y_dataset

print("Sanity check: Is our samples and labels looking okay?")
print(pp.ids_to_sentence(train_X[0], i2w).replace("<MASK>",""))
print(pp.one_hots_to_sentence(make_y([train_X[0]])[0], i2w).replace("<MASK>",""))

def data_generator():
    generator_counter = 0
    while True:
        next_target = generator_counter + batch_size
        if next_target > num_samples:
            next_target = num_samples
        x_set = train_X[generator_counter:next_target]
        y_set = make_y(x_set)
        print(" - samples:", generator_counter, "-", next_target, end="")
        generator_counter = next_target % num_samples
        yield x_set, y_set


print("Shape of train_X:", train_X.shape)
print("Shape of test_X:", test_X.shape)


model_file_name = "./model/" + model_name + ".model"
generate_model = True
if generate_model:
    print("Creating model")
    input = Input(shape=(pp.max_sent_length,), name="Input_list_of_word_ids")

    emb = Embedding(pp.vocab_size, glove_size,
                    input_length=pp.max_sent_length,
                    mask_zero=True,
                    weights=[embedding_matrix],
                    trainable=True,
                    name="Pretrained_word_vectors")(input)

    lstm = LSTM(hidden_size,
                input_shape=(pp.max_sent_length, glove_size),
                return_sequences=True,
                name="LSTM")(emb)

    time_dist = TimeDistributed(Dense(pp.vocab_size, activation="tanh"), name="Dense_layer_for_each_memory_cell")(lstm)

    softmax = Activation('softmax', name="Softmax")(time_dist)

    model = Model(inputs=input, outputs=softmax)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    print(model.summary())

    # Callback to save model between epochs
    checkpointer = ModelCheckpoint(filepath='./model/' + model_name + '{epoch:02d}.hdf5', verbose=1)

    # train model
    model.fit_generator(data_generator(),
                        steps_per_epoch=(num_samples/batch_size),
                        epochs=50,
                        verbose=1,
                        callbacks=[checkpointer],
                        max_queue_size=5)

    model.save(model_file_name)
else:
    model = load_model(model_file_name)
