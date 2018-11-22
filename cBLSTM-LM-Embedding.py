import numpy as np
import math
from sklearn.utils import shuffle
from keras.models import Sequential
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

num_reviews = 2000
batch_size = 50
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
pp.generate_data_files(num_reviews)
train_pos, train_neg, test_pos, test_neg = pp.load_all_data_files(num_reviews)

# Convert to sentence level training set:
train_X = []
for i in train_pos:
    for j in train_pos[i]:
        train_X.append(np.array(pp.words_to_ids(train_pos[i][j], w2i)))

# Shuffle for good orders sake
train_X = shuffle(np.array(train_X), random_state=420)
num_samples = len(train_X)


# Making the y data, which is just one-hot version of the input data
def make_y(x_dataset):
    x = len(x_dataset)
    y = len(x_dataset[0])
    z = pp.vocab_size
    y_dataset = np.ndarray((x, y, z), np.uint8)
    for s in range(x):
        for w in range(y):
            y_dataset[s][w] = pp.to_one_hot(x_dataset[s][w], z)
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


print("Shape of train_X", train_X.shape)


# Custom function to merge the forwards and backwards layer of the cBLSTM
def cBLSTM_merge(tensor_list):
    # In order to make the save/load model work, we have to import tf here.
    import tensorflow as tf

    forwards = tensor_list[0]
    backwards = tensor_list[1]

    # This is weird, but i think what i am doing is getting the dynamic dimensions of the input tensor
    # as tensors themselves, and then expanding the mask tensor in two extra dimensions, to be able
    # to concatenate our zero-mask to the forwards and backwards tensors.
    mask_tensor = tf.constant(0.0, shape=(1,))
    dim0 = tf.shape(forwards)[0]
    dim1 = tf.shape(forwards)[1]
    mask_tensor = tf.expand_dims(mask_tensor, axis=0)
    mask_tensor = tf.expand_dims(mask_tensor, axis=0)
    mask_tensor = tf.tile(mask_tensor, [dim0, dim1, 1])

    forwards = tf.concat([mask_tensor, forwards], axis=2, name="concat_forwards")
    backwards = tf.concat([backwards, mask_tensor], axis=2, name="concat_backwards")
    merged_tensor = tf.math.add(forwards, backwards)
    return merged_tensor


model_name = "./model/cBLSTM-pos.model"
generate_model = True
if generate_model:
    # define LSTM
    print("Creating model")
    input = Input(shape=(pp.max_sent_length,), name="Input_list_of_word_ids")

    emb = Embedding(pp.vocab_size, hidden_size,
                    input_length=pp.max_sent_length,
                    mask_zero=True,
                    weights=[embedding_matrix],
                    trainable=False,
                    name="Pretrained_word_vectors")(input)

    cBLSTM_forwards = LSTM(pp.max_sent_length - 1,
                           input_shape=(pp.max_sent_length, hidden_size),
                           return_sequences=True,
                           name="cBLSTM_forwards")(emb)
    cBLSTM_backwards = LSTM(pp.max_sent_length - 1,
                            input_shape=(pp.max_sent_length, hidden_size),
                            return_sequences=True,
                            name="cBLSTM_backwards",
                            go_backwards=True)(emb)

    merged = Lambda(cBLSTM_merge, name='cBLSTM_Merge')([cBLSTM_forwards, cBLSTM_backwards])

    time_dist = TimeDistributed(Dense(pp.vocab_size, activation="tanh"), name="Dense_layer_for_each_memory_cell")(merged)

    softmax = Activation('softmax', name="Softmax")(time_dist)

    model = Model(inputs=input, outputs=softmax)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', 'categorical_accuracy'])
    print(model.summary())

    # Callback to save model between epochs
    checkpointer = ModelCheckpoint(filepath='./model/cBLSTM-positive-LM-glove-emb-{epoch:02d}.hdf5', verbose=1)

    # train model
    model.fit_generator(data_generator(),
                        steps_per_epoch=num_samples/batch_size,
                        epochs=10,
                        verbose=1,
                        callbacks=[checkpointer],
                        max_queue_size=5)

    model.save(model_name)
else:
    model = load_model(model_name)


predictions = model.predict(train_X[0:20])

for k in range(len(predictions)):
    print("------------------")
    print("Input:", pp.ids_to_sentence(train_X[k], i2w))
    print("Prediction", pp.one_hots_to_sentence(predictions[k], i2w))
