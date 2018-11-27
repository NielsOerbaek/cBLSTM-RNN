import os
import re  # regex
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Masking
from keras.optimizers import SGD
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict
import prepros as pp

hidden = 300
batch_size = 10
max_sent_length = 40  # At least 93 % of all sentences within 40 words
vocab_size = 1000 + 4  # Start, end, UNK and padding
vocab_file = "./aclimdb/imdb.vocab"
start = "<s>"
end = "</s>"
unk = "<UNK>"


# Create the vocabulary from the vocab file, which is already sorted on the most common words
def make_vocab(path_to_vocab_file):
    # Making Word-to-ID dict
    word_to_id = defaultdict(lambda: len(word_to_id) + 1)
    with open(path_to_vocab_file, "r") as v:
        for word in v:
            if len(word_to_id) >= vocab_size - 3:
                break
            word_to_id[word.replace("'","").strip()]

    # Adding special meta-words
    word_to_id[start]
    word_to_id[end]
    word_to_id[unk]

    # All other words will just be UNK
    word_to_id = defaultdict(lambda: word_to_id[unk], word_to_id)

    # Making ID-to-Word dict
    id_to_word = {v: k for k, v in word_to_id.items()}
    return word_to_id, id_to_word


# TODO: Consider whether to split on "[.?!]+" even though they have an entry in the dictionary
def get_data(path):
    i = 0
    train_x = []
    for file in os.listdir(path):
        with open(path + "/" + file, "r", encoding="utf8") as rfile:
            text = rfile.readline().replace("'","").replace("<br /><br />", ".")
            sent = filter(None, re.split("[.?!]+", text))

            for sentence in sent:
                sent_x = []
                token_word = text_to_word_sequence(sentence)

                for word in token_word:
                    vec = np.zeros(vocab_size, dtype = np.int8)
                    vec[w2i[word] - 1] = 1
                    sent_x.append(vec)

                if len(sent_x) == 0:
                    continue

                if i == 2000:
                    break

                # TODO: Temp_x should have the start and end tokens added
                # TODO: What happens if the end token is cut off by pad_sequences?

                # Pad the sentence as needed
                padded_sent = sent_x
                if len(sent_x) < max_sent_length:
                    pad_length = max_sent_length - len(sent_x)
                    pad = np.zeros((pad_length, vocab_size), dtype = np.int8)
                    padded_sent = np.concatenate((sent_x, pad), axis = 0)
                elif len(sent_x) > max_sent_length:
                    padded_sent = sent_x[:max_sent_length]

                assert(len(padded_sent) == max_sent_length)
                train_x.append(padded_sent)

                i = i + 1

    return np.asarray(train_x)

print('Build vocab...')
w2i, i2w = make_vocab(vocab_file)

print('Get the positive training data...')
x_train_pos = get_data("./aclimdb/train/pos")
y_train_pos = np.random.random((x_train_pos.shape[0], vocab_size))  # Dummy

print('Statistics for the positive data...')
print('x_train shape:', x_train_pos.shape)
print('y_train shape:', y_train_pos.shape)


print('Build the positive LM...')
model = Sequential()
# The Masking layer checks to see if ALL values in the input contains the mask_value.
# The timestep will be skipped in all downstream layers if that is true.
# Explicitly specifying the input_shape is only necessary in the first layer.
model.add(Masking(mask_value=0, input_shape=(max_sent_length, vocab_size)))

# Inputs are passed one at a time. Inputs are sentences with dimensions (max_sent_length, vocab_size).
# See the following link for details on input_shape: https://bit.ly/2A8SkDV
model.add(LSTM(hidden, input_shape = (max_sent_length, vocab_size), return_sequences = True))
# Output dimension is (max_sent_length, hidden) as return_sequences = True.
# The above layer is the "first" layer. The next layer is the hidden layer.

# Input dimension is the same as the output dimension from the previous layer.
model.add(LSTM(vocab_size, activation = 'softmax'))
# The output is one-dimensional with shape (vocab_size,) activated with a softmax function.
# The model estimates the probability that a given word is positive.
# See https://bit.ly/2zee41f for validation that this is equivalent to adding a Dense layer with softmax.

print(model.summary())

# Use a SGD optimizer so that learning rate and momentum can be defined
sgd = SGD(lr=1e-4, momentum = 0.9)

# Compile the model with the binary crossentropy, as we are predicting two labels.
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

print('Train the positive LM...')
model.fit(x_train_pos, y_train_pos,
          batch_size=batch_size,
          epochs=15)




