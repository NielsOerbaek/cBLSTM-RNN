import os
import re  # regex
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.utils.data_utils import get_file
import pickle
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict

max_sent_length = 100  # At least 93 % of all sentences within 40 words
vocab_size = 20000 + 3
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
    train_x = []
    for file in os.listdir(path):
        with open(path + "\\" + file, "r", encoding="utf8") as rfile:
            text = rfile.readline().replace("'","").replace("<br /><br />", ".")
            sent = filter(None, re.split("[.?!]+", text))

            if ("Baby" in text) & ("Mama" in text) & ("central" in text):
                if ("gauge" in text) & ("neutered" in text) & ("director" in text):
                    print("Match found in file: " + file)

            for sentence in sent:
                temp_x = []
                token_word = text_to_word_sequence(sentence)

                for word in token_word:
                    temp_x.append(w2i[word])

                if len(temp_x) == 0:
                    continue

                # TODO: Temp_x should have the start and end tokens added
                # TODO: What happens if the end token is cut off by pad_sequences?
                train_x.append(temp_x)

    return train_x


print('Build vocab...')
w2i, i2w = make_vocab(vocab_file)

print('Get the data...')
x_train_pos = get_data("./aclimdb/train/pos")
x_train_neg = get_data("./aclimdb/train/neg")

y_train_pos = [1] * len(x_train_pos)
y_train_neg = [0] * len(x_train_neg)

x_train = x_train_pos + x_train_neg
y_train = y_train_pos + y_train_neg

print('Pad the sentences...')
x_train = pad_sequences(x_train, maxlen = max_sent_length)
print('x_train shape:', x_train.shape)
print('y_train shape:', len(y_train))

print('Build model...')
model = Sequential()
model.add(Embedding(vocab_size + 1, 128, mask_zero=True))  # Padding = 0 adds one to the vocab_size
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=32,
          epochs=15)




