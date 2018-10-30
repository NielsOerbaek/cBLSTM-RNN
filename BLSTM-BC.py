import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import Embedding

# Our Preprocessing Library
import prepros as pp


# Helper functions
def perplexity(review):
    product = 1
    size = 0
    for sentence in review:
        size += len(sentence)
        for wordProb in sentence:
            product *= wordProb
    return product**(-1/size)


# -- Preprocessing
# Vocab files
w2i, i2w = pp.make_vocab(pp.vocab_file)

# Once you have generated the data files, you can outcomment the following line.
pp.generate_data_files()
train_pos, train_neg, test_pos, test_neg = pp.retrieve_data_from_file()

print()

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
        train_y.append(1)

train_X = np.array(train_X)

#Shuffle the data
train_X, train_y = shuffle(train_X, train_y, random_state=42)

print(train_X.shape)

# define LSTM
print("Creating model")
model = Sequential()
model.add(Embedding(10003, 600, input_length=300))
model.add(Bidirectional(LSTM(300)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# train LSTM
model.fit(train_X, train_y, epochs=10, batch_size=100, verbose=1)

# evaluate LSTM
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
    print('Expected:', y[0, i], 'Predicted', yhat[0, i])