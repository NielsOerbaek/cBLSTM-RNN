from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

# Helper functions
def perplexity(review):
    product = 1
    size = 0
    for sentence in review:
        size += len(sentence)
        for wordProb in sentence:
            product *= wordProb
    return product**(-1/size)


# Preprocessing

# Load the preprocessed data files

# Do other important stuff needed here

# Define problem properties
samples = 1 # Number of sentences
n_timesteps = 10 # Sentence size after padding
features = 1 # Output size
hidden_units = n_timesteps * 2 # Is this wrong?

# define LSTM
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n_timesteps, features)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# train LSTM
for epoch in range(1000):
    # generate new random sequence

    # fit model for one epoch on this sequence
    model.fit(X, y, epochs=1, batch_size=1, verbose=2)

# evaluate LSTM
X, y = get_sequence(n_timesteps)
yhat = model.predict_classes(X, verbose=0)
for i in range(n_timesteps):
    print('Expected:', y[0, i], 'Predicted', yhat[0, i])