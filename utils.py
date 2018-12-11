import numpy as np
import prepros as pp
import math
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def generate_embedding_matrix(glove_filename, glove_size, w2i):
    embeddings_index = dict()
    f = open(glove_filename)
    i = 0
    print("Generating Pretrained Embedding Matrix")
    for line in f:
        values = line.split()
        word = "".join(values[0:-glove_size])
        coefs = np.asarray(values[-glove_size:], dtype='float32')
        embeddings_index[word] = coefs
        i += 1
        if i % 500 == 0:
            print(i)
    f.close()

    embedding_matrix = np.zeros((pp.vocab_size, glove_size))
    for word, index in zip(w2i.keys(), w2i.values()):
        if index > pp.vocab_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
            else:
                print("Not found in glove:", word, index)

    return embedding_matrix


# Use this function to re-split the positive or negative data from 50/50 to 90/10
def split_data(data1, data2):
    return train_test_split((data1 + data2), test_size=0.1, shuffle=False)


def append_sentences(x_array, y_array, data, label, w2i):
    for i in data:
        for j in i:
            x_array.append(np.array(pp.words_to_ids(data[i][j], w2i)))
            y_array.append(label)


def append_reviews(x_array, y_array, data, label, w2i, offset=0):
    for i, review in enumerate(data):
        dataset_i = i + offset
        x_array.append(np.zeros((len(review), pp.max_sent_length)))
        y_array.append(label)
        for j, sentence in enumerate(review):
            x_array[dataset_i][j] = np.array(pp.words_to_ids(sentence, w2i))


def make_binary_classifier_sentence_dataset(pos_data, neg_data, w2i):
    X = []
    y = []
    append_sentences(X, y, pos_data, np.full((pp.max_sent_length, 1), 1), w2i)
    append_sentences(X, y, neg_data, np.full((pp.max_sent_length, 1), 0), w2i)
    X = np.array(X)
    y = np.array(y)
    return X, y


def make_binary_classifier_review_dataset(pos_data, neg_data, w2i):
    X = []
    y = []
    append_reviews(X, y, pos_data, 1, w2i)
    append_reviews(X, y, neg_data, 0, w2i, offset=len(pos_data))
    return X, y


def make_language_model_sentence_dataset(data1, w2i):
    x = []
    for i, review in enumerate(data1):
        for j in review:
            x.append(np.array(pp.words_to_ids(j, w2i)))
    # Shuffle for good orders sake
    x = shuffle(np.array(x), random_state=420)
    return x


# Made this aux function to make sure we make the same train/test split for the LMs and the BC
# USE: Generate the LM-dataset using the function above, then transform the train and test parts separately using this.
def from_lm_to_bc_dataset(positive_sentences, negative_sentences):
    X = []
    y = []
    for s in positive_sentences:
        X.append(s)
        y.append(np.full((pp.max_sent_length, 1), 1))
    for s in negative_sentences:
        X.append(s)
        y.append(np.full((pp.max_sent_length, 1), 0))
    X, y = shuffle([np.array(X), np.array(y)], random_state=420)
    return X, y


# Perplexity functions
def perplexity(review):
    log_sum = 0
    size = 0
    for sentence in review:
        size += len(sentence)
        for wordProb in sentence:
            log_sum += math.log(wordProb)
    # return math.exp(sum)**(-1/size)
    return log_sum / (-size)


def sentence_perplexity(sentence):
    product = 1
    size = len(sentence)
    for wordProb in sentence:
        product *= wordProb
    return product ** (-1 / size)
