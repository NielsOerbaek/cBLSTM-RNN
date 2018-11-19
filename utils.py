import numpy as np
import prepros as pp


def generate_embedding_matrix(glove_filename, w2i):
    embeddings_index = dict()
    f = open(glove_filename)
    i = 0
    print("Generating Pretrained Embedding Matrix")
    for line in f:
        values = line.split()
        word = "".join(values[0:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs
        i += 1
        if i % 500 == 0:
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

    return embedding_matrix


def append_data(x_array, y_array, data, label, w2i):
    for i in data:
        for j in data[i]:
            x_array.append(np.array(pp.words_to_ids(data[i][j], w2i)))
            y_array.append(label)


def make_binary_classifier_dataset(pos_data, neg_data, w2i):
    X = []
    y = []
    append_data(X, y, pos_data, 1, w2i)
    append_data(X, y, neg_data, 0, w2i)
    X = np.array(X)
    y = np.array(y)
    return X, y


# Perplexity functions
def perplexity(review):
    product = 1
    size = 0
    for sentence in review:
        size += len(sentence)
        for wordProb in sentence:
            product *= wordProb
    return product**(-1/size)


def sentence_perplexity(sentence):
    product = 1
    size = len(sentence)
    for wordProb in sentence:
        product *= wordProb
    return product ** (-1 / size)
