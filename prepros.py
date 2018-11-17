from nltk import word_tokenize, sent_tokenize
from nltk.tokenize import RegexpTokenizer
import numpy as np
import os
import pickle
from collections import defaultdict
import re

paths = ["/aclimdb/train/pos/", "/aclimdb/train/neg/", "/aclimdb/test/pos/", "/aclimdb/test/neg/"]
file_names = ["train_pos", "train_neg", "test_pos", "test_neg"]
data_folder = "./data/"
vocab_file = "/aclimdb/imdb.vocab"
unk_token = "<UNK>"
start_token = "<s>"
end_token = "</s>"
mask_token = "<MASK>"
# When you change these, you need to regenerate the data
vocab_size = 10000 + 4
max_sent_length = 100
tokenizer = RegexpTokenizer(r"\w+'\w+|\w+|\!|\?")


def make_vocab(path_to_vocab_file):
    # Making Word-to-ID dict
    word_to_id = defaultdict(lambda: len(word_to_id))
    word_to_id[mask_token]
    with open("." + path_to_vocab_file, "r") as v:
        for i, word in enumerate(v):
            if i >= vocab_size-4:
                break
            word_to_id[word.strip()]
    # Adding special meta-words
    word_to_id[start_token]
    word_to_id[end_token]
    word_to_id[unk_token]
    # All other words will just be UNK
    word_to_id = defaultdict(lambda: word_to_id[unk_token], word_to_id)

    # Making ID-to-Word dict
    id_to_word = {v: k for k, v in word_to_id.items()}

    return word_to_id, id_to_word


def tokenize_and_pad(sentence):
    words = []
    if len(sentence) > 2:
        words = [start_token]
        for word in tokenizer.tokenize(sentence):
            words.append(word)
        words.append(end_token)
        words = make_sentence_fixed_length(words)
    return words


def get_data_dict(path_to_data, limit=0):
    data = dict()
    # RegEx'es to help the sentence segmentation along.
    nsf1 = re.compile("(\w)\.(\w)")
    nsf2 = re.compile("(\w)\.\.?\.(.)")
    for i, filename in enumerate(os.listdir(os.getcwd() + path_to_data)):
        if i == limit - 1:
            break
        if limit == 0:
            limit = 12500
        if i % 500 == 0:
            print(str(i / limit * 100) + "%")
        with open("." + path_to_data + filename, "r") as f:
            review = dict()
            for line in f:
                # To lowercase and fixing examples with no space after periods.
                line = str.lower(line.replace("<br /><br />", ".").replace(",", "").replace("-", ""))
                line = nsf1.sub("\g<1>. \g<2>", nsf2.sub("\g<1>. \g<2>", line))
                for j, sentence in enumerate(sent_tokenize(line)):
                    words = tokenize_and_pad(sentence)
                    if len(words) > 2:
                        review[j] = words
            data[i] = review
    return data


def make_sentence_fixed_length(tokenized_sentence, length=max_sent_length):
    while len(tokenized_sentence) < length:
        tokenized_sentence.append(mask_token)
    return tokenized_sentence[0:length]


def words_to_ids(tokenized_sentence, w2i_vocab):
    for i, word in enumerate(tokenized_sentence):
        tokenized_sentence[i] = w2i_vocab[word]
    return tokenized_sentence


def ids_to_words(tokenized_sentence, i2w_vocab):
    for i, word in enumerate(tokenized_sentence):
        tokenized_sentence[i] = i2w_vocab[word]
    return tokenized_sentence


def to_one_hot(word_id, vocab_size):
    v = np.full(vocab_size, 0)
    v[word_id] = 1
    return v


def from_one_hot(one_hot_vector):
    for i, v in enumerate(one_hot_vector):
        if v == 1:
            return i


def one_hots_to_sentence(one_hots, i2w_vocab):
    a = []
    for w in one_hots:
        w_id = np.argmax(w)
        a.append(i2w_vocab[w_id])
    s = ""
    for w in a:
        s += " " + w
    return s


def ids_to_sentence(ids, i2w_vocab):
    a = []
    for w in ids:
        a.append(i2w_vocab[w])
    s = ""
    for w in a:
        s += " " + w
    return s


def save_data_to_pickle(data, filename):
    pickle.dump(data, open(data_folder + filename, "wb"))


def load_data_from_pickle(filename):
    return pickle.load(open(data_folder + filename, "rb"))


# For some reason the json files cannot be decoded. Don't know why yet.
def generate_data_files(limit=0):
    for i in range(4):
        print("generating  " + file_names[i])
        if limit == 0:
            limitText = "ALL"
        else:
            limitText = str(limit)
        save_data_to_pickle(get_data_dict(paths[i], limit), file_names[i] + limitText + ".pickle")


def get_data_dicts(limit=0):
    dicts = []
    s = ""
    if limit > 0:
        s = " with limit " + str(limit)
    for i in range(4):
        print("making " + file_names[i] + s)
        dicts.append(get_data_dict(paths[i], limit))
    return dicts[0], dicts[1], dicts[2], dicts[3]


def load_data_file(filename, limit=0):
    print("Unpacking pickle: " + filename + str(limit))
    return load_data_from_pickle(filename + str(limit) + ".pickle")


def load_all_data_files(limit=0):
    dicts = []
    if limit == 0:
        limit = "ALL"
    for i in range(4):
        print("Unpacking pickle: " + file_names[i] + str(limit))
        dicts.append(load_data_from_pickle(file_names[i] + str(limit) + ".pickle"))
    return dicts[0], dicts[1], dicts[2], dicts[3]


