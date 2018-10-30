from nltk import word_tokenize, sent_tokenize
import os
import json
import pickle
from collections import defaultdict
import re

paths = ["/aclimdb/train/pos/", "/aclimdb/train/neg/", "/aclimdb/test/pos/", "/aclimdb/test/neg/"]
file_names = ["train_pos", "train_neg", "test_pos", "test_neg"]
data_folder = "./data/"
vocab_file = "/aclimdb/imdb.vocab"
unk_token = "UNK"
start_token = "<s>"
end_token = "</s>"
vocab_size = 10003


def make_vocab(path_to_vocab_file):
    # Making Word-to-ID dict
    word_to_id = defaultdict(lambda: len(word_to_id))
    with open("." + path_to_vocab_file, "r") as v:
        for i, word in enumerate(v):
            if i >= 10000:
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


def get_data_dict(path_to_data, limit=0):
    # max_sentence_length = 0
    data = dict()
    # RegEx'es to help the sentence segmentation along.
    nsf1 = re.compile("(\w)\.(\w)")
    nsf2 = re.compile("(\w)\.\.?\.(.)")
    for i, filename in enumerate(os.listdir(os.getcwd() + path_to_data)):
        if i == limit - 1:
            break
        if limit == 0:
            limit = 12500;
        if i % 500 == 0:
            print(str(i / limit * 100) + "%")
        with open("." + path_to_data + filename, "r") as f:
            review = dict()
            for line in f:
                # To lowercase and fixing examples with no space after periods.
                line = str.lower(line.replace("<br /><br />", "."))
                line = nsf1.sub("\g<1>. \g<2>", nsf2.sub("\g<1>. \g<2>", line))
                for j, sentence in enumerate(sent_tokenize(line)):
                    if len(sentence) > 2:
                        words = [start_token]
                        for word in word_tokenize(sentence):
                            words.append(word)
                        words.append(end_token)
                        words = make_sentence_fixed_length(words)
                        # if len(words) > max_sentence_length:
                            # max_sentence_length = len(words)
                            # print(max_sentence_length, filename, words)
                        review[j] = words
            data[i] = review
    # print("\nThe longest sentence we have is", max_sentence_length)
    return data


def make_sentence_fixed_length(tokenized_sentence, length=300):
    while len(tokenized_sentence) < length:
        tokenized_sentence.append(unk_token)
    return tokenized_sentence[0:length]


def words_to_ids(tokenized_sentence, w2i_vocab):
    for i, word in enumerate(tokenized_sentence):
        tokenized_sentence[i] = int(w2i_vocab[word])
    return tokenized_sentence


def save_data_to_pickle(data, filename):
    pickle.dump(data, open(data_folder + filename, "wb"))


def load_data_from_pickle(filename):
    return pickle.load(open(filename, "rb"))


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
    return load_data_from_pickle(data_folder + filenames + str(limit) + ".pickle")


def load_all_data_files(limit=0):
    dicts = []
    if limit == 0:
        limit = "ALL"
    for i in range(4):
        print("Unpacking pickle: " + file_names[i] + str(limit))
        dicts.append(load_data_from_pickle(data_folder + file_names[i] + str(limit) + ".pickle"))
    return dicts[0], dicts[1], dicts[2], dicts[3]


