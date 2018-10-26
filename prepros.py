from nltk import word_tokenize
import os
import json
from collections import defaultdict

paths = ["/aclimdb/train/pos/", "/aclimdb/train/neg/", "/aclimdb/test/pos/", "/aclimdb/test/neg/"]
file_names = ["train_pos", "train_neg", "test_pos", "test_neg"]
data_folder = "./data/"
vocab_file = "/aclimdb/imdb.vocab"
unk_token = "UNK"
start_token = "<s>"
end_token = "</s>"

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
	word_to_id = defaultdict(lambda: unk_token, word_to_id)

	# Making ID-to-Word dict
	id_to_word = {v: k for k, v in word_to_id.items()}

	return word_to_id, id_to_word


def get_data_dict(path_to_data, limit=-1):
	data = dict()
	for i, filename in enumerate(os.listdir(os.getcwd() + path_to_data)):
		if i == limit-1:
			break

		if i % 500 == 0:
			print (i)

		with open("." + path_to_data + filename, "r") as f:
			review = dict()
			for line in f:
				for j, sentence in enumerate(line.split(".")):
					if len(sentence) > 0:
						words = [start_token]
						for word in word_tokenize(sentence):
							words.append(str.lower(word))
						words.append(end_token)
						review[j] = words
		data[i] = review
	return data


def save_data_to_json_file(data, filename):
	f = open(data_folder + filename, "a")
	f.write(json.dumps(data))


w2i, i2w = make_vocab(vocab_file)

for i in range(4):
	print("making " + file_names[i])
	save_data_to_json_file(get_data_dict(paths[i], 1000), file_names[i] + "1000" + ".json")


