from nltk import word_tokenize
from collections import defaultdict
import os
import json

rel_path_to_train = "/aclimdb/train/pos/"

pos_data = dict()
for i,filename in enumerate(os.listdir(os.getcwd() + rel_path_to_train)):
	if(i%500==0):
		print (i)

	with open("." + rel_path_to_train + filename, "r") as f:
		for line in f:
			review = dict()
			for j,sentence in enumerate(line.split(".")):
				review[j] = word_tokenize(sentence)
	pos_data[i] = review

f = open("pos_reviews.json", "a")
f.write(json.dumps(pos_data))