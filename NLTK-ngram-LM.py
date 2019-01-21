import numpy as np
import math
import pickle
from sklearn.utils import shuffle
from nltk.util import everygrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE 


# Our Preprocessing Library
import prepros as pp
import utils

num_reviews = 0
train_positive = False
model_name = "Alt-NLTK-LM-negative"

# -- Preprocessing
print("Making vocab dicts of size " + str(pp.vocab_size))
w2i, i2w = pp.make_vocab(pp.vocab_file)

# Once you have generated the data files, you can outcomment the following line.
# pp.generate_data_files(num_reviews)
train_pos, train_neg, test_pos, test_neg = pp.load_all_data_files(num_reviews)

# Make the 90/10 data split instead of default 50/50
if train_positive:
    train, test = utils.split_data(train_pos, test_pos)
else:
    train, test = utils.split_data(train_neg, test_neg)

# Convert to sentence level dataset
train_X = utils.make_language_model_sentence_dataset_nltk(train)
test_X = utils.make_language_model_sentence_dataset_nltk(test)
num_samples = len(train_X)

print("Samples: ", num_samples)

tr, vo = padded_everygram_pipeline(5,train_X)

lm = MLE(5)

lm.fit(tr,vo)

print(lm.counts)
print(lm.generate(30))

print("Saving LM to ./model/"+model_name)
pickle.dump(lm, open("./model/"+model_name, "wb"))

print("Sanity Check: Load same model and try to generate with it")
lm2 = pickle.load(open("./model/"+model_name, "rb"))
print(lm2.counts)
print(lm2.generate(30))
