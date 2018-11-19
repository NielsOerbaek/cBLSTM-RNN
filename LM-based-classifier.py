import numpy as np
from keras.models import load_model
from sklearn.utils import shuffle

import prepros as pp
import utils

model_folder_path = "./model/"
positive_LM_filename = model_folder_path + "positive-LM-emb-model-40.hdf5"
negative_LM_filename = model_folder_path + "negative-LM-emb-model-40.hdf5"
num_samples = 50


# Vocab files
print("Making vocab dicts of size " + str(pp.vocab_size))
w2i, i2w = pp.make_vocab(pp.vocab_file)

# Once you have generated the data files, you can outcomment the following line.
pp.generate_data_files(num_samples)
train_pos, train_neg, test_pos, test_neg = pp.load_all_data_files(num_samples)

test_X, test_y = utils.make_binary_classifier_dataset(test_pos,test_neg,w2i)
test_X, test_y = shuffle(test_X, test_y, random_state=420)

print("Loading Models...")
positive_LM = load_model(positive_LM_filename)
negative_LM = load_model(negative_LM_filename)


def extract_probabilities(sentence, prediction):
    probabilities = np.zeros(len(sentence))
    for i, word_id in enumerate(sentence):
        probabilities[i] = prediction[i, word_id]
    return probabilities


def classify(sentence, pos_prediction, neg_prediction):
    pos_probs = extract_probabilities(sentence, pos_prediction)
    neg_probs = extract_probabilities(sentence, neg_prediction)

    pos_perplexity = utils.sentence_perplexity(pos_probs)
    neg_perplexity = utils.sentence_perplexity(neg_probs)

    if pos_perplexity <= neg_perplexity:
        return 1
    else:
        return 0


pos_predictions = positive_LM.predict(test_X, batch_size=100, verbose=2)
neg_predictions = negative_LM.predict(test_X, batch_size=100, verbose=2)

hits = 0
samples = len(test_X)
print("Classifying...")
for i in range(samples):
    classification = classify(test_X[i],pos_predictions[i], neg_predictions[i])
    truth = test_y[i]
    print("Truth:", truth, "Our classification:", classification)
    if truth == classification:
        hits += 1

print("Accuracy:", hits/samples)


