import numpy as np
from keras.models import load_model
from sklearn.utils import shuffle

import prepros as pp
import utils

model_folder_path = "./model/"
positive_LM_filename = model_folder_path + "positive-LM-emb-model-40.hdf5"
negative_LM_filename = model_folder_path + "negative-LM-emb-model-40.hdf5"
num_samples = 1000


# Vocab files
print("Making vocab dicts of size " + str(pp.vocab_size))
w2i, i2w = pp.make_vocab(pp.vocab_file)

# Once you have generated the data files, you can outcomment the following line.
# pp.generate_data_files(num_samples)
train_pos, train_neg, test_pos, test_neg = pp.load_all_data_files(num_samples)

test_X, test_y = utils.make_binary_classifier_review_dataset(test_pos, test_neg, w2i)
test_X, test_y = shuffle(test_X, test_y, random_state=420)


print("Loading Models...")
positive_LM = load_model(positive_LM_filename)
negative_LM = load_model(negative_LM_filename)


def extract_probabilities_from_sentence(sentence, prediction):
    probabilities = []
    for i, word_id in enumerate(sentence):
        if word_id == 0:
            break
        probabilities.append(prediction[i][int(word_id)])
    return probabilities


def extract_probabilities_from_review(review, predictions):
    probabilities = []
    for i, sentence in enumerate(review):
        probabilities.append(extract_probabilities_from_sentence(sentence, predictions[i]))
    return probabilities


def classify_review(review, pos_prediction, neg_prediction):
    pos_probs = extract_probabilities_from_review(review, pos_prediction)
    neg_probs = extract_probabilities_from_review(review, neg_prediction)

    pos_perplexity = utils.perplexity(pos_probs)
    neg_perplexity = utils.perplexity(neg_probs)

    print(pos_perplexity, neg_perplexity)

    if pos_perplexity <= neg_perplexity:
        return 1
    else:
        return 0


hits = 0
samples = len(test_X)
print("Classifying...")
for i in range(samples):
    pos_predictions = positive_LM.predict(test_X[i], batch_size=100, verbose=2)
    neg_predictions = negative_LM.predict(test_X[i], batch_size=100, verbose=2)
    classification = classify_review(test_X[i], pos_predictions, neg_predictions)
    truth = test_y[i]
    if truth == classification:
        hits += 1
    print("Sample " + str(i) + " of " + str(samples),
          "\tTruth:", truth,
          "\tOur classification:", classification,
          "\tAccuracy so far:", hits/(i+1))

print("Accuracy:", hits/samples)


