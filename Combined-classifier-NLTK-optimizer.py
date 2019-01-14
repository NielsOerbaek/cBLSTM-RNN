from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from sklearn.utils import shuffle
from scipy.interpolate import interp1d
import pickle
import numpy as np

import prepros as pp
import utils

model_folder_path = "./model/"
positive_LM_filename = model_folder_path + "NLTK-LM-positive"
negative_LM_filename = model_folder_path + "NLTK-LM-negative"
binary_classifier_filename = model_folder_path + "BLSTM-Glove-300-05.hdf5"
num_samples = 0

print("models used:", positive_LM_filename, negative_LM_filename, binary_classifier_filename)

# Vocab files
print("Making vocab dicts of size " + str(pp.vocab_size))
w2i, i2w = pp.make_vocab(pp.vocab_file)

# Once you have generated the data files, you can outcomment the following line.
# pp.generate_data_files(num_samples)
train_pos, train_neg, test_pos, test_neg = pp.load_all_data_files(num_samples)

# Split the data by 90/10 instead of 50/50
train_pos, test_pos = utils.split_data(train_pos, test_pos)
train_neg, test_neg = utils.split_data(train_neg, test_neg)

train_X, train_y = utils.make_binary_classifier_review_dataset(train_pos, train_neg, w2i)
train_X, train_y = shuffle(train_X, train_y, random_state=420)

test_X, test_y = utils.make_binary_classifier_review_dataset(test_pos, test_neg, w2i)
test_X, test_y = shuffle(test_X, test_y, random_state=420)


print("Loading Language Models...", end="", flush=True)
print("...The positive one...", end="", flush=True)
positive_LM = pickle.load(open(positive_LM_filename, "rb"))
print("...and the negative one...", end="", flush=True)
negative_LM = pickle.load(open(negative_LM_filename, "rb"))
print("...done!")

def extract_probabilities_from_sentence(sentence, lm):
    probabilities = []
    sentence = pp.ids_to_words(sentence.tolist(), i2w)
    for i, word in enumerate(sentence):
        if word == "<MASK>":
            break
        hist = 4 if i >= 0 else i
        while True:
            score = lm.score(word, context=sentence[i-hist:i])
            if score > 0 or hist <= 0: break
            hist -= 1
        probabilities.append(score)
    return probabilities


def extract_probabilities_from_review(review, lm):
    probabilities = []
    for i, sentence in enumerate(review):
        probabilities.append(extract_probabilities_from_sentence(sentence, lm))
    return probabilities


def classify_review_by_lm(review, pb, pt):
    pos_probs = extract_probabilities_from_review(review, positive_LM)
    neg_probs = extract_probabilities_from_review(review, negative_LM)

    pos_perplexity = utils.perplexity(pos_probs)
    neg_perplexity = utils.perplexity(neg_probs)

    delta = pos_perplexity - neg_perplexity

    # Init interpolation mappers
    p_map_bottom = interp1d([pb, 0], [1, 0.5])
    p_map_top = interp1d([0, pt], [0.5, 0])

    if delta < 0:
        if delta < pb:
            return 1.0
        return float(p_map_bottom(delta))
    else:
        if delta > pt:
            return 0.0
        return float(p_map_top(delta))


def get_perplexities(review):
    pos_probs = extract_probabilities_from_review(review, positive_LM)
    neg_probs = extract_probabilities_from_review(review, negative_LM)

    pos_perplexity = utils.perplexity(pos_probs)
    neg_perplexity = utils.perplexity(neg_probs)

    return pos_perplexity, neg_perplexity


def classify_review_by_bc(review, predictions):
    probs = []
    for i, sent in enumerate(review):
        for j, word_id in enumerate(sent):
            if word_id == 0:
                break
            probs.append(predictions[i][j])
    avg_pred = sum(probs) / len(probs)
    return avg_pred[0]


def evaluate_p_maps(factor):
    pb = p_bottom*f
    pt = p_top*f
    hits = 0
    samples = len(train_predictions)
    for p in train_predictions:
        LM_classification = classify_review_by_lm(p[0], pb, pt)
        classification = int(round((LM_classification + p[1]) / 2))
        if p[2] == classification:
            hits += 1
    score = hits / samples
    print("Optimization result - ", "Factor: ", factor, "Acc:", score)
    return score


cali_limit = 1000

print("Calibrating perplexity weights...")
p_bottom = 1000000000
p_top = -10000000000
for i in range(len(train_X[:cali_limit])):
    pos_p, neg_p = get_perplexities(train_X[i])
    delta = pos_p - neg_p

    p_bottom = min(p_bottom, delta)
    p_top = max(p_top, delta)
    if i % 100 == 0: print("Calibrating:", i, "of", cali_limit)


print("Loading BLSTM-BC model")
binary_classifier = load_model(binary_classifier_filename)

extra_train = 1000
print("Using", extra_train, "of", len(test_X), "samples for interpolation optimization")
print("Generating predictions for optimization...")
train_predictions = []
for i in range(len(test_X[-extra_train:])):
    bc_predictions = binary_classifier.predict(test_X[i], verbose=2)

    LM_classification = classify_review_by_lm(test_X[i], p_bottom, p_top)
    BC_classification = classify_review_by_bc(test_X[i], bc_predictions)
    train_predictions.append((test_X[i], BC_classification, test_y[i]))
    if i % 50 == 0: print("Generating:", i, "of", extra_train)

print("Before optimization:", "Bottom:", p_bottom, "Top:", p_top)
f = 1.0
old_score = evaluate_p_maps(f)
new_score = evaluate_p_maps(f*0.5)

while new_score >= old_score:
    f = f*0.5
    old_score = new_score
    new_score = evaluate_p_maps(f*0.5)

p_bottom *= f
p_top *= f
print("After optimization:", "Bottom:", p_bottom, "Top:", p_top)

LM_hits = 0
BC_hits = 0
Comb_hits = 0
test_X = test_X[:extra_train]
test_y = test_y[:extra_train]
samples = len(test_X)
print("Classifying...")
for i in range(samples):
    bc_predictions = binary_classifier.predict(test_X[i], verbose=2)

    LM_classification = classify_review_by_lm(test_X[i], p_bottom, p_top)
    BC_classification = classify_review_by_bc(test_X[i], bc_predictions)
    classification = int(round((LM_classification + BC_classification) / 2))

    truth = test_y[i]
    if truth == classification:
        Comb_hits += 1
    if truth == round(LM_classification):
        LM_hits += 1
    if truth == round(BC_classification):
        BC_hits += 1

    def p_a(hits):
        return str(round(hits / (i + 1) * 100, 2))+"%"
    
    print("Sample " + str(i+1) + "/" + str(samples),
          "Truth:", truth,
          "LM+BC:", classification,
          "LM:", round(LM_classification,2),
          "\tBC:", round(BC_classification,2),
          "\tLM+BC-acc:", p_a(Comb_hits),
          "\tLM-acc:", p_a(LM_hits),
          "\tBC-acc:", p_a(BC_hits))

print("\n\nDONE! --- Final Accuracy:", Comb_hits/samples)

