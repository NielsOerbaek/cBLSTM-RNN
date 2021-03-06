from keras.models import load_model
from sklearn.utils import shuffle
from scipy.interpolate import interp1d

import prepros as pp
import utils

model_folder_path = "./model/"
positive_LM_filename = model_folder_path + "Dec-7-cBLSTM-pos-LM-20.hdf5"
negative_LM_filename = model_folder_path + "Dec-8-cBLSTM-neg-LM-20.hdf5"
binary_classifier_filename = model_folder_path + "Dec9-BLSTM-BC-20.hdf5"
num_samples = 0


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


print("Loading Models...")
positive_LM = load_model(positive_LM_filename)
negative_LM = load_model(negative_LM_filename)
binary_classifier = load_model(binary_classifier_filename)


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


def classify_review_by_lm(review, pos_prediction, neg_prediction):
    pos_probs = extract_probabilities_from_review(review, pos_prediction)
    neg_probs = extract_probabilities_from_review(review, neg_prediction)

    pos_perplexity = utils.perplexity(pos_probs)
    neg_perplexity = utils.perplexity(neg_probs)

    delta = pos_perplexity - neg_perplexity

    if delta < 0:
        if delta < p_bottom:
            return 1.0
        return float(p_map_bottom(delta))
    else:
        if delta > p_top:
            return 0.0
        return float(p_map_top(delta))


def get_perplexities(review, pos_prediction, neg_prediction):
    pos_probs = extract_probabilities_from_review(review, pos_prediction)
    neg_probs = extract_probabilities_from_review(review, neg_prediction)

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


print("Calibrating perplexity weights...")
p_bottom = 1000000000
p_top = -10000000000
for i in range(len(train_X[:1000])):
    pos_predictions = positive_LM.predict(train_X[i], verbose=2)
    neg_predictions = negative_LM.predict(train_X[i], verbose=2)

    pos_p, neg_p = get_perplexities(train_X[i], pos_predictions, neg_predictions)
    delta = pos_p - neg_p

    p_bottom = min(p_bottom, delta)
    p_top = max(p_top, delta)

# Init interpolation mappers
p_map_bottom = interp1d([p_bottom, 0], [1, 0.5])
p_map_top = interp1d([0, p_top], [0.5, 0])

LM_hits = 0
BC_hits = 0
Comb_hits = 0
samples = len(test_X)
print("Classifying...")
for i in range(samples):
    pos_predictions = positive_LM.predict(test_X[i], verbose=2)
    neg_predictions = negative_LM.predict(test_X[i], verbose=2)
    bc_predictions = binary_classifier.predict(test_X[i], verbose=2)

    LM_classification = classify_review_by_lm(test_X[i], pos_predictions, neg_predictions)
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

    print("Sample " + str(i) + "/" + str(samples),
          "\tTruth:", truth,
          "\tLM+BC:", classification,
          "\tLM:", round(LM_classification,2),
          "\tBC:", round(BC_classification,2),
          "\tLM+BC-acc:", p_a(Comb_hits),
          "\tLM-acc:", p_a(LM_hits),
          "\tBC-acc:", p_a(BC_hits))

print("\n\nDONE! --- Final Accuracy:", Comb_hits/samples)
