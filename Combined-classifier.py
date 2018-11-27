import numpy as np
from keras.models import load_model
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import normalize

import prepros as pp
import utils

model_folder_path = "./model/"
positive_LM_filename = model_folder_path + "positive-LM-emb-model-40.hdf5"
negative_LM_filename = model_folder_path + "negative-LM-emb-model-40.hdf5"
binary_classifier_filename = model_folder_path + "Nov22-glove-emb-BC-model-01.hdf5"
num_samples = 5000


# Vocab files
print("Making vocab dicts of size " + str(pp.vocab_size))
w2i, i2w = pp.make_vocab(pp.vocab_file)

# Once you have generated the data files, you can outcomment the following line.
pp.generate_data_files(num_samples)
train_pos, train_neg, test_pos, test_neg = pp.load_all_data_files(num_samples)

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

    return pos_perplexity, neg_perplexity


def classify_review_by_bc(sent_pred):
    avg_pred = sum(sent_pred) / len(sent_pred)
    return avg_pred[0]


def get_predictions(review):
    pos_predictions = positive_LM.predict(train_X[i], verbose=2)
    neg_predictions = negative_LM.predict(train_X[i], verbose=2)
    bc_predictions = binary_classifier.predict(train_X[i], verbose=2)

    pos_p, neg_p = classify_review_by_lm(train_X[i], pos_predictions, neg_predictions)
    bc = classify_review_by_bc(bc_predictions)

    return normalize(np.array([pos_p-neg_p, bc]))


print("Defining FF-model")
classifier_model = Sequential()
classifier_model.add(Dense(100))
classifier_model.add(Dense(1, activation="sigmoid"))
classifier_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


print("Making classifier model training data")
classifier_X = np.zeros(shape=(len(train_X), 2))
classifier_y = np.zeros(shape=(len(train_X)))

for i in range(len(train_X)):
    classifier_X[i] = get_predictions(train_X[i])
    classifier_y[i] = train_y[i]
    print(str(round(i / len(train_X) * 100, 2))+"%", end="-", flush=True)
    if i % 20 == 0:
        print()


print("\n", classifier_X.shape)

print("Training classifier model")
classifier_model.fit(classifier_X, classifier_y, epochs=5)

LM_hits = 0
BC_hits = 0
Comb_hits = 0
samples = len(test_X)
print("Classifying...")
for i in range(samples):
    predictions = get_predictions(test_X[i])
    prediction = classifier_model.predict(np.array(predictions))[0, 0]
    classification = int(round(prediction))

    truth = test_y[i]
    if truth == classification:
        Comb_hits += 1

    def p_a(hits):
        return str(round(hits / (i + 1) * 100, 2))+"%"

    print("Sample " + str(i) + "/" + str(samples),
          "\tTruth:", truth,
          "\tPrediction:", round(prediction, 2),
          "\tClassification:", classification,
          "\tAcc:", p_a(Comb_hits))

print("\n\nDONE! --- Final Accuracy:", Comb_hits/samples)
