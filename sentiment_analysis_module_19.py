
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from nltk.tokenize import word_tokenize

from nltk.classify import ClassifierI
from statistics import mode #to choose who got the most votes


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v= c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf
short_pos = open("Data/positive.txt","r").read()
short_neg = open("Data/negative.txt","r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append((r,"pos"))

for r in short_neg.split('\n'):
    documents.append((r,"neg"))

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

    
all_words = nltk.FreqDist(all_words)
##print(all_words.most_common(15))
######
####print(all_words["stupid"])

# we need limited words
word_features = list(all_words.keys())[:5000]

#document = list of words
def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

##print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
##
featuresets = [(find_features(rev), category) for (rev, category) in documents]


random.shuffle(featuresets)

###positive data example
training_set = featuresets[:10000]
testing_set = featuresets[10000:]

###negative data example
##training_set = featuresets[100:]
##testing_set = featuresets[:100]


# posterior = prior occurences x liklihood/evidence

# Naive Bayes
####
classifier = nltk.NaiveBayesClassifier.train(training_set)

##classifier_f = open("naivebayes.pickle","rb")
##classifier = pickle.load(classifier_f)
##classifier_f.close()

print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)      


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier Algo accuracy percent: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)



##GuassianNB_classifier = SklearnClassifier(GaussianNB())
##GuassianNB_classifier.train(training_set)
##print("GuassianNB_classifier Algo accuracy percent: ", (nltk.classify.accuracy(GuassianNB_classifier, testing_set))*100)



BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier Algo accuracy percent: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


LogisticRegressionNB_classifier = SklearnClassifier(LogisticRegression())
LogisticRegressionNB_classifier.train(training_set)
print("LogisiticRegressionNB_classifier Algo accuracy percent: ", (nltk.classify.accuracy(LogisticRegressionNB_classifier, testing_set))*100)


SGDClassifierNB_classifier = SklearnClassifier(SGDClassifier())
SGDClassifierNB_classifier.train(training_set)
print("SGDClassifierNB_classifier Algo accuracy percent: ", (nltk.classify.accuracy(SGDClassifierNB_classifier, testing_set))*100)

##SVCNB_classifier = SklearnClassifier(SVC())
##SVCNB_classifier.train(training_set)
##print("SVCNB_classifier Algo accuracy percent: ", (nltk.classify.accuracy(SVCNB_classifier, testing_set))*100)

LinearSVCNB_classifier = SklearnClassifier(LinearSVC())
LinearSVCNB_classifier.train(training_set)
print("LinearSVCNB_classifier Algo accuracy percent: ", (nltk.classify.accuracy(LinearSVCNB_classifier, testing_set))*100)

NuSVCNB_classifier = SklearnClassifier(NuSVC())
NuSVCNB_classifier.train(training_set)
print("NuSVCNB_classifier Algo accuracy percent: ", (nltk.classify.accuracy(NuSVCNB_classifier, testing_set))*100)



voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegressionNB_classifier,
                                  SGDClassifierNB_classifier,
                                  LinearSVCNB_classifier,
                                  NuSVCNB_classifier )

print("voyted_classifier Algo accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)


#   #print("Classification: ", voted_classifier.classify(testing_set[0][0]), "Confidence %:", voted_classifier.confidence(testing_set[0][0])*100)
##
##print("Classification: ", voted_classifier.classify(testing_set[1][0]), "Confidence %:", voted_classifier.confidence(testing_set[1][0])*100)
##print("Classification: ", voted_classifier.classify(testing_set[2][0]), "Confidence %:", voted_classifier.confidence(testing_set[2][0])*100)
##print("Classification: ", voted_classifier.classify(testing_set[3][0]), "Confidence %:", voted_classifier.confidence(testing_set[3][0])*100)
##print("Classification: ", voted_classifier.classify(testing_set[4][0]), "Confidence %:", voted_classifier.confidence(testing_set[4][0])*100)
##








