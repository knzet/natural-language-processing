import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
import math
import numpy
from functools import reduce


def trainNaiveBayes(D, C):
    # returns log P(x) and log P(w|c)
    nDoc = len(D)  # NDoc=number of documents in D

    logprior = {}
    loglikelihood = {}
    bigdoc = {"POSITIVE": "", "NEGATIVE": ""}
    for c in C:  # C=["POSITIVE","NEGATIVE"]
        # Nc = number of documents from D in class c
        nc = 0
        for d in D:
            if (d.c == c):
                nc += 1
                bigdoc[c] += d.text  # bigdoc={'+':["123','1234'].'-'}
        logprior[c] = math.log10(nc / nDoc)

        # bigdoc[c] is a string with all documents in class concatenated, then we split and generate unigrams -> V
        #V = list(ngrams(bigdoc[c].split(), 1))

        # remove stopwords
        stopWords = stopwords.words('english')
        words = nltk.tokenize.word_tokenize(bigdoc[c])
        nonStopWords = list(filter(lambda w: w not in stopWords, words))
        # print(words[:10])
        # print(nonStopWords[:10])
        # V=vocabulary of D (unique unigram frequency list)
        V = dict(nltk.FreqDist(nonStopWords).items()
                 )  # this returns {'word1': 1, 'word2': 1, ...}
        # for w in nonStopWords:  # TODO: maybe use an nltk builtin for this sum
        #     # calculate sum over all w(i) in V, for denominator of loglikelihood
        #     # sumWiC+=bigdoc[c].count(w)+1 # count occurrences of substring
        #     sumWiC += V[w] + 1  # todo: replace with a reduce

        sumWiC = 0  # going to be denominator of loglikelihood
        for w, countWC in V.items():
            sumWiC += countWC + 1
        for w, countWC in V.items():  # calculate P(w|c) terms
            # countWC=V[w]
            # countWC=bigdoc[c].count(w) #number of occurrences of w in bigdoc[c]
            loglikelihood[w, c] = math.log10(
                (countWC + 1) / (sumWiC))  # laplace add-one smoothing
    # print(logprior)
    print('P(hebrew|pos): ', loglikelihood[('hebrew', 'POSITIVE')])
    print('P(wish|neg): ', loglikelihood[('wish', 'NEGATIVE')])
    # print(V)
    return logprior, loglikelihood, V


class Document:
    def __init__(self, text, c):
        self.text = text
        self.c = c


C = {"POSITIVE", "NEGATIVE"}


def parseTrainingData(fname):
    with open(fname) as f:
        lines = f.read().split('\n')
    sents = list(map(lambda line: line.split('\t'), lines))
    sents = sents[0:-1]  # trailing newline
    docs = list(map(lambda sent: Document(sent[1].lower(), sent[2]), sents))

    return docs


docs = parseTrainingData('training.txt')
# print(docs)
logprior, loglikelihood, V = trainNaiveBayes(docs, C)
# trainNaiveBayes({d1, d2}, {"POSITIVE"})





# test ####################

def testNaiveBayes(testdoc, logprior, loglikelihood, C, V):  # returns best c
    sum = {"POSITIVE": 0, "NEGATIVE": 0}
    for c in C:
        sum[c] = logprior[c]
        for word in testdoc.text.split():
            for vWord in V:
                if (word == vWord):
                    # this bit isnt working, so all the weights are coming out too big (negative)
                    if (word, c) in loglikelihood:
                        sum[c] += loglikelihood[word, c]
    # print(testdoc.text + " " + str(max(sum.items(), key=lambda k: k[1])[0]))
    # print(sum)  # for some reason the loglikelihood is coming out the same
    return testdoc.c, max(sum.items(),
                          key=lambda k: k[1])  # return max, comparing value


alltestdata = parseTrainingData('test.txt')

# print(loglikelihood)


def testFullData(alltestdata):
    return list(
        map(lambda doc: testNaiveBayes(doc, logprior, loglikelihood, C, V),
            alltestdata))


# print(testFullData(alltestdata))
doc = testFullData(alltestdata)
# count the number of docs we predicted correctly
tp = 0
fp = 0
tn = 0
fn = 0
# print(filter(lambda d: d[0] == d[1][0], doc))
pcount = 0
ncount = 0
for d in doc:
    # does the test doc class match the predicted class?
    # if d[0] == d[1][0]:
    #     tp += 1

    if d[0] == "POSITIVE":
        # pcount += 1
        if d[1][0] == "NEGATIVE":
            fp += 1
        if d[1][0] == "POSITIVE":
            tp += 1
    if d[0] == "NEGATIVE":
        # ncount += 1
        if d[1][0] == "POSITIVE":
            fn += 1
        if d[1][0] == "NEGATIVE":
            tn += 1

# print('tp: ', tp)
# print('fp: ', fp)
# print('tn: ', tn)
# print('fn: ', fn)
# print('recall: ', tp / (tp + fn))
# print('precision: ', tp / (tp + fp))
# print('specificity: ', tn / (tn + fp))
# print('false positive: ', fp / (tn + fp))
# print("tp/all" :str(tp / len(doc))) # accuracy=number correct pos+neg predictions/total predictions
# map(lambda x: x, fulldata)

# print(V['singapore-based'])
# testdoc = Document("test of bad bad bad bad bad sentence bad", "NEGATIVE")
# print(doc[4].text)
# # test = testNaiveBayes(doc, logprior, loglikelihood, C, V)
# print(test)

# TODO: table of results: accuracy, precision, recall
#   column for original, nltk stoplist version
