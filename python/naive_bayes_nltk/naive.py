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
    Vocab = ""  # all words in the training data
    for d in D:
        Vocab += " "+d.text
    for c in C:  # C=["POSITIVE","NEGATIVE"]
        nc = 0  # number of documents from D in class c
        for d in D:
            if (d.c == c):
                nc += 1
                bigdoc[c] += " "+d.text  # bigdoc={'+':["123','1234'].'-'}
        logprior[c] = math.log10(nc / nDoc)

        # remove stopwords
        stopWords = stopwords.words('english')
        # bigdoc[c] is a string with all documents in the class concatenated, then we tokenize and generate unigrams and unigram frequencies
        classWords = nltk.tokenize.word_tokenize(bigdoc[c])
        nonStopWords = list(filter(lambda w: w not in stopWords, classWords))
        # vocabulary of c (unique unigram frequency list)
        classVocab = dict(nltk.FreqDist(nonStopWords).items()
                          )  # this returns {'word1': 1, 'word2': 1, ...}

        allWords = nltk.tokenize.word_tokenize(Vocab)
        nonstopvocab = list(filter(lambda w: w not in stopWords, allWords))
        allVocab = dict(nltk.FreqDist(nonstopvocab).items())

        sumWiC = 0  # going to be denominator of loglikelihood
        for w in allVocab:  # sum over full vocab, where w in class c
            sumWiC += bigdoc[c].count(w)+1

        # for every word, how often does it occur in each class?
        for w, countWC in classVocab.items():  # calculate P(w|c) terms
            # print(sumWiC)
            # print(countWC)
            # print(w)
            # print("")

            # log(count(w,c)/sum(count(wi,c), for w in V))
            loglikelihood[w, c] = abs(math.log10(
                (countWC + 1) / (sumWiC)))  # laplace add-one smoothing
    # print(logprior)
    # print('P(hebrew|pos): ', loglikelihood[('hebrew', 'POSITIVE')])
    # print('P(wish|neg): ', loglikelihood[('wish', 'NEGATIVE')])
    # print(V)
    return logprior, loglikelihood, allVocab


class Document:
    def __init__(self, text, c):
        self.text = text
        self.c = c


C = {"POSITIVE", "NEGATIVE"}
testdocs = []
testdocs.append(Document("hebrew wish", "POSITIVE"))
testdocs.append(Document("hebrew wish", "POSITIVE"))
testdocs.append(Document("bad word wish", "NEGATIVE"))
testdocs.append(Document("bad word", "NEGATIVE"))
testdocs.append(Document("test 123", "NEGATIVE"))
testdocs.append(Document("test 123", "NEGATIVE"))
testdocs.append(Document("test 123", "NEGATIVE"))
testdocs.append(Document("test 123", "NEGATIVE"))
testdocs.append(Document("test 123", "NEGATIVE"))
testdocs.append(Document("test 123", "NEGATIVE"))

# extract data, sentiment labels


def parseTrainingData(fname):
    with open(fname) as f:
        lines = f.read().split('\n')
    sents = list(map(lambda line: line.split('\t'), lines))
    sents = sents[0:-1]  # remove trailing newline
    # returns document:class pairs
    docs = list(map(lambda sent: Document(sent[1].lower(), sent[2]), sents))

    for c in C:  # C=["POSITIVE","NEGATIVE"]
        nc = 0  # number of documents from D in class c
        for d in docs:
            if (d.c == c):
                nc += 1
        # how many documents in each class?
        print(c, ": ", nc)
    return docs


docs = parseTrainingData('training.txt')
# print(docs)
logprior, loglikelihood, Vocab = trainNaiveBayes(docs, C)
# print(loglikelihood)
print("log probability of each class: ", logprior)


# test ####################

def testNaiveBayes(testdoc, logprior, loglikelihood, C, V):  # returns best c
    sum = {"POSITIVE": 0, "NEGATIVE": 0}
    for c in C:

        sum[c] = logprior[c]
        testwords = nltk.tokenize.word_tokenize(testdoc.text)
        for word in testwords:
            # print(word)
            # if(word in V.keys()):
            # for vWord in V:
            #     if (word == vWord):
            if (word, c) in loglikelihood:
                sum[c] += loglikelihood[word, c]
                # print(sum[c])
    # print(testdoc.text + " " + str(max(sum.items(), key=lambda k: k[1])[0]))
    return testdoc.c, max(sum.items(),
                          key=lambda k: abs(k[1]))  # return max, comparing value


alltestdata = parseTrainingData('test.txt')

# print(loglikelihood)


def testFullData(alltestdata):
    return list(
        map(lambda doc: testNaiveBayes(doc, logprior, loglikelihood, C, Vocab),
            alltestdata))


# print(testFullData(alltestdata))
doc = testFullData(docs)
# count the number of docs we predicted correctly
tp = 0
fp = 0
tn = 0
fn = 0
pcount = 0
ncount = 0
for d in doc:
    # does the actual test doc class match the predicted class?
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

print('tp: ', tp)
print('fp: ', fp)
print('tn: ', tn)
print('fn: ', fn)
print('recall: ', tp / (tp + fn))
print('precision: ', tp / (tp + fp))
print('specificity: ', tn / (tn + fp))
print('false positive: ', fp / (tn + fp))
# print("tp/all" :str(tp / len(doc))) # accuracy=number correct pos+neg predictions/total predictions
# map(lambda x: x, fulldata)

# print(V['singapore-based'])
# testdoc = Document("test of bad bad bad bad bad sentence bad", "NEGATIVE")
# print(doc[4].text)
# # test = testNaiveBayes(doc, logprior, loglikelihood, C, V)
# print(test)

# TODO: table of results: accuracy, precision, recall
#   column for original, nltk stoplist version
