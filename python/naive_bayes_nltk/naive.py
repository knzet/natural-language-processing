import nltk
from nltk.util import ngrams
import math
import numpy


def trainNaiveBayes(D, C):
    #returns log P(x) and log P(w|c)
    nDoc = len(D)
    logprior = {}
    loglikelihood = {}
    bigdoc = {"POSITIVE": "", "NEGATIVE": ""}
    for c in C:  # C=["POSITIVE","NEGATIVE"]
        #NDoc=number of documents in D
        #nDoc=len(D)
        #Nc = number of documents from D in class c
        nc = 0
        for document in D:
            if (document.c == c):
                nc += 1
            #else:
            #    print(document.c)

        logprior[c] = math.log10(nc / nDoc)
        #V=vocabulary of D # unique unigram list from nltk
        for d in D:
            if (d.c == c):
                bigdoc[c] += d.text
        sumWiC = 0
        # bigdoc[c] is a string with all documents in class concatenated, then we split and generate unigrams -> V
        #V = list(ngrams(bigdoc[c].split(), 1))
        words = nltk.tokenize.word_tokenize(bigdoc[c])
        V = dict(nltk.FreqDist(
            words).items())  # this returns {'word1': 1, 'word2': 1, ...}
        #bigdoc={'+':["123','1234'].'-'}
        sumWiC = 0
        for w in V:  # TODO: maybe use an nltk builtin for this sum
            # calculate sum over all w(i) in V, for denominator of loglikelihood
            #sumWiC+=bigdoc[c].count(w)+1 # count occurrences of substring
            sumWiC += V[w]  # todo: replace with a reduce
        # print(V["the"])
        for w, countWC in V.items():  # calculate P(w|c) terms
            #countWC=V[w]
            #countWC=bigdoc[c].count(w) #number of occurrences of w in bigdoc[c]
            loglikelihood[w, c] = math.log10(
                (countWC + 1) / (sumWiC))  # laplace add-one smoothing
    # print(logprior)
    # print(loglikelihood)
    # print(V)
    return logprior, loglikelihood, V


class Document:
    def __init__(self, text, c):
        self.text = text
        self.c = c


docs = []
docs.append(Document("test of of of of of document ", "POSITIVE"))
docs.append(Document("test document test of the test document ", "POSITIVE"))
docs.append(Document("bad bad sentence the ", "NEGATIVE"))
docs.append(Document("strangely bad words in order ", "NEGATIVE"))

C = {"POSITIVE", "NEGATIVE"}


def parseTrainingData(fname):
    with open(fname) as f:
        lines = f.read().split('\n')
    sents = list(map(lambda line: line.split('\t'), lines))
    # print(sents[-1])
    sents = sents[0:-1]  # trailing newline
    docs = list(map(lambda sent: Document(sent[1].lower(), sent[2]), sents))
    # print(docs[500].c)

    return docs


docs = parseTrainingData('training.txt')
# print(docs)
logprior, loglikelihood, V = trainNaiveBayes(docs, C)
# trainNaiveBayes({d1, d2}, {"POSITIVE"})


def testNaiveBayes(testdoc, logprior, loglikelihood, C, V):  #returns best c
    sum = {"POSITIVE": 0, "NEGATIVE": 0}
    for c in C:
        sum[c] = logprior[c]
        for word in testdoc.text.split():
            for vWord in V:
                if (word == vWord):
                    #this bit isnt working, so all the weights are coming out the same
                    if (word, c) in loglikelihood:
                        # print(word)
                        sum[c] += loglikelihood[word, c]
    print(max(sum.items(), key=lambda k: k[1]))
    # print(sum)  # for some reason the loglikelihood is coming out the same
    return max(sum.items(), key=lambda k: k[1])  # return max, comparing value


alltestdata = parseTrainingData('test.txt')
# print(loglikelihood)


def testFullData(alltestdata):
    return list(
        map(lambda doc: testNaiveBayes(doc, logprior, loglikelihood, C, V),
            alltestdata))


# print(testFullData(alltestdata))
testFullData(alltestdata)

# print(V['singapore-based'])
# testdoc = Document("test of bad bad bad bad bad sentence bad", "NEGATIVE")
# print(doc[4].text)
# # test = testNaiveBayes(doc, logprior, loglikelihood, C, V)
# print(test)

# TODO: table of results: accuracy, precision, recall
#   column for original, nltk stoplist version
