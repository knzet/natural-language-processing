from gensim.models import LdaModel

DEMO_TOPIC_NUM  = 30
DEMO_DICT       = 'demo.bow.dict'
DEMO_BOW        = 'demo.bow.mm'
DEMO_LDA        = 'demo.%03d.lda' % DEMO_TOPIC_NUM

def pretty_topics(lda_model):
    """Print LDA model topics as a human-readable lists of words and
    probailities

    Example:
        Topic 1:
            prob1 word1
            prob2 word2
            ...

        Topic 2:
            prob1 word1
            ...
        ...

    Args:
        lda_model: gensim LDA model
    Returns:
        None
    """
    for n, topic in enumerate(lda_model.show_topics(num_topics=20, num_words=10, log=False, formatted=False)):
    	print ('Topic %d:' % (n+1))
    	for pair in list(topic[1]):
    		word = pair[0]
    		prob = pair[1]
    		print ('\t%6.5f %-22s' % (prob, word))
    	print

################################################################################

if __name__ == "__main__":

    from gensim.corpora import Dictionary, MmCorpus

    print( 'Loading raw documents ...')
    docs  = [ l.strip() for l in open('brown.txt', 'r') ]
    print ('%d documents' % len(docs))

    print ('Preparing texts ...')
    texts = [ [ w for w in d.lower().split() ]
              for d in docs ]

    print ('Building dictionary of terms ...')
    dictionary = Dictionary(texts)
    print( '%d word types' % len(dictionary))

    print ('Filtering infrequent and frequent terms ...')
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    print ('%d word types, after filtering' % len(dictionary))

    print ('Saving dictionary (%s)...' % DEMO_DICT)
    dictionary.save(DEMO_DICT)

    print( 'Building bag-of-words corpus ...')
    bow_corpus = [ dictionary.doc2bow(t) for t in texts ]

    print( 'Serializing corpus (%s) ...' % DEMO_BOW)
    MmCorpus.serialize(DEMO_BOW, bow_corpus)

    size = int(len(bow_corpus) * 4 / 5)
    training = bow_corpus[:size]
    testing = bow_corpus[size:]

    print( 'Training LDA w/ %d topics on first %d texts ...' % (DEMO_TOPIC_NUM, len(training)))
    lda = LdaModel(training, id2word=dictionary, num_topics=DEMO_TOPIC_NUM, passes=5)

    print( 'Saving LDA model (%s) ...' % DEMO_LDA)
    lda.save(DEMO_LDA)

    print( 'Random subset of topics:')
    print( lda.print_topics() )

    print( 'Computing perplexity on %d held-out documents ...' % len(testing))
    perplexity = 2 ** -(lda.log_perplexity(testing))
    print( 'Perplexity: %.2f' % perplexity)
