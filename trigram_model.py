import sys
from collections import defaultdict
import math
import random
import os
import os.path
import copy
"""
COMS W4705 - Natural Language Processing - Summer 2012 
Prorgramming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    ngrams = []
    seq_copy = copy.deepcopy(sequence)
    seq_copy = ['START'] + seq_copy + ['STOP']

    for i in range(n-2):
        seq_copy = ['START'] + seq_copy

    for i in range(len(seq_copy)-n+1):
        tp = ()
        for j in range(n):
            tp += (seq_copy[i+j],)
        ngrams.append(tp)

    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        uni = []
        bi = []
        tri = []

        for sentence in corpus:
            uni += get_ngrams(sentence, 1)
            bi += get_ngrams(sentence, 2)
            tri += get_ngrams(sentence, 3)

        for tuple in uni:
            self.unigramcounts[tuple] += 1

        for tuple in bi:
            self.bigramcounts[tuple] += 1

        for tuple in tri:
            self.trigramcounts[tuple] += 1

        self.totalcounts = sum(self.unigramcounts.values())

        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if (trigram[0] is 'START') and (trigram[1] is 'START'):
            denominator = self.unigramcounts[('START',)]
        else:
            denominator = self.bigramcounts[trigram[0:2]]

        if denominator is 0:
            return 0.000001
        else:
            return self.trigramcounts[trigram]/denominator

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        denominator = self.unigramcounts[(bigram[0],)]

        if denominator is 0:
            return 0.000001
        else:
            return self.bigramcounts[bigram]/denominator
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.
        denominator = self.totalcounts

        if denominator is 0:
            return 0.000001

        return self.unigramcounts[unigram]/denominator

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return lambda1*self.raw_trigram_probability(trigram) + lambda2*self.raw_bigram_probability(trigram[1:]) + lambda3*self.raw_unigram_probability((trigram[2],))
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        prob = 0
        trigrams = get_ngrams(sentence, 3)
        for tri in trigrams:
            tri_p = self.smoothed_trigram_probability(tri)
            prob += math.log2(tri_p)

        return prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        log_p = 0
        count = 0
        for sentence in corpus:
            log_p += self.sentence_logprob(sentence)
            count += len(sentence)
        l = log_p/count
        return 2 ** -l


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total += 1
            if pp1 < pp2:
                correct += 1
    
        for f in os.listdir(testdir2):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            total += 1
            if pp2 < pp1:
                correct += 1
        
        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment:
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)

