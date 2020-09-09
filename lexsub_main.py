#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context
from collections import defaultdict

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import operator

import numpy as np
import tensorflow

import gensim
import transformers

from typing import List

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    candidates = []

    for l1 in wn.lemmas(lemma, pos=pos):
        for s1 in l1.synset().lemmas():
            synonym = s1.name().replace('_', ' ')
            if synonym != lemma.replace('_', ' ') and synonym not in candidates:
                candidates.append(synonym)

    return candidates

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # Part 2

    candidates = {}

    for l1 in wn.lemmas(context.lemma, pos=context.pos):
        for s1 in l1.synset().lemmas():
            synonym = s1.name().replace('_', ' ')
            if synonym != context.lemma.replace('_', ' '):
                if synonym not in candidates:
                    candidates[synonym] = s1.count()
                else:
                    candidates[synonym] += s1.count()

    if bool(candidates):
        return max(candidates.keys(), key=(lambda k: candidates[k]))
    else:
        return 'smurf'


def wn_simple_lesk_predictor(context : Context) -> str:
    # Part 3

    candidates = defaultdict(int)

    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    ctxt = []
    ctxt += context.left_context
    ctxt += context.right_context

    for i in range(len(ctxt)):
        ctxt[i] = lemmatizer.lemmatize(ctxt[i].lower(), pos=context.pos)
    ctxt = list(dict.fromkeys(ctxt)) #remove duplicates
    for word in list(ctxt):
        if word in stop_words:
            ctxt.remove(word)

    for l1 in wn.lemmas(context.lemma, pos=context.pos):
        for s1 in l1.synset().lemmas():
            definition = []
            synonym = s1.name().replace('_', ' ').lower()
            if synonym == context.lemma.replace('_', ' ').lower():
                continue
            definition += tokenize(s1.synset().definition().lower())
            for ex in s1.synset().examples():
                definition += tokenize(ex.lower())
            for hyper in s1.synset().hypernyms():
                definition += tokenize(hyper.definition().lower())
                for hy_ex in hyper.examples():
                    definition += tokenize(hy_ex.lower())
            for i in range(len(definition)):
                definition[i] = lemmatizer.lemmatize(definition[i], pos=context.pos)
            definition = list(dict.fromkeys(definition))
            for word in list(definition):
                if word in stop_words:
                    definition.remove(word)
            for word in ctxt:
                if word in definition:
                    if synonym in candidates:
                        candidates[synonym] += 1
                    else:
                        candidates[synonym] = 1

    if bool(candidates):
        frequency = {}
        max_val = max(candidates.items(), key=(lambda x: x[1]))[1]
        max_keys = []

        for key, value in candidates.items():
            if value == max_val:
                max_keys.append(key)
                if key not in frequency.keys() and l1.synset().name().lower() != context.lemma.lower():
                    frequency[key] = 0
                for l1 in wn.lemmas(key.replace(' ', '_'), pos=context.pos):
                    if l1.count() > frequency[key] and l1.synset().name().lower() != context.lemma.lower():
                        frequency[key] = l1.count()

        if len(max_keys) == 1: #return if there is only 1 max value
            return max_keys[0]

        else: #if there are multiple max values, return the most frequent count
            max_tuple = ('smurf', 0)
            for item in max_keys:
                if frequency[item] > max_tuple[1] and item.replace(' ', '_').lower() != context.lemma.lower():
                    max_tuple = (item, frequency[item])
            return max_tuple[0]
    else:
        result = 'smurf'
        max_val = 0
        for l1 in wn.lemmas(context.lemma, pos=context.pos):
            for s1 in l1.synset().lemmas():
                if s1.count() > max_val and s1.name().lower() != context.lemma.lower():
                    max_val = s1.count()
                    result = s1.name().lower()
        return result

class Word2VecSubst(object):

    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self,context : Context) -> str:
        # Part 4

        candidates = {}

        list = get_candidates(context.lemma, pos=context.pos)
        word_vectors = self.model.vocab

        for item in list:
            if item.replace(' ', '_') in word_vectors:
                candidates[item] = self.model.similarity(context.lemma, item.replace(' ', '_'))

        if bool(candidates):
            return max(candidates.keys(), key=(lambda k: candidates[k]))
        else:
            return 'smurf'


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        # Part 5

        candidates = get_candidates(context.lemma, pos=context.pos)
        input_str = ' '.join(context.left_context) + ' [MASK] ' + ' '.join(context.right_context)
        input_toks = self.tokenizer.encode(input_str)
        index = self.tokenizer.convert_ids_to_tokens(input_toks).index('[MASK]')
        input_mat = np.array(input_toks).reshape((1, -1))
        outputs = self.model.predict(input_mat)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0][index])[::-1]
        result = 'smurf'
        best_words_list = self.tokenizer.convert_ids_to_tokens(best_words)
        for word in best_words_list:
            if word in candidates:
                return word

        return result

def part6_predictor(BERT_predictor, context : Context) -> str:
    # Part 6

    ## I create two inputs: a masked input and an unmasked input
    ## I merge two outputs from two inputs  and sort the merged output,
    ## then return the synonym that is not the same as the target word and has the highest score.

    candidates = get_candidates(context.lemma, pos=context.pos)
    input_str1 = ' '.join(context.left_context) + ' [MASK] ' + ' '.join(context.right_context)
    input_str2 = ' '.join(context.left_context) + ' ' + context.lemma + ' ' + ' '.join(context.right_context)
    input_toks1 = BERT_predictor.tokenizer.encode(input_str1)
    input_toks2 = BERT_predictor.tokenizer.encode(input_str2)
    index1 = BERT_predictor.tokenizer.convert_ids_to_tokens(input_toks1).index('[MASK]')
    index2 = BERT_predictor.tokenizer.convert_ids_to_tokens(input_toks2).index(context.lemma)
    input_mat1 = np.array(input_toks1).reshape((1, -1))
    input_mat2 = np.array(input_toks2).reshape((1, -1))
    outputs1 = BERT_predictor.model.predict(input_mat1)
    outputs2 = BERT_predictor.model.predict(input_mat2)
    predictions1 = outputs1[0]
    predictions2 = outputs2[0]
    merged_list = predictions1[0][index1]
    merged_list += predictions2[0][index2]
    best_words = np.argsort(merged_list)[::-1]
    best_words_list = BERT_predictor.tokenizer.convert_ids_to_tokens(best_words)

    lemmatizer = WordNetLemmatizer()

    for i in range(len(best_words_list)):
        best_words_list[i] = lemmatizer.lemmatize(best_words_list[i].lower())
    best_words_list = list(dict.fromkeys(best_words_list))
    for synonym in best_words_list:
        if synonym != context.lemma and synonym in candidates:
            return synonym

    return 'smurf'


if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = './GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)
    predictor = BertPredictor()

    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        #prediction = smurf_predictor(context)
        #prediction = wn_frequency_predictor(context)
        #prediction = wn_simple_lesk_predictor(context)
        #prediction = predictor.predict_nearest(context)
        #prediction = predictor.predict(context)
        prediction = part6_predictor(predictor, context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
    #model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True)
    #print(get_candidates('slow', 'a'))
    #stop_words = stopwords.words('english')
    #print(stop_words)