"""
COMS W4705 - Natural Language Processing - Summer 19 
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps:
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict):
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table:
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str):
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar):
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2
        table = {}
        for i in range(len(tokens)):
            table[(i, i+1)] = []
            for rule in self.grammar.rhs_to_rules[(tokens[i],)]:
                table[(i, i+1)].append(rule[0])

        for length in range(2, len(tokens)+1):
            for i in range(len(tokens)-length+1):
                j = i+length
                table[(i,j)] = []
                for k in range(i+1, j):
                    for rule1 in table[(i, k)]:
                        for rule2 in table[(k, j)]:
                            for rule in self.grammar.rhs_to_rules[(rule1, rule2)]:
                                table[(i, j)]. append(rule[0])

        for result in table[(0, len(tokens))]:
            if self.grammar.startsymbol == result:
                return True

        return False

    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        # TODO, part 3
        table = defaultdict(lambda: defaultdict(dict))
        probs = defaultdict(lambda: defaultdict(float))

        for i in range(len(tokens)):
            for rule in self.grammar.rhs_to_rules[(tokens[i],)]:
                probs[(i, i+1)][rule[0]] = math.log(rule[-1])
                table[(i, i+1)][rule[0]] = tokens[i]

        for length in range(2, len(tokens)+1):
            for i in range(len(tokens)-length+1):
                j = i+length
                for k in range(i+1, j):
                    for rule1 in table[(i, k)]:
                        for rule2 in table[(k, j)]:
                            for rule in self.grammar.rhs_to_rules[(rule1, rule2)]:
                                prob = probs[(i, k)][rule1] + probs[(k, j)][rule2] + math.log(rule[-1])
                                if rule[0] in probs[(i, j)] and prob > probs[(i, j)][rule[0]]:
                                        probs[(i, j)][rule[0]] = prob
                                        table[(i, j)][rule[0]] = ((rule1, i, k), (rule2, k, j))
                                elif rule[0] not in probs[(i, j)]:
                                    probs[(i, j)][rule[0]] = prob
                                    table[(i, j)][rule[0]] = ((rule1, i, k), (rule2, k, j))

        if self.grammar.startsymbol not in table[(0, len(tokens))]:
            return None, None

        return table, probs


def get_tree(chart, i,j,nt):
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4
    if type(chart[(i, j)][nt]) is tuple:
        return (nt, get_tree(chart, chart[(i,j)][nt][0][1], chart[(i, j)][nt][0][2], chart[(i, j)][nt][0][0]),get_tree(chart, chart[(i,j)][nt][1][1], chart[(i, j)][nt][1][2], chart[(i, j)][nt][1][0]))
    else:
        return (nt, chart[(i, j)][nt])



if __name__ == "__main__":

    with open('atis3.pcfg','r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.']
        print(parser.is_in_language(toks))
        toks2 = ['miami','flights','cleveland','from','to','.']
        print(parser.is_in_language(toks2))
        toks3 = ['thank','you','.']
        print(parser.is_in_language(toks3))
        toks4 = ['you','thank','.']
        print(parser.is_in_language(toks4))
        table, probs = parser.parse_with_backpointers(toks)
        assert check_table_format(table)
        #print "Table formatted correctly: {}".format(check_table_format(table))
        assert check_probs_format(probs)
        #print "Probabilities formatted correctly: {}".format(check_probs_format(probs))
        print(get_tree(table, 0, len(toks), grammar.startsymbol))

