import pandas as pd

from nltk.corpus import wordnet
from datamuse import datamuse


# Convert tree bank tags to ones that are compatible w google
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


# def penn_to_wn(tag):
#     if is_adjective(tag):
#         return wn.ADJ
#     elif is_noun(tag):
#         return wn.NOUN
#     elif is_adverb(tag):
#         return wn.ADV
#     elif is_verb(tag):
#         return wn.VERB
#     return None

def penn_to_google(tag):
    if is_adjective(tag):
        return 'adj'
    elif is_noun(tag):
        return 'n'
    elif is_adverb(tag):
        return 'adv'
    elif is_verb(tag):
        return 'v'
    return None


class ExtractFeatures(object):
    def __init__(self, sentences=None):
        self.sentences = sentences
        self.api = datamuse.Datamuse()
        self.ogden = pd.read_table('modeldata/Ogden1000.txt')
        self.subimdb = pd.read_table('modeldata/subimdb1000.txt')
        self.simplewiki = pd.read_table('modeldata/simple_wiki_freq.txt')
        self.lang8 = pd.read_table('modeldata/lang8_freq.tsv')

    # Lexical Features
    @staticmethod
    def word_length(word):
        return len(word)

    @staticmethod
    def synonyms(word):
        synonyms = 0
        try:
            results = wordnet.synsets(word)
            synonyms = len(results)
            return synonyms
        except:
            return synonyms

    @staticmethod
    def hypernyms(word):
        hypernyms = 0
        try:
            results = wordnet.synsets(word)
            hypernyms = len(results[0].hypernyms())
            return hypernyms
        except:
            return hypernyms

    @staticmethod
    def hyponyms(word):
        hyponyms = 0
        try:
            results = wordnet.synsets(word)
            hyponyms = len(results[0].hyponyms())
            return hyponyms
        except:
            return hyponyms

    # @staticmethod
    def get_syllables(self, word):
        syllables = 0
        word_results = self.api.words(sp=word, max=1, md='psf')
        if len(word_results) > 0:
            word = word_results[0]["word"]
            syllables = int(word_results[0]["numSyllables"])
        return syllables

    # BINARY FEATURES
    def ogdens_basic_english(self, word):
        return 1 if any(self.ogden.words == word.lower()) else 0

    def subimdb_frequent(self, word):
        return 1 if any(self.subimdb.words == word.lower()) else 0

    def simplewiki_frequent(self, word):
        return 1 if any(self.simplewiki.words == word.lower()) else 0

    # Word Frequency

    def lang8_word_freq(self, word):
        try:
            df_new = self.lang8.query("words==@word.lower()")
            value = "{:.7f}".format(df_new["normalized"].iloc[0])
            return value
        except:
            return 0.0

    def get_google_freq(self, row):
        nofreq = float(0.000000)
        word = row["phrase"]
        word = str(word)
        tag = row["pos"]
        tag = penn_to_google(tag)

        try:
            word_results = self.api.words(sp=word, max=1, md='pf')
            tag_list = (word_results[0]['tags'][:-1])
            frequency = word_results[0]['tags'][-1][2:]
            frequency = float(frequency)
            if tag in tag_list:
                return frequency
            else:
                lemma = row['lemma']
                try:
                    word_results = self.api.words(sp=lemma, max=1, md='pf')
                    tag_list = (word_results[0]['tags'][:-1])
                    frequency = word_results[0]['tags'][-1][2:]
                    frequency = float(frequency)
                    if tag in tag_list:
                        return frequency
                    else:
                        return nofreq
                except:
                    return nofreq
        except:
            return nofreq

