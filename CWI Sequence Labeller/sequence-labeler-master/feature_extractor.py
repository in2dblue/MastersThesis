import pandas as pd

from nltk.corpus import wordnet
from datamuse import datamuse


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
        try:
            word_results = self.api.words(sp=word, max=1, md='psf')
            if len(word_results) > 0:
                word = word_results[0]["word"]
                syllables = int(word_results[0]["numSyllables"])
            return syllables
        except:
            print('Error to get syllables for: ' + word)
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
