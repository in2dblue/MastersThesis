from nltk.corpus import wordnet
from datamuse import datamuse


class ExtractFeatures(object):
    def __init__(self, sentences=None):
        self.sentences = sentences
        self.api = datamuse.Datamuse()

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
        except:
            return hyponyms
        try:
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
