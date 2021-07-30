import pandas as pd
import string
import pycorenlp
import csv

from nltk import word_tokenize
from datamuse import datamuse
from pycorenlp import StanfordCoreNLP
from nltk.corpus import wordnet


# function to obtain syllables for words
def get_syllables(word):
    syllables = 0
    word_results = api.words(sp=word, max=1, md='psf')
    if len(word_results)>0:
        word = word_results[0]["word"]
        syllables = int(word_results[0]["numSyllables"])
    return syllables


def removefirsttoken(x):
    x = x.split(' ', 1)[1]
    return x


#function to parse sentences
def parse(string):
    output = nlp.annotate(string, properties={'annotators': 'pos,depparse',
                                              'outputFormat': 'json'
                                              })
    return output

file_read_path = './News_Train.tsv'
file_write_path = './News_Train3.tsv'
Wikinews=False

data_frame = pd.read_table(file_read_path, names=('ID', 'sentence', 'start_index', 'end_index', 'phrase', 'total_native', 'total_non_native','native_complex','non_native_complex','complex_binary','complex_probabilistic'))

data_frame['split'] = data_frame['phrase'].apply(lambda x: x.split())
data_frame['count'] = data_frame['split'].apply(lambda x: len(x))

# We create a table that contains only the words
words = data_frame[data_frame['count']==1]
word_set = words.phrase.str.lower().unique()
word_set = pd.DataFrame(word_set)
word_set.columns = ['phrase']

# Cleaning function for words
remove = string.punctuation
remove = remove.replace("-", "")
remove = remove.replace("'", "")    # don't remove apostraphies
remove = remove + '“'
remove = remove +'”'
pattern = r"[{}]".format(remove)    # create the pattern
word_set['phrase'] = word_set['phrase'].apply(lambda x :x.translate({ord(char): None for char in remove}))

# Apply function to get syllables
api = datamuse.Datamuse()
print(word_set['phrase'])
word_set['syllables'] = word_set['phrase'].apply(lambda x: get_syllables(x))
print(word_set['syllables'])

# # Apply function to get word length
# word_set['length'] = word_set['phrase'].apply(lambda x: len(x))
#
# # take words and merge with values first you will need to clean the phrase column
# words['original phrase'] = words['phrase']
# words['phrase'] = words['phrase'].str.lower()
# words['phrase'] = words['phrase'].apply(lambda x: x.translate({ord(char): None for char in remove}))
#
# # words.to_csv('words.tsv', sep='\t', index=False, header=False, quotechar=' ')
# # word_set.to_csv('word_set.tsv', sep='\t', index=False, header=False, quotechar=' ')
# word_features = pd.merge(words, word_set)
# # word_features.to_csv(file_write_path, sep='\t', index=False, header=False, quotechar=' ')
#
# sentences = data_frame[['sentence', 'ID']].copy()
# sentences = sentences.drop_duplicates()
# if Wikinews:
#     sentences['clean sentence'] = sentences['sentence'].apply(lambda x: removefirsttoken(x))
# else:
#     sentences['clean sentence'] = sentences['sentence']
#
# # Now parse
# nlp = StanfordCoreNLP('http://localhost:9000')
# # apply parsing to sentences
# sentences['parse'] = sentences['clean sentence'].apply(lambda x: parse(x))
# # Merge to word features
# word_parse_features = pd.merge(sentences, word_features)
# # word_parse_features.to_csv(file_write_path, sep='\t', index=False, header=False, quotechar=' ')




# with open(file_read_path, "r") as f:
#     reader = csv.reader(f, delimiter="\t", quotechar='"')
#     split_list = []
#     sentence = []
#     sen = ''
#     i = 0
#     dataframe = pd.DataFrame()
#     for row in reader:
#         i = i+1
#         if sen != row[0].strip():
#             sen = row[0].strip()
#             split_list.append(word_tokenize(sen))
#             split_list.append(None)
#             # if i > 3:
#             #     print(split_list)
#             #     exit()
#
#
#             dataframe['word'] = split_list
#             dataframe['binary'] = 'N'
#             dataframe.to_csv(file_write_path, sep='\t', index=False, header=False, quotechar=' ')
#         # print(sen)
#
#         # line = line.strip()
#         # line = line.split('\t')
#         # split_list = word_tokenize(line)
#         # print(split_list)
#         # exit()
#
#         # dataframe = pd.DataFrame()
#         # dataframe['word'] = split_list
#         # dataframe['binary'] = 'N'
#         # dataframe.to_csv(file_write_path, sep='\t', index=False, header=False, quotechar=' ')
#
#
#
#
# def convert_format_string(self, string):
#
#     split_list = word_tokenize(string)
#
#     dataframe = pd.DataFrame()
#     dataframe['word'] = split_list
#     dataframe['binary'] = 'N'
#     dataframe.to_csv(self.temp_file, sep = '\t' ,index=False, header=False, quotechar=' ')


# in_data = pd.read_csv (file_read_path, sep = '\t')
#
# for line in in_data:
#     print(line)