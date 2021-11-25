import pandas as pd
import gensim.downloader as api
import csv
import string
# import pycorenlp
# import gensim.models.keyedvectors as word2vec  # for word2vec embeddings

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from datamuse import datamuse
from pycorenlp import StanfordCoreNLP
from collections import defaultdict
from tqdm import tqdm
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


def del_first_token(x):
    x = x.split(' ', 1)[1]
    return x


# Convert tree bank tags to ones that are compatible w google
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


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


# function to parse sentences
def parse(string):
    output = nlp.annotate(string, properties={
        'annotators': 'pos,depparse',
        'outputFormat': 'json'
        })
    return output


# Function to get the proper lemma
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatiser(row):
    word = row['phrase']
    pos = row['pos']
    try:
        lemma = wordnet_lemmatizer.lemmatize(word, get_wordnet_pos(pos))
        return lemma
    except:
        try:
            lemma = wordnet_lemmatizer.lemmatize(word)
            return lemma
        except:
            print(word)


def get_pos(row):
    word = row['phrase']
    word = word.lower()
    parse = row['parse']
    for i in range(len(parse['sentences'][0]['tokens'])):
        comp_word = parse['sentences'][0]['tokens'][i]['word']
        comp_word = comp_word.lower()
        comp_word = comp_word.translate({ord(char): None for char in remove})
        if comp_word == word:
            return parse['sentences'][0]['tokens'][i]['pos']


def get_google_freq(row):
    nofreq = float(0.000000)
    word = row["phrase"]
    word = str(word)
    tag = row["pos"]
    tag = penn_to_google(tag)

    try:
        word_results = muse_api.words(sp=word, max=1, md='pf')
        tag_list = (word_results[0]['tags'][:-1])
        frequency = word_results[0]['tags'][-1][2:]
        frequency = float(frequency)
        if tag in tag_list:
            return frequency
        else:
            lemma = row['lemma']
            try:
                word_results = muse_api.words(sp=lemma, max=1, md='pf')
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


file_read_path1 = './data/News_Train.tsv'
file_read_path2 = './data/WikiNews_Train.tsv'
file_read_path3 = './data/Wikipedia_Train.tsv'
file_write_path = './data/full_train_final.tsv'

remove = string.punctuation
remove = remove.replace("-", "")
remove = remove.replace("'", "")  # don't remove apostraphies
remove = remove + '“'
remove = remove + '”'
remove2 = '"'
# pattern = r"[{}]".format(remove)  # create the pattern

print('Loading files...')
data_frame1 = pd.read_table(file_read_path1, names=('ID', 'sentence', 'start_index', 'end_index', 'phrase', 'total_native', 'total_non_native','native_complex','non_native_complex','complex_binary','complex_probabilistic'))
data_frame2 = pd.read_table(file_read_path2, names=('ID', 'sentence', 'start_index', 'end_index', 'phrase', 'total_native', 'total_non_native','native_complex','non_native_complex','complex_binary','complex_probabilistic'))
data_frame3 = pd.read_table(file_read_path3, names=('ID', 'sentence', 'start_index', 'end_index', 'phrase', 'total_native', 'total_non_native','native_complex','non_native_complex','complex_binary','complex_probabilistic'))

data_frame2['sentence'] = data_frame2['sentence'].apply(lambda x: del_first_token(x))

data_frame = pd.concat([data_frame1, data_frame2, data_frame3], ignore_index=True, sort=False)
# # for testing small file
# data_frame = pd.read_table('./data/test_file_small.tsv', names=('ID', 'sentence', 'start_index', 'end_index', 'phrase', 'total_native', 'total_non_native','native_complex','non_native_complex','complex_binary','complex_probabilistic'))


data_frame['split'] = data_frame['phrase'].apply(lambda x: x.split())
data_frame['count'] = data_frame['split'].apply(lambda x: len(x))
df = data_frame[data_frame['count']==1]
sentences = df[['sentence', 'ID', 'phrase', 'complex_binary']].copy()
sentences['sentence'] = sentences['sentence'].apply(lambda x: x.translate({ord(char): None for char in remove2}))

# mask = sentences["complex_binary"]==1
# complex_sentences = sentences.loc[mask]

wordnet_lemmatizer = WordNetLemmatizer()
muse_api = datamuse.Datamuse()
nlp = StanfordCoreNLP('http://localhost:9000')
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

# apply parsing to sentences
print('Parsing...')
sentences['clean sen'] = sentences['sentence'].apply(lambda x: x.translate({ord(char): None for char in remove}))
sentences['parse'] = sentences['clean sen'].apply(lambda x: parse(x))

print('Computing POS...')
sentences['pos'] = sentences.apply(get_pos, axis=1)

print('Computing word lemmas...')
sentences['lemma'] = sentences.apply(lemmatiser, axis=1)

print('Computing google frequency...')
sentences['google frequency'] = sentences.apply(get_google_freq, axis=1)

print('Loading pretrained fasttext embeddings...')
model_fasttext = api.load("fasttext-wiki-news-subwords-300")
print('Embedding load finished...')

# tfidf_vectorizer = TfidfVectorizer(analyzer="char")
# model_word2vec = word2vec.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

sentence = ''
google_features = defaultdict(list)

print('Computing cosine similarity of target word with other targets word in sentence...')
with open(file_write_path, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for index, row in tqdm(sentences.iterrows()):
        label = 'N'
        if sentence != row['sentence']:
            if sentence != '':
                print(google_features)
                sen_length = len(split_sentence)
                # target_length = len(target_words)
                sen_complex = "{:.3f}".format(len(complex_words)/sen_length)
                for word in split_sentence:
                    min_sim = 0.0
                    max_sim = 0.0
                    mean_sim = 0.0
                    total_sim = 0.0
                    pos = None
                    google_freq = None
                    if word in target_words:
                        pos, google_freq = google_features[word]
                        # pos = 'None' if pos is None else pos
                        min_sim = 1.0
                        for target in split_sentence_wo_sw:
                            if word != target:
                                # tfidf_matrix = tfidf_vectorizer.fit_transform((word, target))
                                # cs = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
                                # cos_sim = cs[0][1]
                                try:
                                    cos_sim = model_fasttext.similarity(word, target)
                                    cos_sim = cos_sim
                                    max_sim = cos_sim if cos_sim >= max_sim else max_sim
                                    min_sim = cos_sim if cos_sim <= min_sim else min_sim
                                    total_sim += cos_sim
                                except:
                                    continue
                        # mean_sim = total_sim if target_length==1 else total_sim/(split_sentence_wo_sw-1)
                        mean_sim = total_sim/(len(split_sentence_wo_sw)-1)
                    label = 'C' if word in complex_words else 'N'
                    tsv_writer.writerow([word, sen_length, sen_complex, "{:.7f}".format(min_sim), "{:.7f}".format(max_sim), "{:.7f}".format(mean_sim), pos, google_freq, label])
                tsv_writer.writerow('')

            complex_words = []
            target_words = []
            google_features.clear()
            sentence = row['sentence']
            split_sentence = word_tokenize(sentence)
            split_sentence_wo_sw = [word for word in split_sentence if not word in stopwords.words()]
            split_sentence_wo_sw = [word for word in split_sentence_wo_sw if not word in remove]
            split_sentence_wo_sw = [word for word in split_sentence_wo_sw if not word == '’']

        if row["complex_binary"]==1:
            complex_words.append(row['phrase'])
        target_words.append(row['phrase'])
        if row['phrase'] not in google_features:
            google_features[row['phrase']].append(row['pos'])
            google_features[row['phrase']].append(row['google frequency'])

    # repeated code for the last sentence
    sen_length = len(split_sentence)
    target_length = len(target_words)
    sen_complex = "{:.3f}".format(len(complex_words)/sen_length)
    for word in split_sentence:
        min_sim = 0.0
        max_sim = 0.0
        mean_sim = 0.0
        total_sim = 0.0
        pos = None
        google_freq = None
        if word in target_words:
            pos, google_freq = google_features[word]
            # pos = 'None' if pos is None else pos
            min_sim = 1.0
            for target in split_sentence_wo_sw:
                if word != target:
                    try:
                        cos_sim = model_fasttext.similarity(word, target)
                        max_sim = cos_sim if cos_sim >= max_sim else max_sim
                        min_sim = cos_sim if cos_sim <= min_sim else min_sim
                        total_sim += cos_sim
                    except:
                        continue
            # mean_sim = total_sim if target_length==1 else total_sim/(target_length-1)
            mean_sim = total_sim/(len(split_sentence_wo_sw)-1)
        label = 'C' if word in complex_words else 'N'
        tsv_writer.writerow([word, sen_length, sen_complex, "{:.7f}".format(min_sim), "{:.7f}".format(max_sim), "{:.7f}".format(mean_sim), pos, google_freq, label])

