import pandas as pd
import csv
import gensim.downloader as api
import gensim.models.keyedvectors as word2vec  # for word2vec embeddings

from nltk import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity


def removefirsttoken(x):
    x = x.split(' ', 1)[1]
    return x

file_read_path1 = './data/News_Train.tsv'
file_read_path2 = './data/WikiNews_Train.tsv'
file_read_path3 = './data/Wikipedia_Train.tsv'
file_write_path = './data/full_train_final.tsv'

data_frame1 = pd.read_table(file_read_path1, names=('ID', 'sentence', 'start_index', 'end_index', 'phrase', 'total_native', 'total_non_native','native_complex','non_native_complex','complex_binary','complex_probabilistic'))
data_frame2 = pd.read_table(file_read_path2, names=('ID', 'sentence', 'start_index', 'end_index', 'phrase', 'total_native', 'total_non_native','native_complex','non_native_complex','complex_binary','complex_probabilistic'))
data_frame3 = pd.read_table(file_read_path3, names=('ID', 'sentence', 'start_index', 'end_index', 'phrase', 'total_native', 'total_non_native','native_complex','non_native_complex','complex_binary','complex_probabilistic'))

data_frame2['sentence'] = data_frame2['sentence'].apply(lambda x: removefirsttoken(x))

data_frame = pd.concat([data_frame1, data_frame2, data_frame3], ignore_index=True, sort=False)
# for testing small file
data_frame = pd.read_table('./data/test_file_small.tsv', names=('ID', 'sentence', 'start_index', 'end_index', 'phrase', 'total_native', 'total_non_native','native_complex','non_native_complex','complex_binary','complex_probabilistic'))


data_frame['split'] = data_frame['phrase'].apply(lambda x: x.split())
data_frame['count'] = data_frame['split'].apply(lambda x: len(x))
sentences = data_frame[data_frame['count']==1]

# tfidf_vectorizer = TfidfVectorizer(analyzer="char")
# model_word2vec = word2vec.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
model_fasttext = api.load("fasttext-wiki-news-subwords-300")

# mask = sentences["complex_binary"]==1
# complex_sentences = sentences.loc[mask]

sentence = ''

with open(file_write_path, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for index, row in sentences.iterrows():
        label = 'N'
        if sentence != row['sentence']:
            if sentence != '':
                print(sentence)
                print(target_words)
                sen_length = len(split_sentence)
                sen_complex = len(complex_words)
                for word in split_sentence:
                    min_sim = 0.0
                    max_sim = 0.0
                    mean_sim = 0.0
                    total_sim = 0.0
                    if word in target_words:
                        print(word)
                        min_sim = 1.0
                        for target in target_words:
                            if word != target:
                                print(target)
                                # tfidf_matrix = tfidf_vectorizer.fit_transform((word, target))
                                # cs = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
                                # cos_sim = cs[0][1]
                                try:
                                    cos_sim = model_fasttext.similarity(word, target)
                                    print(cos_sim)
                                    max_sim = cos_sim if cos_sim >= max_sim else max_sim
                                    min_sim = cos_sim if cos_sim <= min_sim else min_sim
                                    total_sim += cos_sim
                                except:
                                    continue
                        mean_sim = total_sim/(len(target_words)-1)
                    label = 'C' if word in complex_words else 'N'
                    tsv_writer.writerow([word, sen_length, sen_complex, "{:.7f}".format(min_sim), "{:.7f}".format(max_sim), "{:.7f}".format(mean_sim), label])
                tsv_writer.writerow('')

            complex_words = []
            target_words = []
            sentence = row['sentence']
            split_sentence = word_tokenize(sentence)

        if row["complex_binary"]==1:
            complex_words.append(row['phrase'])
        target_words.append(row['phrase'])

    # repeated code for the last sentence
    sen_length = len(split_sentence)
    sen_complex = len(complex_words)
    for word in split_sentence:
        min_sim = 0.0
        max_sim = 0.0
        mean_sim = 0.0
        total_sim = 0.0
        if word in target_words:
            min_sim = 1.0
            for target in target_words:
                if word != target:
                    try:
                        cos_sim = model_fasttext.similarity(word, target)
                        max_sim = cos_sim if cos_sim >= max_sim else max_sim
                        min_sim = cos_sim if cos_sim <= min_sim else min_sim
                        total_sim += cos_sim
                    except:
                        continue
            mean_sim = total_sim/(len(target_words)-1)
        label = 'C' if word in complex_words else 'N'
        tsv_writer.writerow([word, sen_length, sen_complex, "{:.7f}".format(min_sim), "{:.7f}".format(max_sim), "{:.7f}".format(mean_sim), label])

