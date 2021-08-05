import pandas as pd
import csv

from nltk import word_tokenize

file_read_path = './data/Wikipedia_Dev.tsv'
file_write_path = './data/Wikipedia_Dev_final2.tsv'

data_frame = pd.read_table(file_read_path, names=('ID', 'sentence', 'start_index', 'end_index', 'phrase', 'total_native', 'total_non_native','native_complex','non_native_complex','complex_binary','complex_probabilistic'))

data_frame['split'] = data_frame['phrase'].apply(lambda x: x.split())
data_frame['count'] = data_frame['split'].apply(lambda x: len(x))

sentences = data_frame[data_frame['count']==1]
mask = sentences["complex_binary"]==1
complex_sentences = sentences.loc[mask]
# print(complex_sentences)
sentence = ''

with open(file_write_path, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for index, row in complex_sentences.iterrows():
        label = 'N'
        if sentence != row['sentence']:
            if sentence != '':
                for word in split_sentence:
                    if word != '.' and word[-1] == '.':
                        word = word[:-1]
                        label = 'C' if word in complex_words else 'N'
                        tsv_writer.writerow([word, label])
                        tsv_writer.writerow(['.', 'N'])
                    elif word[-1] == ',':
                        word = word[:-1]
                        label = 'C' if word in complex_words else 'N'
                        if word.strip() != '':
                            tsv_writer.writerow([word, label])
                        tsv_writer.writerow([',', 'N'])
                    elif word[-1] == '"':
                        word = word[:-1]
                        if word[-1] == '.':
                            word = word[:-1]
                            label = 'C' if word in complex_words else 'N'
                            tsv_writer.writerow([word, label])
                            tsv_writer.writerow(['.', 'N'])
                            tsv_writer.writerow(['"', 'N'])
                        else:
                            label = 'C' if word in complex_words else 'N'
                            tsv_writer.writerow([word, label])
                            tsv_writer.writerow(['"', 'N'])
                    elif word[0] == '"':
                        word = word[1:]
                        label = 'C' if word in complex_words else 'N'
                        tsv_writer.writerow(['"', 'N'])
                        tsv_writer.writerow([word, label])
                    else:
                        label = 'C' if word in complex_words else 'N'
                        tsv_writer.writerow([word, label])
                tsv_writer.writerow('')

            complex_words = []
            # id = row['ID']
            sentence = row['sentence']
            # split_sentence = word_tokenize(sentence)
            split_sentence = row['sentence'].split()

        complex_words.append(row['phrase'])
        print(complex_words)

    for word in split_sentence:
        if word != '.' and word[-1] == '.':
            word = word[:-1]
            label = 'C' if word in complex_words else 'N'
            tsv_writer.writerow([word, label])
            tsv_writer.writerow(['.', 'N'])
        elif word[-1] == ',':
            word = word[:-1]
            label = 'C' if word in complex_words else 'N'
            if word.strip() != '':
                tsv_writer.writerow([word, label])
            tsv_writer.writerow([',', 'N'])
        elif word[-1] == '"':
            word = word[:-1]
            if word[-1] == '.':
                word = word[:-1]
                label = 'C' if word in complex_words else 'N'
                tsv_writer.writerow([word, label])
                tsv_writer.writerow(['.', 'N'])
                tsv_writer.writerow(['"', 'N'])
            else:
                label = 'C' if word in complex_words else 'N'
                tsv_writer.writerow([word, label])
                tsv_writer.writerow(['"', 'N'])
        elif word[0] == '"':
            word = word[1:]
            label = 'C' if word in complex_words else 'N'
            tsv_writer.writerow(['"', 'N'])
            tsv_writer.writerow([word, label])
        else:
            label = 'C' if word in complex_words else 'N'
            tsv_writer.writerow([word, label])

