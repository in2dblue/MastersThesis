import pandas as pd
import csv
import nltk

from nltk import word_tokenize

nltk.download('punkt')

file_read_path = './data/lcp_single_train_corrected_new.tsv'
file_write_path = './data/lcp_single_train_corrected_new_token.tsv'

# data_frame = pd.read_table(file_read_path, names=('ID', 'sentence', 'start_index', 'end_index', 'phrase', 'total_native', 'total_non_native','native_complex','non_native_complex','complex_binary','complex_probabilistic'))
data_frame = pd.read_table(file_read_path, names=('ID', 'corpus', 'sentence', 'token', 'complexity'))

# data_frame['split'] = data_frame['phrase'].apply(lambda x: x.split())
# data_frame['count'] = data_frame['split'].apply(lambda x: len(x))
#
# sentences = data_frame[data_frame['count']==1]
# mask = sentences["complex_binary"]==1
# complex_sentences = sentences.loc[mask]
complex_sentences = data_frame
print(complex_sentences)
# exit()
sentence = ''
complexity = 0

with open(file_write_path, 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    for index, row in complex_sentences.iterrows():
        label = 'N'
        comp = None
        if sentence != row['sentence']:
            if sentence != '':
                for word in split_sentence:
                    if word != '.' and word[-1] == '.':
                        word = word[:-1]
                        label = 'C' if word in complex_words else 'N'
                        comp = complexity if word in complex_words else None
                        tsv_writer.writerow([word, comp, label])
                        tsv_writer.writerow(['.', '', 'N'])
                    elif word[-1] == ',':
                        word = word[:-1]
                        label = 'C' if word in complex_words else 'N'
                        comp = complexity if word in complex_words else None
                        if word.strip() != '':
                            tsv_writer.writerow([word, comp, label])
                        tsv_writer.writerow([',', '', 'N'])
                    elif word[-1] == '"':
                        word = word[:-1]
                        if word[-1] == '.':
                            word = word[:-1]
                            label = 'C' if word in complex_words else 'N'
                            comp = complexity if word in complex_words else None
                            tsv_writer.writerow([word, comp, label])
                            tsv_writer.writerow(['.', '', 'N'])
                            tsv_writer.writerow(['"', '', 'N'])
                        else:
                            label = 'C' if word in complex_words else 'N'
                            comp = complexity if word in complex_words else None
                            tsv_writer.writerow([word, comp, label])
                            tsv_writer.writerow(['"', '', 'N'])
                    elif word[0] == '"':
                        word = word[1:]
                        label = 'C' if word in complex_words else 'N'
                        comp = complexity if word in complex_words else None
                        tsv_writer.writerow(['"', '', 'N'])
                        tsv_writer.writerow([word, comp, label])
                    else:
                        label = 'C' if word in complex_words else 'N'
                        comp = complexity if word in complex_words else None
                        tsv_writer.writerow([word, comp, label])
                tsv_writer.writerow('')

            complex_words = []
            # id = row['ID']
            sentence = row['sentence']
            complexity = row['complexity']
            split_sentence = word_tokenize(sentence)
            # split_sentence = row['sentence'].split()

        complex_words.append(row['token'])
        print(complex_words)

    for word in split_sentence:
        if word != '.' and word[-1] == '.':
            word = word[:-1]
            label = 'C' if word in complex_words else 'N'
            comp = complexity if word in complex_words else None
            tsv_writer.writerow([word, comp, label])
            tsv_writer.writerow(['.', '', 'N'])
        elif word[-1] == ',':
            word = word[:-1]
            label = 'C' if word in complex_words else 'N'
            comp = complexity if word in complex_words else None
            if word.strip() != '':
                tsv_writer.writerow([word, comp, label])
            tsv_writer.writerow([',', '', 'N'])
        elif word[-1] == '"':
            word = word[:-1]
            if word[-1] == '.':
                word = word[:-1]
                label = 'C' if word in complex_words else 'N'
                comp = complexity if word in complex_words else None
                tsv_writer.writerow([word, comp, label])
                tsv_writer.writerow(['.', '', 'N'])
                tsv_writer.writerow(['"', '', 'N'])
            else:
                label = 'C' if word in complex_words else 'N'
                comp = complexity if word in complex_words else None
                tsv_writer.writerow([word, comp, label])
                tsv_writer.writerow(['"', '', 'N'])
        elif word[0] == '"':
            word = word[1:]
            label = 'C' if word in complex_words else 'N'
            comp = complexity if word in complex_words else None
            tsv_writer.writerow(['"', '', 'N'])
            tsv_writer.writerow([word, comp, label])
        else:
            label = 'C' if word in complex_words else 'N'
            comp = complexity if word in complex_words else None
            tsv_writer.writerow([word, comp, label])

