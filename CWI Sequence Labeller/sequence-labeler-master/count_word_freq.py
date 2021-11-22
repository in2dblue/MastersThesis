import re
import csv

from collections import Counter

# code for most common words in simple wikipedia
words = re.findall(r'\w+', open('modeldata/simple_wiki.txt').read().lower())
most_common=Counter(words).most_common(6370)
# print(most_common)
# print(len(most_common))

with open('modeldata/simple_wiki_freq.txt', 'w') as f:
    f.write('words')
    f.write('\n')
    for word_freq in most_common:
        f.write(word_freq[0])
        f.write('\n')

# Code for lang8 frequency and normalized it
words = re.findall(r'\w+', open('modeldata/lang8_entries.train').read().lower())
most_common=Counter(words).most_common()
# print(most_common)
# print(len(most_common))

with open('modeldata/lang8_freq.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(['words', 'frequency', 'normalized'])
    for word_freq in most_common:
        # 1105193 is the highest frequency of word which I already checked
        tsv_writer.writerow([word_freq[0], word_freq[1], "{:.7f}".format(word_freq[1]/1105193)])


