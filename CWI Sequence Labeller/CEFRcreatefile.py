import csv
import pandas as pd

# i = 0
# with open('cefr_levels_raw.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     with open('cefr_levels.tsv', 'wt') as out_file:
#         tsv_writer = csv.writer(out_file, delimiter='\t')
#         tsv_writer.writerow(['word', 'level'])
#         for row in reader:
#             # print(row)
#             if len(row) == 2 and row[0] != '' and row[1] != '' and row[0] != 'Base Word' and row[0] != '19/07/2021':
#                 i += 1
#                 print(row)
#                 if row[1] == 'A1':
#                     level = 1
#                 elif row[1] == 'A2':
#                     level = 2
#                 elif row[1] == 'B1':
#                     level = 3
#                 elif row[1] == 'B2':
#                     level = 4
#                 elif row[1] == 'C1':
#                     level = 5
#                 elif row[1] == 'C2':
#                     level = 6
#                 tsv_writer.writerow([row[0], level])
#         print(i)

all_levels = pd.read_table('cefr_levels.tsv', names=('word', 'level'))
def levels(word):
    word = ''.join(word.split()).lower()
    try:
        df = all_levels.loc[all_levels['word'] == word]
        print(df)
        # level = df.iloc[0]['level']
        level = min(df['level'])
        return level
    except:
        try:
            df = all_levels.loc[all_levels['word'] == word]
            level = df.iloc[0]['level']
            return level
        except:
            return 0


print(levels('as'))